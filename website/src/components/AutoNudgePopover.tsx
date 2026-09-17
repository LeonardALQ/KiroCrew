import { type ReactNode, useEffect, useId, useMemo, useRef, useState } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { Goal, Pause, Play, Radar, Save as SaveIcon, Square, X, Zap } from 'lucide-react'
import { Popover, PopoverTrigger, PopoverContent } from './ui/popover'
import { Btn } from './ui'
import ErrorNotice from './ErrorNotice'
import { cronJobsQuery } from '../api/cronJobsQuery'
import { runBelongsToSlot } from '../apps/workflows/runModel'
import { loadGoalDraft, saveGoalDraft, type GoalDraft } from '../utils/goalDrafts'
import { DRAFT_SAVE_DEBOUNCE_MS } from '../utils/draftConstants'

import { i18nT } from '../i18n/t'
import { fmtTimeNumeric } from '../i18n/format'
import { type AutoNudgeLoop, cycleText as loopCycleText, nextCycleText, AUTONUDGE_LOOPS_QUERY_KEY } from './autoNudgeLoop'
export type { AutoNudgeLoop } from './autoNudgeLoop'

interface Props {
  slotKey: string
  loop: AutoNudgeLoop | null
  open: boolean
  onOpenChange: (open: boolean) => void
  onChange: (loop: AutoNudgeLoop | null) => void
  /** Present when this editor is the popover's default view and a bounded monitor can still be armed. */
  onSetUpBoundedMonitor?: () => void
  /** Disable legacy-loop writes while leaving Stop available for stale state. Also renders the reason. */
  writeDisabled?: boolean
  /**
   * True when the slot's last turn ended interrupted (the composer is showing
   * Resume). The chip stops pulsing and turns warn-coloured: the loop is still
   * armed, but nothing is running until the user resumes or the next idle-timer
   * cycle fires, and a pulsing chip would claim active work for that whole gap.
   */
  interrupted?: boolean
  /** Shared composer trigger supplied by the structured-monitor compatibility shell. */
  trigger?: ReactNode
  /** Structured body supplied by that shell; omitted to render the legacy editor. */
  content?: ReactNode
}

/**
 * The kill-switch placeholder the server substitutes at FIRE time
 * (`render_nudge_message` in `dashboard/handlers/autonudge.py` replaces it with
 * the loop's `stop_sentinel_path`). It must travel to `/api/autonudge`
 * verbatim -- substituting it in the form would leave the server nothing to
 * replace -- so the textarea keeps the raw token and the help line under it
 * explains what the token becomes (#10458). `DEFAULT_MSG` below ends with this
 * exact spelling; a test pins that the template still carries it.
 */
export const STOP_FILE_TOKEN = '{{STOP_FILE}}'

/**
 * The service's `MANUAL_STOP_REASON` (`src/kiro_crew/autonudge.py`): the
 * `stopped_reason` a `PATCH active:false` records, and the one the revive logic
 * never auto-resumes. Spelled here because the frontend shares no constants
 * module with the service; `autoNudgeLoop.ts` lists the other codes.
 */
const MANUAL_STOP_REASON = 'manual'

const DEFAULT_MSG = `Your north star is in north_star.md, roadmap in roadmap.md, tasks in tasks.md. Pick the single highest-leverage next step toward the goal and execute it. Update tasks.md. Post a blocker ONCE if genuinely stuck. To halt the loop, create {{STOP_FILE}}`

/** One armed script cron owned by this chat slot. */
interface SlotWatch {
  id: string
  name: string
  schedule: string
  next_run_ts: number | null
}

export default function AutoNudgePopover({ slotKey, loop, open, onOpenChange, onChange, onSetUpBoundedMonitor, writeDisabled = false, interrupted = false, trigger, content }: Props) {
  // `||` (not `??`) is deliberate on the loop tier: it preserves the fallback
  // so a loop with idle_secs/max_cycles of 0 or an empty message still shows
  // the 60 / 0 / default template rather than a bare 0 / "".
  const [message, setMessage] = useState(() => loop?.message || DEFAULT_MSG)
  // Idle-seconds and max-cycles are held as RAW STRINGS while the popover is
  // open so every edit (including a fully-cleared field or a transient "") is
  // allowed as-typed. Coercing to a number on each keystroke would snap a
  // backspaced-to-empty field straight back to its default and prevent removing
  // the leading digit. The string is parsed
  // into a number only when the field commits (blur / save); an empty or
  // unparseable value falls back to the field default — 60 idle, 0 cycles.
  const [idleInput, setIdleInput] = useState(() => String(loop?.idle_secs || 60))
  const [maxCyclesInput, setMaxCyclesInput] = useState(() => String(loop?.max_cycles || 0))
  const [saving, setSaving] = useState(false)
  /* Two-step on the clear only. The erase is irreversible and sits beside the
     primary CTA, so one press asks and the second performs. */
  const [confirmClear, setConfirmClear] = useState(false)
  const [error, setError] = useState('')
  // Watches armed on this slot, read through the SHARED `cron-jobs` query rather
  // than a private fetch. That key is invalidated by the websocket hook, so a
  // watch deleted or paused elsewhere disappears from an open popover instead of
  // lingering until it is reopened -- and the request dedupes with the other
  // consumer of the same key. `enabled: open` keeps a zero-token watch from
  // costing a request on every chat render just to say "still nothing".
  const queryClient = useQueryClient()
  const { data: cronJobs, isError: watchesFailed, refetch: refetchWatches } = useQuery({
    ...cronJobsQuery,
    enabled: open && content === undefined,
  })

  const watches: SlotWatch[] = useMemo(() => {
    const rows: unknown[] = Array.isArray(cronJobs) ? cronJobs : []
    return rows
      .filter((j): j is Record<string, unknown> => !!j && typeof j === 'object')
      .filter(j => {
        // One ownership rule, one spelling. `runBelongsToSlot` already maps a
        // session_key onto a chat slot against the same backend convention
        // (`dashboard:<slotKey>`); a second inline predicate here would drift
        // from it the day that key format moves.
        if (!runBelongsToSlot(typeof j.session_key === 'string' ? j.session_key : '', slotKey)) {
          return false
        }
        // A watch is a SCRIPT cron: it runs a Python callable and never reaches a
        // model. A message-only cron on this slot is an ordinary reminder that
        // DOES wake the agent, so it does not belong under a heading that
        // promises zero tokens.
        return typeof j.script === 'string' && !!j.script && j.enabled !== false
      })
      .map(j => ({
        id: String(j.id ?? ''),
        name: String(j.name ?? ''),
        schedule: String(j.schedule ?? ''),
        next_run_ts: typeof j.next_run_ts === 'number' ? j.next_run_ts : null,
      }))
  }, [cronJobs, slotKey])

  const parseIdle = (s: string) => parseInt(s, 10) || 60
  const parseCycles = (s: string) => parseInt(s, 10) || 0

  // Only a genuine user edit should persist a draft. Seeding from the live loop
  // or restoring a remembered draft on open must NOT re-write the store (doing
  // so would reset the slot's TTL / LRU position on a mere view, and could
  // mirror a live loop's config into the user-draft store). `hasEdited` gates
  // the persist so it fires on real onChange edits only.
  const hasEdited = useRef(false)
  // Latest field values, kept current every render so the close-flush below
  // (which runs from a stable handler) can read them.
  const latest = useRef({ slotKey, message, idleInput, maxCyclesInput, loop })
  latest.current = { slotKey, message, idleInput, maxCyclesInput, loop }

  // Compute the draft to persist for the current field state, or null to drop
  // the slot: the blank / pristine-default case stores nothing so an emptied or
  // untouched popover never pins the template. (Only reached when no loop is
  // running — a live loop is authoritative and its config is never mirrored
  // into the user-draft store; persistence is skipped entirely while a loop is
  // present.)
  function draftToPersist(s: typeof latest.current): GoalDraft | null {
    const idleSecs = parseIdle(s.idleInput)
    const maxCycles = parseCycles(s.maxCyclesInput)
    const isPristineDefault = s.message === DEFAULT_MSG && idleSecs === 60 && maxCycles === 0
    return isPristineDefault ? null : { message: s.message, idleSecs, maxCycles }
  }

  /* A pending confirmation belongs to the record the reader was LOOKING at. The
     popover re-renders from websocket state without closing, so another tab can
     swap that record underneath it -- edit and restart the same loop id, then a
     cycle cap (max_cycles=1 fires once) stops it again -- and the primed press
     would erase a goal the confirmation never described. The intent guard does
     not catch it: the record is inactive at render AND at press, so the server
     sees no mismatch. Keyed on identity, state and the text itself, since the
     text is what the erase destroys and drafts are not persisted while a loop
     exists. */

  useEffect(() => {
    setConfirmClear(false)
  }, [loop?.id, loop?.active, loop?.message])

  // Seed/restore fields on each open (rising edge). A live loop is the
  // authoritative source; otherwise the last per-slot draft is restored.
  // One read seeds all three fields. Runs in an effect (not render) so the
  // render itself performs no storage read/write.
  useEffect(() => {
    if (!open) return
    hasEdited.current = false
    setError('')
    // A pending confirmation must not survive a close: reopening later would
    // put a primed erase under the next press.
    setConfirmClear(false)
    if (loop) {
      // `||` (not `??`) is deliberate: a loop with idle_secs/max_cycles of 0
      // or an empty message shows the 60 / 0 / default template.
      setMessage(loop.message || DEFAULT_MSG)
      setIdleInput(String(loop.idle_secs || 60))
      setMaxCyclesInput(String(loop.max_cycles || 0))
    } else {
      const remembered = loadGoalDraft(slotKey)
      setMessage(remembered ? remembered.message : DEFAULT_MSG)
      setIdleInput(String(remembered ? remembered.idleSecs : 60))
      setMaxCyclesInput(String(remembered ? remembered.maxCycles : 0))
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- open-edge seed only; loop/slotKey are read fresh each open
  }, [open])

  // Flush a pending debounced edit synchronously when the popover closes OR
  // unmounts while open, so edits within the last DRAFT_SAVE_DEBOUNCE_MS
  // window aren't lost. Effect cleanup covers both paths.
  useEffect(() => {
    if (!open) return
    return () => {
      if (!hasEdited.current || latest.current.loop) return
      saveGoalDraft(latest.current.slotKey, draftToPersist(latest.current))
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- stable cleanup reading the latest ref
  }, [open])

  // Persist edits per slot, debounced with the same DRAFT_SAVE_DEBOUNCE_MS as
  // chat drafts so a long goal doesn't drive a synchronous localStorage write on
  // every keystroke. Skips until the user actually edits a field (so opening the
  // popover or the open-restore setState above never writes).
  useEffect(() => {
    if (!open || !hasEdited.current || loop) return
    const timer = setTimeout(() => saveGoalDraft(slotKey, draftToPersist(latest.current)), DRAFT_SAVE_DEBOUNCE_MS)
    return () => clearTimeout(timer)
    // eslint-disable-next-line react-hooks/exhaustive-deps -- `draftToPersist` is a pure transform of the ref snapshot it is handed, redeclared each render, so its identity carries no information the deps above miss. Depending on it would restart the debounce timer on every unrelated re-render — the coalescing this effect exists for.
  }, [open, slotKey, message, idleInput, maxCyclesInput, loop])

  /** A loop somebody PAUSED -- inactive, with the reason a manual pause records
   *  -- as opposed to one a bound spent (`cycle_cap`, `runtime_budget`,
   *  `approval_stalled`), a tool tombstoned (`autonudge_stop`), or one whose
   *  reason is unknown here. Only this state reads "Paused" and labels its Play
   *  "Resume loop"; everything else inactive reads "Stopped" and its Play reads
   *  "Start loop", with Stop becoming the erase of the retained record. Strict
   *  equality on purpose: an absent reason means "not known here", and the safe
   *  reading of unknown is the non-resumable one. */
  const pausedManually = !!loop && !loop.active && loop.stopped_reason === MANUAL_STOP_REASON

  /** The three fields as the form holds them right now. Parsed from the raw
   *  strings (not a committed number state) so a value typed and then pressed
   *  without an intervening blur is still captured. */
  function formFields() {
    return { message, idle_secs: parseIdle(idleInput), max_cycles: parseCycles(maxCyclesInput) }
  }

  const JSON_HEADERS = { 'Content-Type': 'application/json' }

  /** Save, on a RUNNING loop only: persist the three fields, nothing else.
   *
   *  Never carries `active`. Starting, resuming and reviving belong to Play,
   *  so a save of edited fields leaves the loop running exactly as it was; the
   *  field would be a no-op while the loop is still running and exactly wrong
   *  when it is not -- another tab's Pause, or a spent bound, can land between
   *  this render and the press, and a save carrying `active: true` would then
   *  revive the loop (`update` clears the stop reason and re-arms the timer)
   *  as a side effect of editing text. The one control on this surface that
   *  CLOSES the popover on success, as it always did: it changes nothing the
   *  popover could show, so closing is its confirmation. Every control that
   *  fires or changes the run state stays open instead (see `writeThenFire`). */
  async function save() {
    if (!loop || writeDisabled) return
    setSaving(true)
    setError('')
    try {
      const resp = await fetch(`/api/autonudge/${loop.id}`, { method: 'PATCH', headers: JSON_HEADERS, body: JSON.stringify(formFields()) })
      const data = await resp.json()
      if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`)
      onChange(data.loop)
      onOpenChange(false)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setSaving(false)
    }
  }

  /** Stop a live (running or paused) loop, or clear an already-stopped one. */
  async function stop(intent: 'stop' | 'clear') {
    if (!loop) return
    setSaving(true)
    try {
      // The INTENT travels with the request, because the server otherwise
      // decides what this verb means from the record's state at arrival time:
      // a press meant as "Stop loop" on a popover rendered moments earlier
      // would silently ERASE a record that went terminal in between. The server
      // 409s on a mismatch instead, and the popover surfaces that. It is the
      // LABEL the user pressed, supplied by the button, never re-derived from
      // `loop.active` here: a paused loop is inactive yet its control is Stop.
      const resp = await fetch(`/api/autonudge/${loop.id}?intent=${intent}`, { method: 'DELETE' })
      if (!resp.ok) {
        // Parse JSON body for server-supplied error (e.g. 503 when feature disabled).
        // Only on error path: a successful DELETE may return 204 No Content.
        const data = await resp.json().catch(() => ({}))
        throw new Error(data.error || `HTTP ${resp.status}`)
      }
      onChange(null)
      onOpenChange(false)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setSaving(false)
    }
  }

  /** Pause the loop IN PLACE (`active: false`).
   *
   *  No new backend: `PATCH /api/autonudge/{id}` already accepts `active`, and
   *  the service records `stopped_reason: "manual"` on a pause
   *  (`_update_unserialized` in `src/kiro_crew/autonudge.py`) -- the reason that
   *  tells the paused state from a stopped one when the record comes back. The
   *  body carries ONLY `active`: a pause is not a save, so whatever sits in the
   *  fields stays unsaved and un-sent.
   *
   *  Does NOT close the popover, for the same reason `triggerNow` does not: the
   *  textarea may hold an unsaved edit with no dirty guard, and the outcome is
   *  visible in place -- the countdown gives way to "Paused" and this control's
   *  slot flips to Play. */
  async function pause() {
    if (!loop || writeDisabled) return
    setSaving(true)
    setError('')
    try {
      const resp = await fetch(`/api/autonudge/${loop.id}`, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ active: false }) })
      const data = await resp.json().catch(() => ({}))
      if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`)
      onChange(data.loop)
      // The full-registry readers (the Crew Members patrol block) must not keep
      // showing a countdown for a loop that just paused; same reconciliation
      // `triggerNow` performs.
      void queryClient.invalidateQueries({ queryKey: AUTONUDGE_LOOPS_QUERY_KEY })
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setSaving(false)
    }
  }

  /** WRITE the loop's record, hand the returned record up, then FIRE it.
   *
   *  Every control on this surface that fires goes through here, because every
   *  fire implicitly saves ("any Trigger implicitly calls the Save logic"):
   *  Play on a paused or stopped loop writes the form with `active: true` and
   *  fires (`resumeNow`); Play with no loop creates the loop from the form and
   *  fires the id the server returned (`startNow`); Trigger on a running loop
   *  writes the form without `active` and fires (`triggerNow`). The flow this
   *  buys is pause -> edit the goal, interval or cap -> press Play, with no
   *  separate Save, and a cap raised in the form travelling with the revive
   *  instead of the loop re-stopping a tick later on the spent cap.
   *
   *  The write comes FIRST because `fire_now` fires whatever the loop holds and
   *  refuses an inactive loop with 409: firing before the write would fire the
   *  old goal, or nothing. The two legs are NOT a transaction, on purpose. A
   *  refused write fires nothing -- there is nothing to fire. A write that
   *  lands followed by a refused fire (409 while a turn is in flight, or
   *  mid-fire) leaves the loop written -- saved, resumed or created -- with the
   *  refusal in the inline notice: the user asked for two things and got one,
   *  and rolling the write back would turn a refused shortcut into an undone
   *  edit or an undone pause.
   *
   *  STAYS OPEN, in every case, and so do Pause and Play. The outcome is
   *  visible in place -- the lane flips (a created or resumed loop shows Stop |
   *  Trigger Pause Save, a schedule line reads due), and a refusal needs
   *  somewhere to land. Only Save closes the popover (see `save`): it is the
   *  one control whose result the popover cannot show. */
  async function writeThenFire(write: () => Promise<Response>) {
    if (writeDisabled) return
    setSaving(true)
    setError('')
    try {
      const resp = await write()
      const data = await resp.json().catch(() => ({}))
      if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`)
      onChange(data.loop)
      // The full-registry readers (the Crew Members patrol block) must not keep
      // a stale copy of a record this write just changed.
      void queryClient.invalidateQueries({ queryKey: AUTONUDGE_LOOPS_QUERY_KEY })
      // The fire needs the record's id: for a create it is the one the server
      // just minted, so it is read off the response rather than the closure.
      if (data.loop?.id) await fireNow(String(data.loop.id))
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setSaving(false)
    }
  }

  /** The fire leg: bring the loop's next cycle forward to now.
   *
   *  Sends NO body -- the nudge fired is whatever the loop holds, read
   *  server-side, which the write a moment earlier made the form's text.
   *  The route returns the loop UNCHANGED: the server-side deadline write was
   *  removed because it could not be made durable without a suspension point
   *  that raced several lock-free writers. Rendering the response verbatim
   *  would therefore leave the countdown showing the very cycle this press
   *  superseded -- the one visible confirmation a press has. So the armed
   *  deadline is set here instead. Not a fiction: the cycle IS armed to run
   *  now, and the delivery's `autonudge_state` frame reconciles the shared
   *  cache moments later. Throws on refusal so `writeThenFire` lands it in
   *  the inline notice. */
  async function fireNow(loopId: string) {
    const resp = await fetch(`/api/autonudge/${loopId}/fire`, { method: 'POST' })
    const data = await resp.json().catch(() => ({}))
    if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`)
    onChange({ ...data.loop, next_due_ts: Date.now() / 1000 })
    void queryClient.invalidateQueries({ queryKey: AUTONUDGE_LOOPS_QUERY_KEY })
  }

  /** Play on a paused or stopped loop: save the form, resume (or revive), fire.
   *  `active: true` clears the stop reason and re-arms the timer on a fresh full
   *  countdown (`_update_unserialized`), one interval too late for a user who
   *  just pressed Play -- hence the fire leg. */
  function resumeNow() {
    if (!loop) return
    return writeThenFire(() =>
      fetch(`/api/autonudge/${loop.id}`, { method: 'PATCH', headers: JSON_HEADERS, body: JSON.stringify({ ...formFields(), active: true }) }))
  }

  /** Play with no loop: create it from the form (today's POST), then fire the
   *  new id so the first nudge goes out now rather than after `idle_secs`. */
  function startNow() {
    return writeThenFire(() =>
      fetch('/api/autonudge', { method: 'POST', headers: JSON_HEADERS, body: JSON.stringify({ slot_key: slotKey, ...formFields() }) }))
  }

  /** Trigger on a running loop: save the form (never `active` -- a running
   *  loop's write must not be able to revive one another tab paused between
   *  render and press), then fire. */
  function triggerNow() {
    if (!loop) return
    return writeThenFire(() =>
      fetch(`/api/autonudge/${loop.id}`, { method: 'PATCH', headers: JSON_HEADERS, body: JSON.stringify(formFields()) }))
  }

  // ── Countdown to the next trigger (#6482) ──
  // The 1s ticker runs only while the popover is OPEN (review finding: a
  // closed-but-armed loop must not re-render the toolbar button every second
  // all day). The hover affordance needs no ticker: a native title tooltip
  // snapshots at hover-start, so the trigger's onMouseEnter/onFocus refresh
  // nowTs once, which is exactly the freshness a tooltip glance can show.
  const ticking = open && !!loop?.active && (loop.next_due_ts || 0) > 0
  const [nowTs, setNowTs] = useState(() => Date.now() / 1000)
  useEffect(() => {
    if (!ticking) return
    setNowTs(Date.now() / 1000)
    const timer = setInterval(() => setNowTs(Date.now() / 1000), 1000)
    return () => clearInterval(timer)
  }, [ticking])
  const refreshNow = () => setNowTs(Date.now() / 1000)
  /** Hover/popover line for the next trigger, or '' when no active loop — the
   *  shared deadline-preserving reading (see `nextCycleText`). */
  const countdownText = nextCycleText(loop, nowTs)
  /** The tooltip only carries a REAL deadline signal (counting or due) — the
   *  "not yet scheduled" placeholder is popover-only, so an armed-but-unscheduled
   *  loop keeps the plain "Goal active (cycle N)" title. */
  const titleCountdown = loop?.active && (loop.next_due_ts || 0) > 0 ? countdownText : ''
  /** Cycle readout for the chip, tooltip and popover header ("3/24", or a
   *  bare "3" under an infinite cap). Interpolated as the {{cycle}} VALUE of
   *  the existing strings, so no catalogue text changes. Unlike the countdown
   *  this is safe in aria-label: it changes once per cycle, not once per
   *  second. */
  /** Whether a cycle is ALREADY armed to run. Derived from the same countdown
   *  the schedule line renders, so the button and the text can never disagree. */
  const cycleAlreadyDue =
    countdownText === i18nT('components.autoNudgePopover.next_cycle_due')
  /** Help line under the goal textarea while it carries the raw kill-switch
   *  token; '' otherwise. See the JSX comment at the render site (#10458). */
  const stopFileHelp = message.includes(STOP_FILE_TOKEN)
    ? loop && loop.stop_sentinel_path === ''
      ? i18nT('components.autoNudgePopover.stop_file_help_none', { token: STOP_FILE_TOKEN })
      : i18nT('components.autoNudgePopover.stop_file_help', { token: STOP_FILE_TOKEN })
    : ''
  const stopFileHelpId = useId()

  const cycleText = loopCycleText(loop)

  return (
    <Popover open={open} onOpenChange={onOpenChange}>
      {trigger ? <PopoverTrigger asChild>{trigger}</PopoverTrigger> : (
      <PopoverTrigger asChild>
        <button
          className={`h-8 px-2 rounded-lg text-[12px] font-mono flex items-center gap-1 cursor-pointer transition-all bg-transparent border-none shrink-0 whitespace-nowrap ${
            loop?.active
              ? interrupted
                ? 'text-warn hover:text-warn hover:bg-warn/10'
                : 'text-accent hover:text-accent hover:bg-accent/10 animate-pulse'
              : 'text-muted hover:text-text hover:bg-bg-hover'
          }`}
          title={loop?.active ? `${interrupted ? i18nT('components.autoNudgePopover.goal_interrupted_cycle', { cycle: cycleText }) : i18nT('components.autoNudgePopover.goal_active_cycle', { cycle: cycleText })}${titleCountdown ? ` · ${titleCountdown}` : ''}` : i18nT('components.autoNudgePopover.set_a_goal')}
          // The countdown stays OUT of aria-label (review finding): a
          // per-second label change re-announces the button to screen readers.
          aria-label={loop?.active ? (interrupted ? i18nT('components.autoNudgePopover.goal_interrupted_cycle', { cycle: cycleText }) : i18nT('components.autoNudgePopover.goal_active_cycle', { cycle: cycleText })) : i18nT('components.autoNudgePopover.set_a_goal')}
          onMouseEnter={refreshNow}
          onFocus={refreshNow}
        >
          <Goal size={16} className="shrink-0" />
          {loop?.active && loop.cycle_count > 0 ? cycleText : null}
        </button>
      </PopoverTrigger>
      )}
      {content ?? <PopoverContent
        side="top"
        align="start"
        /* Viewport-capped rather than a pinned 420px: at the 320px floor a fixed
           width pushes this panel -- and the right-aligned action below -- past the
           usable viewport. Written as a max so there is no `md:` counterpart to keep
           in sync: 420px is simply the ceiling, and a phone gets the width it has. */
        className="w-[min(calc(100vw-1rem),26.25rem)] max-h-[min(80vh,42rem)] overflow-y-auto p-4 text-[12px]"
      >
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2 font-medium text-text">
            <Goal size={14} className={loop?.active ? 'text-accent' : 'text-muted'} />
            {i18nT('components.autoNudgePopover.set_a_goal')}
            {loop?.active && <span className="text-muted text-[11px]">{i18nT('components.autoNudgePopover.cycle')} {cycleText}</span>}
          </div>
          <button aria-label={i18nT('components.autoNudgePopover.close')} onClick={() => onOpenChange(false)} className="text-muted hover:text-text bg-transparent border-none cursor-pointer">
            <X size={14} />
          </button>
        </div>
        {onSetUpBoundedMonitor ? (
          <>
            {/* An OFFER, not a way back: this editor is the view the popover
                opens on, so a reader arriving here has no bounded monitor
                behind them to return to. Hence a Radar glyph rather than a left
                arrow, and a label naming the SUBJECT that surface takes -- it
                accepts a pull request URL and nothing else, so a label reading
                only "bounded monitor" walks a reader with any other goal into
                a form whose one field they cannot fill.
                Underlined without hovering, because this is now the ONLY route
                to the monitor: a usability reader could not tell 11px muted
                text was clickable at all, and a hover-only affordance is
                invisible on a touch viewport. */}
            <button
              type="button"
              onClick={onSetUpBoundedMonitor}
              className="mb-2 inline-flex items-center gap-1 border-none bg-transparent p-0 text-[11px] text-muted underline cursor-pointer hover:text-text"
            >
              <Radar size={13} className="lucide-inline" aria-hidden />
              {i18nT('components.sessionAutomationPopover.set_up_bounded_monitor')}
            </button>
            {/* Warn-coloured, unchanged from when this form was opt-in. Muting
                it read better to the author and worse to review: on the view
                every reader now lands on, this sentence is the only cost cue
                the surface carries, and dropping its colour weakened that cue
                in the same change that made the surface the default. */}
            <p role="note" className="mb-2 rounded-md border border-warn/30 bg-warn-subtle px-2 py-1.5 text-[11px] text-warn-fg">
              {i18nT('components.sessionAutomationPopover.legacy_notice')}
            </p>
          </>
        ) : null}
        <p className="text-muted text-[11px] mb-3 leading-relaxed">{i18nT('components.autoNudgePopover.give_the_agent_a_goal_and_it_will_keep_working_t')}</p>

        {watchesFailed && (
          <div className="flex items-center justify-between gap-2 mb-3">
            {/* No hand-off: the popover holds the unsaved goal message, idle and max-cycle inputs.
                Retry is the recovery path, as on every sibling load-failure notice. */}
            <ErrorNotice
              variant="inline"
              testId="auto-nudge-watches-error"
              message={i18nT('components.autoNudgePopover.watches_load_failed')}
            />
            <button
              type="button"
              onClick={() => { void refetchWatches() }}
              className="px-2 py-0.5 rounded border border-border text-[11px] text-muted hover:text-text bg-transparent cursor-pointer shrink-0"
            >
              {i18nT('components.autoNudgePopover.retry')}
            </button>
          </div>
        )}

        {watches.length > 0 && (
          <div className="border border-border rounded p-2 mb-3">
            <div className="text-text text-[11px] font-medium mb-1">
              {i18nT('components.autoNudgePopover.watches_title')}
            </div>
            <ul className="list-none p-0 m-0 mb-1">
              {watches.map(w => (
                <li key={w.id} className="text-muted text-[11px] leading-relaxed">
                  <span className="text-text">{w.name}</span>
                  {w.schedule && <span> · {w.schedule}</span>}
                  {w.next_run_ts && (
                    <span> · {i18nT('components.autoNudgePopover.watches_next')} {fmtTimeNumeric(w.next_run_ts)}</span>
                  )}
                </li>
              ))}
            </ul>
            <div className="text-muted text-[11px] leading-relaxed">
              {i18nT('components.autoNudgePopover.watches_note')}
            </div>
          </div>
        )}

        {/* The reason the fields below are dead. `writeDisabled` alone renders a
            form a crew/member reader cannot use and does not say why: the
            explanation used to live on the bounded view, which was the default,
            and making the goal loop the default left the disabled form with no
            reason attached.
            Rendered from the boolean rather than through a `reason` prop. The
            prop was a one-consumer generalization -- its single caller passed
            one constant gated on this same condition -- and the rationale for
            it ("the editor knows nothing about session modes") was already
            false, since this component reads `sessionAutomationPopover` strings
            two lines up. A second reason for disabling writes would need the
            reason back as a parameter; there is exactly one today. */}
        {writeDisabled ? (
          <p
            role="status"
            data-testid="auto-nudge-write-disabled-reason"
            className="mb-3 rounded-md border border-border bg-bg px-2 py-1.5 text-[11px] leading-relaxed text-muted"
          >
            {i18nT('components.sessionAutomationPopover.session_mode_unavailable')}
          </p>
        ) : null}

        <div className="text-muted text-[11px] mb-1">{i18nT('components.autoNudgePopover.goal_description')}</div>
        <textarea
          aria-label={i18nT('components.autoNudgePopover.goal_description')}
          value={message}
          disabled={writeDisabled}
          onChange={e => { hasEdited.current = true; setMessage(e.target.value) }}
          rows={6}
          className="w-full bg-bg border border-border rounded p-2 text-[12px] font-mono resize-y mb-3 text-text"
          placeholder={i18nT('components.autoNudgePopover.describe_what_you_want_the_agent_to_accomplish')}
          aria-describedby={stopFileHelp ? stopFileHelpId : undefined}
        />
        {stopFileHelp ? (
          /* Display-only explanation of the raw token above (#10458). The
             textarea keeps `{{STOP_FILE}}` because the server substitutes it
             when each nudge is sent; only the human reading the form needed
             telling what it turns into. Shown while the goal text carries the
             token, so a custom goal without it gets no orphan help line. The
             empty-sentinel arm reads the ARMED loop's record: a loop that
             carries an explicitly empty `stop_sentinel_path` has nothing to
             substitute, so the honest line is that the token goes out blank
             and Stop loop is the way to halt it. The path itself is never
             rendered: the websocket frame withholds it and this surface has no
             owner gate. */
          <p id={stopFileHelpId} className="text-muted text-[11px] leading-relaxed -mt-2 mb-3">
            {stopFileHelp}
          </p>
        ) : null}

        <div className="flex flex-col gap-3 mb-3 sm:flex-row">
          <div className="flex-1">
            <div className="text-muted text-[11px] mb-1">{i18nT('components.autoNudgePopover.seconds_between_nudges')}</div>
            <input
              type="number"
              aria-label={i18nT('components.autoNudgePopover.seconds_between_nudges')}
              min={15}
              max={86400}
              value={idleInput}
              disabled={writeDisabled}
              onChange={e => { hasEdited.current = true; setIdleInput(e.target.value) }}
              onBlur={() => setIdleInput(String(parseIdle(idleInput)))}
              className="w-full bg-bg border border-border rounded px-2 py-1 text-[12px] text-text"
            />
          </div>
          <div className="flex-1">
            <div className="text-muted text-[11px] mb-1">{i18nT('components.autoNudgePopover.max_cycles_0')}</div>
            <input
              type="number"
              aria-label={i18nT('components.autoNudgePopover.max_cycles_0_infinite')}
              min={0}
              value={maxCyclesInput}
              disabled={writeDisabled}
              onChange={e => { hasEdited.current = true; setMaxCyclesInput(e.target.value) }}
              onBlur={() => setMaxCyclesInput(String(parseCycles(maxCyclesInput)))}
              className="w-full bg-bg border border-border rounded px-2 py-1 text-[12px] text-text"
            />
          </div>
        </div>

        {/* The SCHEDULE line: last fire, the countdown (or "Paused" / "Stopped"),
            and the help sentence naming the exits. Text only -- every control
            lives in the single action row below (operator ruling, 2026-09-17),
            so nothing here relocates under a cursor when the countdown flips to
            the longer "due" wording. Rendered for every loop; with no loop
            there is no schedule to read. */}
        {loop && (
          <div className="flex flex-col items-start gap-1 mb-3" data-testid="auto-nudge-schedule">
            <div className="text-muted text-[11px]">
              {i18nT('components.autoNudgePopover.last_fire')} {loop.last_fire_ts ? fmtTimeNumeric(loop.last_fire_ts) : i18nT('components.autoNudgePopover.never')}
              {countdownText && <span> · {countdownText}</span>}
              {/* "Paused" takes the countdown's slot: a paused loop holds no
                  schedule (`next_due_ts` is cleared), so the state IS the
                  schedule reading. Says "Paused" -- the word the Stopped branch
                  below deliberately avoids -- because here it is true: the Play
                  in the row resumes this same record where it left off. */}
              {pausedManually && (
                <span> · <span data-testid="auto-nudge-loop-paused-manually">{i18nT('components.autoNudgePopover.loop_paused')}</span></span>
              )}
            </div>
            {pausedManually ? (
              /* Names both exits, as the stopped help does, and says what Play
                 does beyond resuming: it also sends a nudge now. Stop removes
                 the record for good and nothing else on the surface says so. */
              <span data-testid="auto-nudge-paused-help" className="text-muted text-[11px]">
                {i18nT('components.autoNudgePopover.paused_help')}
              </span>
            ) : !loop.active ? (
              /* Says WHY there is no countdown, rather than leaving a gap. A
                 blind reader of the stopped screenshot could not tell it was the
                 same loop at all, and an inactive loop otherwise looks identical
                 to an active one whose countdown failed to render -- the state is
                 the reason for the absence, so it belongs in the space the
                 absence leaves.
                 Reads "Stopped", not "Paused": the Stop in the row removes this
                 record for good, and a blind reader took "Paused" as "it
                 remembers where it left off" -- a resumable-sounding status next
                 to an erase control is the mixed message a UX review blocked on.
                 That reasoning still holds here, and it is exactly why "Paused"
                 is reserved for the ONE inactive state that IS resumable in
                 place: a loop with the manual-pause reason, handled above. A
                 spent bound, a tool's tombstone and an unknown reason all land
                 here and stay "Stopped".
                 The help line under it names both exits, because the erase is
                 irreversible and nothing else on the surface says so. */
              <div className="flex flex-col items-start gap-0.5">
                <span
                  data-testid="auto-nudge-loop-paused"
                  className="text-muted text-[11px] shrink-0"
                >
                  {i18nT('components.autoNudgePopover.loop_stopped')}
                </span>
                {/* While confirming, this line must not keep naming the two
                    controls that just left the row -- a blind reader looked for
                    the "Start loop" it describes and could not find it -- and
                    the confirmation row itself renders no question. So the help
                    line BECOMES the question for that state. */}
                <span data-testid="auto-nudge-stopped-help" className="text-muted text-[11px]">
                  {confirmClear
                    ? i18nT('components.autoNudgePopover.clear_goal_question')
                    : i18nT('components.autoNudgePopover.stopped_help')}
                </span>
              </div>
            ) : null}
          </div>
        )}

        {/* No hand-off: the popover holds the unsaved goal message, idle and max-cycle inputs. */}
        <ErrorNotice
          variant="inline"
          className="mb-2"
          testId="auto-nudge-error"
          message={error}
          onDismiss={() => setError('')}
        />

        {/* THE ACTION ROW -- every control on ONE lane.
            Layout ruled by the product owner on 2026-09-17: Stop isolated left
            as the destructive control, the loop's transport and save controls
            clustered right; an overflow menu was explicitly rejected because
            every control must stay visible. This knowingly exceeds
            `max-two-buttons-per-row` (website/AUTOSDE.yaml:230): a running
            loop puts four controls on the lane, three of them in the right
            cluster. The finding that rule draws is answered by the ruling, not
            by a rework -- the PR body's Other section carries the same text.
            The lane by state (Stop pinned left, free space, then the cluster):
              RUNNING            [Stop] ........ [Trigger] [Pause] [Save]
              PAUSED (manual)    [Stop] ........ [Play]            accent
              STOPPED (a bound,  [Stop] ........ [Play]            accent;
                a tombstone)                     Stop = erase behind a confirm
              NO LOOP                             [Play]            accent, alone
            Every control that fires SAVES THE FORM FIRST ("any Trigger
            implicitly calls the Save logic", see `writeThenFire`): Play on a
            paused or stopped loop writes the form with `active: true` and
            fires; Play with no loop creates the loop from the form and fires
            it; Trigger on a running loop writes the form and fires. So an
            inactive loop needs no Save -- pause, edit, press Play -- and Save
            alone exists only while the loop runs. Play's label says which:
            "Resume loop and nudge now" for the paused record, "Start loop and
            nudge now" otherwise. Trigger exists only while running, because
            the server refuses to fire an inactive loop (409) and Play already
            fires on the way back. Every control is an icon button with
            aria-label AND title (`icon-buttons-need-labels`): the glyph carries
            no text of its own. */}
        {loop && confirmClear ? (
          /* The erase confirm REPLACES the lane, as the monitor surface's
             identical confirm does: the choice should hold the reader's whole
             attention, and the help line above has become the question. Each
             label restates the ACTION and its object rather than answering a
             question the row does not render: read alone, "Yes" says nothing
             about what is being cleared. */
          <div className="flex gap-2 justify-end">
            <Btn type="button" onClick={() => setConfirmClear(false)} disabled={saving}>
              {i18nT('components.autoNudgePopover.cancel')}
            </Btn>
            <Btn type="button" danger onClick={() => stop('clear')} disabled={saving}>
              {i18nT('components.autoNudgePopover.clear_goal_for_good')}
            </Btn>
          </div>
        ) : (
          <div className="flex items-center gap-2" data-testid="auto-nudge-actions">
            {loop && (
              /* Stop, alone on the left. On a running or paused loop it is the
                 single-press stop, as before -- a paused loop is the same live
                 goal with its countdown suspended, not a stopped record. On an
                 already-stopped loop there is nothing left to stop: the press
                 REMOVES the retained record, which is what frees the slot to
                 watch something else, so it is labelled for that ("Stop loop"
                 read as a no-op) and asks first, because the erase is
                 irreversible. `Btn danger` colours it unconditionally rather
                 than on :hover, which a touch viewport never produces: the
                 glyph alone does not say "removes". Reachable while writes are
                 disabled, so stale state can always be cleared. */
              <Btn
                type="button"
                danger
                className="!px-1.5"
                aria-label={loop.active || pausedManually ? i18nT('components.autoNudgePopover.stop_loop') : i18nT('components.autoNudgePopover.clear_stopped_goal')}
                title={loop.active || pausedManually ? i18nT('components.autoNudgePopover.stop_loop') : i18nT('components.autoNudgePopover.clear_stopped_goal')}
                onClick={() => (loop.active || pausedManually ? stop('stop') : setConfirmClear(true))}
                disabled={saving}
              >
                <Square size={14} fill="currentColor" aria-hidden />
              </Btn>
            )}
            {/* `ml-auto` pushes the cluster right whether or not Stop is
                rendered, so the no-loop Play sits where the cluster sits on a
                loop. Every write-gated control shares one disabled rule: busy,
                writes disabled, or an empty goal -- each of them sends the
                form, and an empty goal is not a goal. */}
            <div className="ml-auto flex items-center gap-2" data-testid="auto-nudge-loop-controls">
              {loop?.active ? (
                <>
                  <Btn
                    type="button"
                    className="!px-1.5"
                    aria-label={i18nT('components.autoNudgePopover.trigger_nudge')}
                    title={i18nT('components.autoNudgePopover.trigger_nudge')}
                    onClick={triggerNow}
                    /* Also disabled once a cycle is already due, which is what a
                       successful press produces. Before this the button
                       re-enabled unchanged, so the press acknowledged itself
                       only through the schedule line's wording -- a usability
                       reader would not press it a second time because they
                       could not tell whether that would double the nudge or do
                       nothing (it does nothing: the cycle is already armed).
                       The disabled state answers that without a new string; an
                       edit made while due still has Save beside it. */
                    disabled={saving || writeDisabled || cycleAlreadyDue || !message.trim()}
                  >
                    <Zap size={14} aria-hidden />
                  </Btn>
                  {/* A write (`active: false` and nothing else), so it follows
                      Save's `writeDisabled` gate -- but not the empty-goal gate:
                      a pause sends no fields. */}
                  <Btn
                    type="button"
                    className="!px-1.5"
                    aria-label={i18nT('components.autoNudgePopover.pause_loop')}
                    title={i18nT('components.autoNudgePopover.pause_loop')}
                    onClick={pause}
                    disabled={saving || writeDisabled}
                  >
                    <Pause size={14} aria-hidden />
                  </Btn>
                  {/* Save as an icon, in the accent the text button wore: the
                      plain persist of the fields, running loop only -- on an
                      inactive loop Play does the saving on the way back. */}
                  <Btn
                    type="button"
                    primary
                    className="!px-1.5"
                    aria-label={i18nT('components.autoNudgePopover.save')}
                    title={i18nT('components.autoNudgePopover.save')}
                    onClick={save}
                    disabled={saving || writeDisabled || !message.trim()}
                  >
                    <SaveIcon size={14} aria-hidden />
                  </Btn>
                </>
              ) : (
                /* Play: the ONE control of every inactive state, in the accent
                   Save wears while the loop runs -- it IS the primary action
                   here. Paused or stopped: save the form, resume or revive,
                   fire (`resumeNow`). No loop: create from the form, fire the
                   new id (`startNow`) -- where today's "Start loop" text button
                   stood and in its accent, so the first nudge goes out now
                   rather than after `idle_secs`. Nothing else renders on an
                   inactive loop: there is no separate Save because Play saves. */
                <Btn
                  type="button"
                  primary
                  className="!px-1.5"
                  aria-label={pausedManually ? i18nT('components.autoNudgePopover.resume_loop') : i18nT('components.autoNudgePopover.start_loop')}
                  title={pausedManually ? i18nT('components.autoNudgePopover.resume_loop') : i18nT('components.autoNudgePopover.start_loop')}
                  onClick={loop ? resumeNow : startNow}
                  disabled={saving || writeDisabled || !message.trim()}
                >
                  <Play size={14} aria-hidden />
                </Btn>
              )}
            </div>
          </div>
        )}
      </PopoverContent>}
    </Popover>
  )
}
