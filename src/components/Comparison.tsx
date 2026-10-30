import { useState } from 'react'
import type { Gpu } from '../data/gpus'
import type { Model } from '../data/models'
import {
  GiB, bytesPerToken, decodeTokensPerSec, memoryPlan, prefillTokensPerSec,
  type MemoryPlan, type SegmentKey, type Settings,
} from '../lib/calc'
import { fmtTokens } from './Controls'
import { GPU_COLORS } from './ContextCharts'

const SEG_COLOR: Record<SegmentKey, string> = {
  reserved: 'var(--seg-reserved)',
  runtime: 'var(--seg-runtime)',
  weights: 'var(--seg-weights)',
  kv: 'var(--seg-kv)',
  compute: 'var(--seg-compute)',
  kvPool: 'var(--seg-kvpool)',
  free: 'var(--free-fill)',
}

const SEG_HELP: Record<SegmentKey, string> = {
  reserved: 'VRAM already taken by your display or other programs.',
  runtime: 'CUDA context, cuBLAS workspace and driver allocations.',
  weights: 'The model parameters at the chosen quantization.',
  kv: 'Keys and values for every token in context, every layer. Grows linearly with context x parallel sequences.',
  compute: 'Scratch space for one forward pass: logits, activations, and attention scores when flash attention is off.',
  kvPool: 'vLLM pre-allocates the rest of its budget as KV blocks, so more concurrent requests fit.',
  free: 'Unallocated VRAM.',
}

const gib = (b: number) => (b / GiB).toFixed(b < 10 * GiB ? 2 : 1)

interface Tip { x: number; y: number; title: string; body: string; value: string }

export function Comparison({ gpus, slots, model, settings }: { gpus: Gpu[]; slots: number[]; model: Model; settings: Settings }) {
  const [tip, setTip] = useState<Tip | null>(null)
  const [showTable, setShowTable] = useState(false)

  if (gpus.length === 0) return <section className="card empty">Add a GPU above to start comparing.</section>

  const rows = gpus.map((gpu, i) => ({
    gpu,
    slot: slots[i],
    plan: memoryPlan(gpu, model, settings),
    tps: decodeTokensPerSec(gpu, model, settings, 0),
    tpsFull: decodeTokensPerSec(gpu, model, settings, settings.context),
    pp: prefillTokensPerSec(gpu, model),
  }))
  const scaleMax = niceMax(Math.max(...rows.map(r => Math.max(r.plan.capacity, r.plan.used))) / GiB)
  const ticks = axisTicks(scaleMax)
  const pct = (bytes: number) => `${(bytes / GiB / scaleMax) * 100}%`
  const isVllm = settings.engine === 'vllm'
  const keysShown = new Set(rows.flatMap(r => r.plan.segments.filter(s => s.bytes > 0).map(s => s.key)))

  return (
    <section className="card compare" aria-label="GPU comparison">
      <div className="compare-head">
        <h2>VRAM breakdown at {fmtTokens(settings.context)} context{settings.parallel > 1 ? ` x ${settings.parallel} sequences` : ''}</h2>
        <div className="seg-ctl" role="group" aria-label="View">
          <button aria-pressed={!showTable} onClick={() => setShowTable(false)}>Chart</button>
          <button aria-pressed={showTable} onClick={() => setShowTable(true)}>Table</button>
        </div>
      </div>

      <div className="legend">
        {rows[0].plan.segments.filter(s => keysShown.has(s.key)).map(s => (
          <span key={s.key}>
            <i className={`swatch${s.key === 'free' ? ' free' : ''}`} style={{ background: s.key === 'free' ? undefined : SEG_COLOR[s.key] }} />
            {s.label}
          </span>
        ))}
        {rows.some(r => !r.plan.fits) && <span><i className="swatch over" />Over capacity</span>}
      </div>

      {showTable ? <DataTable rows={rows} isVllm={isVllm} /> : (
        <>
          {rows.map(({ gpu, slot, plan, tps, tpsFull, pp }) => (
            <div className="gpu-row" key={gpu.id}>
              <div className="gpu-head">
                <div>
                  <div className="gpu-name"><i className="dot" style={{ background: GPU_COLORS[slot] }} />{gpu.name}</div>
                  <div className="gpu-spec">{gpu.vramGiB} GB {gpu.memoryType} · {fmtBw(gpu.bandwidthGBs)}</div>
                </div>
                <Verdict plan={plan} />
              </div>

              <div className="bar-wrap">
                <MemoryBar plan={plan} pct={pct} onTip={setTip} />
                <div className="bar-meta">
                  <span>{plan.fits ? `${gib(plan.used)} of ${gpu.vramGiB} GiB ${isVllm ? 'reserved by vLLM' : 'used'}` : `Needs ${gib(plan.used + (isVllm ? plan.overflow : 0))} GiB, has ${gpu.vramGiB}`}</span>
                  <span>
                    {plan.fits
                      ? isVllm ? '' : `${gib(plan.capacity - plan.used)} GiB free`
                      : `${gib(plan.overflow)} GiB short`}
                  </span>
                </div>
              </div>

              <div className="kpis">
                <Kpi k="Max context" v={plan.maxContext > 0 ? fmtTokens(plan.maxContext) : '–'}
                  s={plan.maxContext > 0 ? (settings.parallel > 1 ? `per sequence, x${settings.parallel}` : 'tokens that fit') : 'weights do not fit'} />
                <Kpi k="Generation" v={plan.fits || plan.maxContext > 0 ? Math.round(tps).toString() : '–'} unit="tok/s"
                  s={plan.fits ? `${Math.round(tpsFull)} tok/s at ${fmtTokens(settings.context)} filled` : 'needs offload'} />
                <Kpi k="Prompt" v={pp && (plan.fits || plan.maxContext > 0) ? fmtK(pp) : '–'} unit={pp ? 'tok/s' : ''}
                  s={pp ? 'llama.cpp measured, scaled' : 'no llama.cpp measurement'} />
                {isVllm
                  ? <Kpi k="Concurrency" v={plan.maxConcurrency ? `${plan.maxConcurrency}x` : '–'}
                      s={`${fmtTokens(plan.kvPoolTokens ?? 0)} tokens in KV pool`} />
                  : <Kpi k="Read per token" v={(bytesPerToken(model, settings, settings.context) / 1e9).toFixed(1)} unit="GB"
                      s={`at ${fmtTokens(settings.context)} filled; speed's main limit`} />}
              </div>
            </div>
          ))}
          <div className="axis" aria-hidden="true">
            {ticks.map(t => <span key={t} style={{ left: `${(t / scaleMax) * 100}%` }}>{t} GiB</span>)}
          </div>
        </>
      )}

      {tip && (
        <div className="tooltip" style={{ left: Math.min(tip.x + 14, window.innerWidth - 270), top: tip.y + 14 }}>
          <b>{tip.title} · {tip.value}</b>
          <span className="muted">{tip.body}</span>
        </div>
      )}
    </section>
  )
}

function MemoryBar({ plan, pct, onTip }: { plan: MemoryPlan; pct: (b: number) => string; onTip: (t: Tip | null) => void }) {
  const visible = plan.segments.filter(s => s.bytes > 0)
  let offset = 0
  return (
    <div className="bar" onMouseLeave={() => onTip(null)}>
      {visible.map((s, i) => {
        const left = offset
        offset += s.bytes
        const wide = s.bytes / plan.capacity > 0.14
        return (
          <div key={s.key}
            className={`seg ${s.key === 'free' ? 'free' : ''} ${i === 0 ? 'first' : ''} ${i === visible.length - 1 && plan.fits ? 'last' : ''}`}
            style={{ left: pct(left), width: pct(s.bytes), background: s.key === 'free' ? undefined : SEG_COLOR[s.key] }}
            onMouseMove={e => onTip({ x: e.clientX, y: e.clientY, title: s.label, value: `${gib(s.bytes)} GiB`, body: SEG_HELP[s.key] })}
          >
            {wide && s.key !== 'free' && <span className="in" style={s.key === 'kvPool' ? { color: 'var(--text-primary)' } : undefined}>{s.label.replace(/ \(.*\)/, '')} {gib(s.bytes)}</span>}
          </div>
        )
      })}
      {!plan.fits && (
        <div className="seg over last" style={{ left: pct(plan.capacity), width: pct(plan.overflow) }}
          onMouseMove={e => onTip({ x: e.clientX, y: e.clientY, title: 'Over capacity', value: `${gib(plan.overflow)} GiB`, body: 'This much more VRAM is needed. Reduce context, quantize harder, or offload layers to CPU.' })} />
      )}
      <div className="cap" style={{ left: `calc(${pct(plan.capacity)} - 1px)` }} title="VRAM capacity" />
    </div>
  )
}

function Verdict({ plan }: { plan: MemoryPlan }) {
  return plan.fits
    ? <div className="verdict ok"><Icon ok />Fits</div>
    : <div className="verdict no"><Icon ok={false} />Does not fit</div>
}

function Icon({ ok }: { ok: boolean }) {
  return (
    <svg width="14" height="14" viewBox="0 0 16 16" aria-hidden="true">
      <circle cx="8" cy="8" r="7" fill="currentColor" />
      {ok
        ? <path d="M4.5 8.2l2.2 2.2 4.6-4.8" stroke="var(--surface-1)" strokeWidth="1.8" fill="none" strokeLinecap="round" strokeLinejoin="round" />
        : <path d="M5.5 5.5l5 5m0-5l-5 5" stroke="var(--surface-1)" strokeWidth="1.8" strokeLinecap="round" />}
    </svg>
  )
}

function Kpi({ k, v, unit, s }: { k: string; v: string; unit?: string; s?: string }) {
  return (
    <div className="kpi">
      <div className="k">{k}</div>
      <div className="v">{v}{unit && v !== '–' && <small>{unit}</small>}</div>
      {s && <div className="s">{s}</div>}
    </div>
  )
}

type Row = { gpu: Gpu; slot: number; plan: MemoryPlan; tps: number; tpsFull: number; pp: number | null }

function DataTable({ rows, isVllm }: { rows: Row[]; isVllm: boolean }) {
  const keys = rows[0].plan.segments.map(s => s.key)
  return (
    <div className="table-scroll">
      <table className="data">
        <thead>
          <tr>
            <th>GPU (GiB)</th>
            {rows[0].plan.segments.map(s => <th key={s.key}>{s.label}</th>)}
            <th>Fits</th><th>Max ctx</th><th>Gen tok/s</th><th>Prompt tok/s</th>
            {isVllm && <th>Concurrency</th>}
          </tr>
        </thead>
        <tbody>
          {rows.map(({ gpu, plan, tps, pp }) => (
            <tr key={gpu.id}>
              <td>{gpu.name}</td>
              {keys.map(k => <td key={k}>{gib(plan.segments.find(s => s.key === k)!.bytes)}</td>)}
              <td>{plan.fits ? 'Yes' : `No (+${gib(plan.overflow)})`}</td>
              <td>{fmtTokens(plan.maxContext)}</td>
              <td>{Math.round(tps)}</td>
              <td>{pp ? Math.round(pp) : '–'}</td>
              {isVllm && <td>{plan.maxConcurrency}</td>}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

const fmtBw = (gbs: number) => (gbs >= 1000 ? `${(gbs / 1000).toFixed(2)} TB/s` : `${gbs} GB/s`)
const fmtK = (n: number) => (n >= 1000 ? `${(n / 1000).toFixed(1)}K` : Math.round(n).toString())

function niceMax(v: number) {
  const steps = [8, 12, 16, 24, 32, 40, 48, 64, 80, 96, 120, 128, 160, 192, 256]
  return steps.find(s => s >= v) ?? Math.ceil(v / 64) * 64
}

function axisTicks(max: number) {
  const step = max <= 16 ? 4 : max <= 48 ? 8 : max <= 96 ? 16 : 32
  const out: number[] = []
  for (let t = 0; t <= max; t += step) out.push(t)
  return out
}
