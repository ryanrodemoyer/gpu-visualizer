import { useEffect, useRef, useState, type ReactNode } from 'react'
import type { Gpu } from '../data/gpus'
import type { Model } from '../data/models'
import { GiB, decodeTokensPerSec, memoryNeeded, memoryPlan, usableCapacity, type Settings } from '../lib/calc'
import { fmtTokens } from './Controls'

/** GPU series colors (validated adjacent-pair CVD order); chosen to avoid the VRAM segment hues. */
export const GPU_COLORS = ['var(--gpu-1)', 'var(--gpu-2)', 'var(--gpu-3)', 'var(--gpu-4)', 'var(--gpu-5)']

const H = 240
const M = { top: 12, right: 92, bottom: 30, left: 44 }

function useWidth<T extends HTMLElement>() {
  const ref = useRef<T>(null)
  const [w, setW] = useState(480)
  useEffect(() => {
    if (!ref.current) return
    const ro = new ResizeObserver(([e]) => setW(e.contentRect.width))
    ro.observe(ref.current)
    return () => ro.disconnect()
  }, [])
  return [ref, w] as const
}

function contextSamples(max: number) {
  const out: number[] = []
  for (let c = 512; c < max; c *= Math.SQRT2) out.push(Math.round(c))
  out.push(max)
  return out
}

interface Props { gpus: Gpu[]; slots: number[]; model: Model; settings: Settings }

export function ContextCharts({ gpus, slots, model, settings }: Props) {
  if (gpus.length === 0) return null
  return (
    <section className="charts">
      <MemoryChart gpus={gpus} slots={slots} model={model} settings={settings} />
      <SpeedChart gpus={gpus} slots={slots} model={model} settings={settings} />
    </section>
  )
}

function Frame({ title, sub, children }: { title: string; sub: string; children: (w: number) => ReactNode }) {
  const [ref, w] = useWidth<HTMLDivElement>()
  return (
    <div className="card chart" ref={ref}>
      <h3>{title}</h3>
      <p className="sub">{sub}</p>
      {children(Math.max(260, w - 32))}
    </div>
  )
}

function xScale(w: number, max: number) {
  const lo = Math.log2(512)
  const hi = Math.log2(max)
  return (c: number) => M.left + ((Math.log2(c) - lo) / (hi - lo)) * (w - M.left - M.right)
}

function xTicks(max: number) {
  return [1024, 4096, 16384, 65536].filter(t => t < max * 0.8).concat(max)
}

function niceStep(max: number, n = 4) {
  const raw = max / n
  const pow = 10 ** Math.floor(Math.log10(raw))
  return [1, 2, 2.5, 5, 10].map(m => m * pow).find(s => s >= raw)!
}

function MemoryChart({ gpus, slots, model, settings }: Props) {
  const [hover, setHover] = useState<number | null>(null)
  const ctxs = contextSamples(model.maxContext)
  const pts = ctxs.map(c => ({ c, ...memoryNeeded(model, settings, c) }))
  const caps = gpus.map(g => ({ g, cap: usableCapacity(g, settings) }))
  const maxCap = Math.max(...caps.map(c => c.cap))
  const last = pts[pts.length - 1]
  const yMaxRaw = Math.max(maxCap * 1.15, Math.min(last.base + last.weights + last.kv + last.compute, maxCap * 1.6)) / GiB
  const step = niceStep(yMaxRaw)
  const yMax = Math.ceil(yMaxRaw / step) * step

  return (
    <Frame title="Memory needed vs context" sub={`Where the stack crosses a GPU's line is its max context${settings.engine === 'vllm' ? ' (line = utilization budget)' : ''}.`}>
      {w => {
        const x = xScale(w, model.maxContext)
        const y = (b: number) => M.top + (1 - Math.min(b / GiB, yMax) / yMax) * (H - M.top - M.bottom)
        const band = (lo: (p: typeof pts[0]) => number, hi: (p: typeof pts[0]) => number) =>
          pts.map((p, i) => `${i ? 'L' : 'M'}${x(p.c)},${y(hi(p))}`).join('') +
          pts.slice().reverse().map(p => `L${x(p.c)},${y(lo(p))}`).join('') + 'Z'
        const fixed = (p: typeof pts[0]) => p.base + p.weights
        const hp = hover !== null ? pts[hover] : null
        return (
          <svg width={w} height={H} role="img" aria-label="Stacked memory needed by context length, with each GPU's capacity"
            onMouseMove={e => setHover(nearest(e, ctxs, x))} onMouseLeave={() => setHover(null)}>
            <YGrid yMax={yMax} step={step} y={b => y(b * GiB)} w={w} unit=" GiB" />
            <path d={band(() => 0, fixed)} fill="var(--seg-weights)" />
            <path d={band(fixed, p => fixed(p) + p.compute)} fill="var(--seg-compute)" stroke="var(--surface-1)" strokeWidth={1} />
            <path d={band(p => fixed(p) + p.compute, p => fixed(p) + p.compute + p.kv)} fill="var(--seg-kv)" stroke="var(--surface-1)" strokeWidth={1} />
            <XAxis max={model.maxContext} x={x} />
            <line x1={x(settings.context)} x2={x(settings.context)} y1={M.top} y2={H - M.bottom} className="now" />
            {caps.map(({ g, cap }, i) => (
              <g key={g.id}>
                <line x1={M.left} x2={w - M.right} y1={y(cap)} y2={y(cap)} stroke={GPU_COLORS[slots[i]]} strokeWidth={2} strokeDasharray="6 4" />
                <text x={w - M.right + 6} y={labelY(caps.map(c => y(c.cap)), i)} className="dl" dominantBaseline="middle">
                  <tspan fill={GPU_COLORS[slots[i]]}>■</tspan> {short(g.name)}
                </text>
              </g>
            ))}
            {hp && <Crosshair x={x(hp.c)} />}
            {hp && (
              <foreignObject x={tipX(x(hp.c), 200, w)} y={M.top} width={200} height={110}>
                <div className="tip-inline">
                  <b>{fmtTokens(hp.c)} context{settings.parallel > 1 ? ` x${settings.parallel}` : ''}</b>
                  <div>Total {gibs(hp.base + hp.weights + hp.compute + hp.kv)}</div>
                  <div><i style={{ background: 'var(--seg-kv)' }} />KV {gibs(hp.kv)}</div>
                  <div><i style={{ background: 'var(--seg-compute)' }} />Compute {gibs(hp.compute)}</div>
                  <div><i style={{ background: 'var(--seg-weights)' }} />Weights + runtime {gibs(hp.base + hp.weights)}</div>
                </div>
              </foreignObject>
            )}
          </svg>
        )
      }}
    </Frame>
  )
}

function SpeedChart({ gpus, slots, model, settings }: Props) {
  const [hover, setHover] = useState<number | null>(null)
  const ctxs = contextSamples(model.maxContext)
  const series = gpus.map(g => {
    const max = memoryPlan(g, model, { ...settings }).maxContext
    return { g, max, pts: ctxs.filter(c => c <= max).map(c => ({ c, t: decodeTokensPerSec(g, model, settings, c) })) }
  })
  const yMaxRaw = Math.max(10, ...series.flatMap(s => s.pts.map(p => p.t))) * 1.05
  const step = niceStep(yMaxRaw)
  const yMax = Math.ceil(yMaxRaw / step) * step
  const none = series.every(s => s.pts.length === 0)
  const missing = series.filter(s => s.pts.length === 0).map(s => short(s.g.name))

  return (
    <Frame title="Generation speed as context fills" sub={`Each token re-reads the KV cache, so speed drops as the conversation grows. Lines end at max context.${missing.length && !none ? ` Not shown (doesn't fit): ${missing.join(', ')}.` : ''}`}>
      {w => {
        const x = xScale(w, model.maxContext)
        const y = (t: number) => M.top + (1 - t / yMax) * (H - M.top - M.bottom)
        const ends = series.map(s => (s.pts.length ? y(s.pts[s.pts.length - 1].t) : -999))
        const hc = hover !== null ? ctxs[hover] : null
        return (
          <svg width={w} height={H} role="img" aria-label="Tokens per second by context depth for each GPU"
            onMouseMove={e => setHover(nearest(e, ctxs, x))} onMouseLeave={() => setHover(null)}>
            <YGrid yMax={yMax} step={step} y={y} w={w} unit="" />
            <XAxis max={model.maxContext} x={x} />
            {none && <text x={w / 2} y={H / 2} textAnchor="middle" className="dl">Model does not fit on any selected GPU</text>}
            {series.map((s, i) => s.pts.length > 0 && (
              <g key={s.g.id}>
                <path d={s.pts.map((p, j) => `${j ? 'L' : 'M'}${x(p.c)},${y(p.t)}`).join('')} fill="none" stroke={GPU_COLORS[slots[i]]} strokeWidth={2} strokeLinejoin="round" />
                <circle cx={x(s.pts[s.pts.length - 1].c)} cy={y(s.pts[s.pts.length - 1].t)} r={4} fill={GPU_COLORS[slots[i]]} stroke="var(--surface-1)" strokeWidth={2} />
                <text x={w - M.right + 6} y={labelY(ends, i)} className="dl" dominantBaseline="middle">
                  <tspan fill={GPU_COLORS[slots[i]]}>■</tspan> {short(s.g.name)}
                </text>
              </g>
            ))}
            {hc !== null && <Crosshair x={x(hc)} />}
            {hc !== null && (
              <foreignObject x={tipX(x(hc), 160, w)} y={M.top} width={160} height={24 + 18 * series.length}>
                <div className="tip-inline">
                  <b>At {fmtTokens(hc)} filled</b>
                  {series.map((s, i) => (
                    <div key={s.g.id}><i style={{ background: GPU_COLORS[slots[i]] }} />{short(s.g.name)}{' '}
                      {hc <= s.max ? `${Math.round(decodeTokensPerSec(s.g, model, settings, hc))} tok/s` : 'won\'t fit'}</div>
                  ))}
                </div>
              </foreignObject>
            )}
          </svg>
        )
      }}
    </Frame>
  )
}

function YGrid({ yMax, step, y, w, unit }: { yMax: number; step: number; y: (v: number) => number; w: number; unit: string }) {
  const ticks: number[] = []
  for (let v = 0; v <= yMax + 1e-9; v += step) ticks.push(v)
  return (
    <g className="grid">
      {ticks.map(v => (
        <g key={v}>
          <line x1={M.left} x2={w - M.right} y1={y(v)} y2={y(v)} />
          <text x={M.left - 6} y={y(v)} textAnchor="end" dominantBaseline="middle">{+v.toFixed(1)}{unit && v === ticks[ticks.length - 1] ? unit : ''}</text>
        </g>
      ))}
    </g>
  )
}

function XAxis({ max, x }: { max: number; x: (c: number) => number }) {
  return (
    <g className="grid">
      {xTicks(max).map(t => (
        <text key={t} x={x(t)} y={H - M.bottom + 16} textAnchor="middle">{fmtTokens(t)}</text>
      ))}
    </g>
  )
}

const Crosshair = ({ x }: { x: number }) => <line x1={x} x2={x} y1={M.top} y2={H - M.bottom} className="cross" />

function nearest(e: React.MouseEvent<SVGSVGElement>, ctxs: number[], x: (c: number) => number) {
  const px = e.clientX - e.currentTarget.getBoundingClientRect().left
  let best = 0
  ctxs.forEach((c, i) => { if (Math.abs(x(c) - px) < Math.abs(x(ctxs[best]) - px)) best = i })
  return best
}

/** Spread direct labels so they never overlap (min 14px apart), keeping their order. */
function labelY(ys: number[], i: number) {
  const order = ys.map((y, j) => ({ y, j })).sort((a, b) => a.y - b.y)
  const placed: number[] = []
  for (const o of order) placed[o.j] = Math.max(o.y, ...placed.filter(v => v !== undefined).map(v => v + 14))
  return placed[i]
}

/** Keep the hover card beside the crosshair, flipping left in the right half. */
const tipX = (cx: number, width: number, w: number) => (cx > w / 2 ? Math.max(0, cx - width - 8) : cx + 8)

const short = (name: string) => name.replace('Radeon ', '').replace(' Blackwell', '')
const gibs = (b: number) => `${(b / GiB).toFixed(1)} GiB`
