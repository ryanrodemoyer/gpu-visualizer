import { useEffect, useMemo, useState } from 'react'
import { GPUS, gpuById } from './data/gpus'
import { MODELS, modelById, type Model } from './data/models'
import { KV_TYPES, QUANTS, kvTypeById, quantById, type Engine } from './data/quants'
import { DEFAULT_SETTINGS, GiB, type Settings } from './lib/calc'
import { paramBreakdown } from './lib/params'
import { Controls } from './components/Controls'
import { GpuPicker } from './components/GpuPicker'
import { Comparison } from './components/Comparison'
import { Methodology } from './components/Methodology'
import { ContextCharts } from './components/ContextCharts'
import { readUrlState, writeUrlState } from './lib/urlState'

export interface AppState {
  engine: Engine
  modelId: string
  quantId: string
  kvTypeId: string
  context: number
  parallel: number
  ubatch: number
  flashAttn: boolean
  gpuMemUtil: number
  batchedTokens: number
  reservedGiB: number
  /** Fixed color slots (null = empty) so a GPU keeps its color when others are removed */
  gpuSlots: (string | null)[]
}

export const MAX_GPUS = 5

export function defaultQuantFor(engine: Engine, m: Model): string {
  if (m.defaultQuant === 'mxfp4') return engine === 'vllm' ? 'vllm-mxfp4' : 'mxfp4'
  return engine === 'vllm' ? 'vllm-bf16' : 'q4_k_m'
}

export const quantsFor = (engine: Engine, m: Model) =>
  QUANTS.filter(q => q.engine === engine && (!q.id.includes('mxfp4') || m.moe))

const INITIAL: AppState = {
  engine: 'llama.cpp',
  modelId: 'qwen3-32b',
  quantId: 'q4_k_m',
  kvTypeId: 'f16',
  context: DEFAULT_SETTINGS.context,
  parallel: DEFAULT_SETTINGS.parallel,
  ubatch: DEFAULT_SETTINGS.ubatch,
  flashAttn: DEFAULT_SETTINGS.flashAttn,
  gpuMemUtil: DEFAULT_SETTINGS.gpuMemUtil,
  batchedTokens: DEFAULT_SETTINGS.batchedTokens,
  reservedGiB: 0,
  gpuSlots: ['rtx-3090', 'rtx-4090', 'rtx-5090', null, null],
}

export default function App() {
  const [state, setState] = useState<AppState>(() => normalize({ ...INITIAL, ...readUrlState() }))
  const update = (patch: Partial<AppState>) => setState(s => normalize({ ...s, ...patch }))
  useEffect(() => writeUrlState(state), [state])

  const model = modelById(state.modelId)!
  const settings: Settings = useMemo(() => ({
    engine: state.engine,
    quant: quantById(state.quantId)!,
    kvType: kvTypeById(state.kvTypeId)!,
    context: state.context,
    parallel: state.parallel,
    ubatch: state.ubatch,
    flashAttn: state.flashAttn,
    gpuMemUtil: state.gpuMemUtil,
    batchedTokens: state.batchedTokens,
    reservedBytes: state.reservedGiB * GiB,
  }), [state])
  const slotted = state.gpuSlots
    .map((id, slot) => ({ gpu: id ? gpuById(id) : undefined, slot }))
    .filter((x): x is { gpu: NonNullable<typeof x.gpu>; slot: number } => !!x.gpu)
  const gpus = slotted.map(x => x.gpu)
  const slots = slotted.map(x => x.slot)
  const p = paramBreakdown(model)

  return (
    <div className="app">
      <header className="top">
        <div>
          <h1>GPU VRAM Visualizer</h1>
          <p>
            {model.name}: {(p.total / 1e9).toFixed(1)}B params
            {model.moe ? `, ${(p.active / 1e9).toFixed(1)}B active per token` : ''}. See what fits, and how fast.
          </p>
        </div>
        <div className="seg-ctl" role="group" aria-label="Inference engine">
          {(['llama.cpp', 'vllm'] as Engine[]).map(e => (
            <button key={e} aria-pressed={state.engine === e} onClick={() => update({ engine: e })}>
              {e === 'vllm' ? 'vLLM' : e}
            </button>
          ))}
        </div>
      </header>

      <Controls state={state} model={model} update={update} />
      <GpuPicker slots={state.gpuSlots} onChange={gpuSlots => update({ gpuSlots })} />
      <Comparison gpus={gpus} slots={slots} model={model} settings={settings} />
      <ContextCharts gpus={gpus} slots={slots} model={model} settings={settings} />
      <Methodology />

      <footer>
        Estimates, not guarantees. Real usage varies by driver, build and flags.{' '}
        <a href="https://github.com/ryanrodemoyer/gpu-visualizer">Source and data on GitHub</a>
      </footer>
    </div>
  )
}

/** Keep dependent fields valid when the engine or model changes. */
function normalize(s: AppState): AppState {
  const m = modelById(s.modelId) ?? MODELS[0]
  const quants = quantsFor(s.engine, m)
  const quantId = quants.some(q => q.id === s.quantId) ? s.quantId : defaultQuantFor(s.engine, m)
  const kvs = KV_TYPES.filter(k => k.engine === s.engine)
  const kvTypeId = kvs.some(k => k.id === s.kvTypeId) ? s.kvTypeId : kvs[0].id
  const context = Math.min(s.context, m.maxContext)
  const gpuSlots = Array.from({ length: MAX_GPUS }, (_, i) => {
    const id = s.gpuSlots[i]
    return id && GPUS.some(g => g.id === id) ? id : null
  })
  return { ...s, modelId: m.id, quantId, kvTypeId, context, gpuSlots }
}
