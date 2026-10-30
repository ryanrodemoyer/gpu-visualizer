import type { AppState } from '../App'
import { defaultQuantFor, quantsFor } from '../App'
import { MODELS, modelById, type Model } from '../data/models'
import { KV_TYPES, quantById } from '../data/quants'

const CONTEXT_STOPS = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]

export const fmtTokens = (n: number) =>
  n >= 1024 ? `${Number.isInteger(n / 1024) ? n / 1024 : (n / 1024).toFixed(1)}K` : String(n)

interface Props {
  state: AppState
  model: Model
  update: (patch: Partial<AppState>) => void
}

export function Controls({ state, model, update }: Props) {
  const stops = CONTEXT_STOPS.filter(c => c <= model.maxContext)
  if (stops[stops.length - 1] !== model.maxContext) stops.push(model.maxContext)
  const ctxIdx = Math.max(0, stops.findIndex(c => c >= state.context))
  const quant = quantById(state.quantId)!
  const isLlama = state.engine === 'llama.cpp'

  return (
    <section className="card controls" aria-label="Settings">
      <div className="field">
        <label htmlFor="model">Model</label>
        <select id="model" value={state.modelId} onChange={e => {
          const m = modelById(e.target.value)!
          update({ modelId: m.id, quantId: defaultQuantFor(state.engine, m) })
        }}>
          {MODELS.map(m => <option key={m.id} value={m.id}>{m.name}</option>)}
        </select>
        <span className="hint">
          {model.layers} layers, {model.kvHeads} KV heads x {model.headDim} dim
          {model.slidingWindow ? `, sliding window ${model.slidingWindow.window}` : ''}
        </span>
      </div>

      <div className="field">
        <label htmlFor="quant">Weights {isLlama ? '(GGUF)' : ''}</label>
        <select id="quant" value={state.quantId} onChange={e => update({ quantId: e.target.value })}>
          {quantsFor(state.engine, model).map(q => <option key={q.id} value={q.id}>{q.label}</option>)}
        </select>
        <span className="hint">{quant.note}</span>
      </div>

      <div className="field">
        <span className="lbl">Context per sequence <output>{fmtTokens(state.context)}</output></span>
        <input type="range" min={0} max={stops.length - 1} step={1} value={ctxIdx}
          aria-label="Context length"
          onChange={e => update({ context: stops[Number(e.target.value)] })} />
        <span className="hint">{isLlama ? '-c' : '--max-model-len'} · model max {fmtTokens(model.maxContext)}</span>
      </div>

      <div className="field">
        <span className="lbl">Parallel sequences <output>{state.parallel}</output></span>
        <input type="range" min={1} max={16} step={1} value={state.parallel}
          aria-label="Parallel sequences"
          onChange={e => update({ parallel: Number(e.target.value) })} />
        <span className="hint">{isLlama ? '-np (total -c = context x parallel)' : 'concurrent requests at full context'}</span>
      </div>

      <div className="field">
        <label htmlFor="kv">KV cache type</label>
        <select id="kv" value={state.kvTypeId} onChange={e => update({ kvTypeId: e.target.value })}>
          {KV_TYPES.filter(k => k.engine === state.engine).map(k => <option key={k.id} value={k.id}>{k.label}</option>)}
        </select>
        <span className="hint">{isLlama ? '-ctk / -ctv (quantized V needs flash attention)' : '--kv-cache-dtype'}</span>
      </div>

      {isLlama ? (
        <div className="field">
          <span className="lbl">Flash attention</span>
          <label className="check">
            <input type="checkbox" checked={state.flashAttn} onChange={e => update({ flashAttn: e.target.checked })} />
            -fa on
          </label>
          <span className="hint">Off materializes the full attention score matrix in the compute buffer.</span>
        </div>
      ) : (
        <div className="field">
          <span className="lbl">GPU memory utilization <output>{state.gpuMemUtil.toFixed(2)}</output></span>
          <input type="range" min={0.5} max={0.98} step={0.01} value={state.gpuMemUtil}
            aria-label="GPU memory utilization"
            onChange={e => update({ gpuMemUtil: Number(e.target.value) })} />
          <span className="hint">--gpu-memory-utilization. vLLM fills this share; the rest becomes KV pool.</span>
        </div>
      )}

      <div className="field">
        <span className="lbl">Display / other apps <output>{state.reservedGiB.toFixed(1)} GiB</output></span>
        <input type="range" min={0} max={4} step={0.1} value={state.reservedGiB}
          aria-label="VRAM used by display and other apps"
          onChange={e => update({ reservedGiB: Number(e.target.value) })} />
        <span className="hint">0 for a headless card. A desktop session often uses 0.5 to 1.5 GiB.</span>
      </div>
    </section>
  )
}
