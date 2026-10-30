import type { AppState } from '../App'

// Settings live in the URL hash so a comparison can be shared as a link.
// Short keys keep links readable: #e=llama.cpp&m=qwen3-32b&q=q4_k_m&g=rtx-3090,rtx-4090
const KEYS: [keyof AppState, string][] = [
  ['engine', 'e'], ['modelId', 'm'], ['quantId', 'q'], ['kvTypeId', 'kv'], ['context', 'c'],
  ['parallel', 'np'], ['ubatch', 'ub'], ['flashAttn', 'fa'], ['gpuMemUtil', 'util'], ['reservedGiB', 'res'],
]

export function readUrlState(): Partial<AppState> {
  const p = new URLSearchParams(window.location.hash.slice(1))
  const out: Record<string, unknown> = {}
  for (const [key, short] of KEYS) {
    const v = p.get(short)
    if (v === null) continue
    if (key === 'flashAttn') out[key] = v === '1'
    else if (key === 'engine' || key === 'modelId' || key === 'quantId' || key === 'kvTypeId') out[key] = v
    else if (Number.isFinite(Number(v))) out[key] = Number(v)
  }
  const g = p.get('g')
  if (g !== null) out.gpuSlots = g.split(',').map(id => id || null)
  return out as Partial<AppState>
}

export function writeUrlState(s: AppState) {
  const p = new URLSearchParams()
  for (const [key, short] of KEYS) {
    const v = s[key]
    p.set(short, typeof v === 'boolean' ? (v ? '1' : '0') : String(v))
  }
  p.set('g', s.gpuSlots.map(id => id ?? '').join(',').replace(/,+$/, ''))
  history.replaceState(null, '', `#${p.toString().replace(/%2C/g, ',')}`)
}
