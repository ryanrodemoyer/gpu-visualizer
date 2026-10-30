import { GPUS, gpuById, type Segment } from '../data/gpus'
import { GPU_COLORS } from './ContextCharts'

const SEGMENTS: [Segment, string][] = [['consumer', 'Consumer'], ['workstation', 'Workstation'], ['datacenter', 'Datacenter']]

export function GpuPicker({ slots, onChange }: { slots: (string | null)[]; onChange: (slots: (string | null)[]) => void }) {
  const available = GPUS.filter(g => !slots.includes(g.id))
  const freeSlot = slots.indexOf(null)
  const set = (i: number, id: string | null) => onChange(slots.map((s, j) => (j === i ? id : s)))
  return (
    <div className="picker" aria-label="GPUs to compare">
      {slots.map((id, i) => id && (
        <span className="chip" key={id}>
          <i className="dot" style={{ background: GPU_COLORS[i] }} />
          {gpuById(id)?.name}
          <button aria-label={`Remove ${gpuById(id)?.name}`} onClick={() => set(i, null)}>×</button>
        </span>
      ))}
      {freeSlot >= 0 && (
        <select aria-label="Add a GPU" value="" onChange={e => e.target.value && set(freeSlot, e.target.value)}>
          <option value="">+ Add GPU</option>
          {SEGMENTS.map(([seg, label]) => (
            <optgroup key={seg} label={label}>
              {available.filter(g => g.segment === seg).map(g => (
                <option key={g.id} value={g.id}>{g.name} · {g.vramGiB} GB</option>
              ))}
            </optgroup>
          ))}
        </select>
      )}
    </div>
  )
}
