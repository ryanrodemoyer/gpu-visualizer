import { GPUS, gpuById, type Segment } from '../data/gpus'

const MAX_GPUS = 6
const SEGMENTS: [Segment, string][] = [['consumer', 'Consumer'], ['workstation', 'Workstation'], ['datacenter', 'Datacenter']]

export function GpuPicker({ gpuIds, onChange }: { gpuIds: string[]; onChange: (ids: string[]) => void }) {
  const available = GPUS.filter(g => !gpuIds.includes(g.id))
  return (
    <div className="picker" aria-label="GPUs to compare">
      {gpuIds.map(id => (
        <span className="chip" key={id}>
          {gpuById(id)?.name}
          <button aria-label={`Remove ${gpuById(id)?.name}`} onClick={() => onChange(gpuIds.filter(x => x !== id))}>×</button>
        </span>
      ))}
      {gpuIds.length < MAX_GPUS && (
        <select aria-label="Add a GPU" value="" onChange={e => e.target.value && onChange([...gpuIds, e.target.value])}>
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
