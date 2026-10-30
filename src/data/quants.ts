// Weight formats. GGUF mixes (Q4_K_M etc.) store different tensors at different types, so
// their bits-per-weight are effective averages measured from real files:
// total file bytes * 8 / params, averaged over bartowski's Llama 3.1 8B and Llama 3.3 70B GGUFs.
//   https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF
//   https://huggingface.co/bartowski/Llama-3.3-70B-Instruct-GGUF
// Block formats use their exact block layout (e.g. MXFP4 = 32 x 4-bit + one 8-bit scale = 4.25).

export type Engine = 'llama.cpp' | 'vllm'

export interface Quant {
  id: string
  label: string
  engine: Engine
  /** Bits per weight for transformer-block weights (attention + FFN) */
  bpw: number
  /** Bits per weight for MoE expert FFN weights, if different from bpw */
  expertBpw?: number
  /** Bits per weight for embedding / output head, if different from bpw */
  embedBpw?: number
  note: string
}

export const QUANTS: Quant[] = [
  { id: 'f16', label: 'F16 / BF16', engine: 'llama.cpp', bpw: 16, note: 'Unquantized half precision.' },
  { id: 'q8_0', label: 'Q8_0', engine: 'llama.cpp', bpw: 8.5, note: 'Near-lossless. 32 x int8 + fp16 scale per block.' },
  { id: 'q6_k', label: 'Q6_K', engine: 'llama.cpp', bpw: 6.57, note: 'Very close to Q8_0 quality.' },
  { id: 'q5_k_m', label: 'Q5_K_M', engine: 'llama.cpp', bpw: 5.69, note: 'High quality.' },
  { id: 'q4_k_m', label: 'Q4_K_M', engine: 'llama.cpp', bpw: 4.86, note: 'The common default: good quality per GB.' },
  { id: 'q4_0', label: 'Q4_0', engine: 'llama.cpp', bpw: 4.55, note: 'Legacy 4-bit, used by the llama.cpp benchmark scoreboard.' },
  { id: 'iq4_xs', label: 'IQ4_XS', engine: 'llama.cpp', bpw: 4.37, note: 'Smaller than Q4_K_M with similar quality.' },
  { id: 'q3_k_m', label: 'Q3_K_M', engine: 'llama.cpp', bpw: 3.95, note: 'Noticeable quality loss.' },
  { id: 'q2_k', label: 'Q2_K', engine: 'llama.cpp', bpw: 3.08, note: 'Large quality loss; last resort.' },
  // ggml-org gpt-oss GGUFs keep experts in MXFP4 and everything else in Q8_0:
  // gpt-oss-20b-MXFP4.gguf = 12.1 GB, gpt-oss-120b-MXFP4.gguf = 63.4 GB
  { id: 'mxfp4', label: 'MXFP4 (native)', engine: 'llama.cpp', bpw: 8.5, expertBpw: 4.25, note: 'gpt-oss native: MoE experts in MXFP4, rest in Q8_0.' },

  { id: 'vllm-bf16', label: 'BF16', engine: 'vllm', bpw: 16, note: 'Unquantized checkpoint.' },
  { id: 'vllm-fp8', label: 'FP8', engine: 'vllm', bpw: 8, embedBpw: 16, note: 'FP8 weights (Ada / Hopper / Blackwell). Embeddings stay BF16.' },
  { id: 'vllm-awq', label: 'AWQ / GPTQ 4-bit', engine: 'vllm', bpw: 4.25, embedBpw: 16, note: 'INT4, group size 128. Embeddings stay BF16.' },
  { id: 'vllm-mxfp4', label: 'MXFP4 (native)', engine: 'vllm', bpw: 16, expertBpw: 4.25, note: 'gpt-oss native: experts MXFP4, attention BF16.' },
]

/** KV cache element types and their bytes per element */
export interface KvType {
  id: string
  label: string
  engine: Engine
  bytes: number
}

export const KV_TYPES: KvType[] = [
  { id: 'f16', label: 'F16', engine: 'llama.cpp', bytes: 2 },
  { id: 'q8_0', label: 'Q8_0', engine: 'llama.cpp', bytes: 34 / 32 },
  { id: 'q4_0', label: 'Q4_0', engine: 'llama.cpp', bytes: 18 / 32 },
  { id: 'auto', label: 'BF16 (auto)', engine: 'vllm', bytes: 2 },
  { id: 'fp8', label: 'FP8', engine: 'vllm', bytes: 1 },
]

export const quantById = (id: string) => QUANTS.find(q => q.id === id)
export const kvTypeById = (id: string) => KV_TYPES.find(k => k.id === id)
