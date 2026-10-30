// Model architectures, copied from each model's Hugging Face config.json (text decoder only).
// Parameter counts are derived from these numbers in lib/params.ts and checked against
// the published totals in tests.

export interface MoeConfig {
  experts: number
  activeExperts: number
  expertIntermediate: number
}

export interface SlidingWindowConfig {
  /** Tokens each sliding layer attends to */
  window: number
  /** Fraction of layers that use the sliding window (rest are full attention) */
  fraction: number
}

export interface Model {
  id: string
  name: string
  family: string
  layers: number
  hidden: number
  heads: number
  kvHeads: number
  headDim: number
  /** Dense FFN width (unused for MoE models whose every layer is MoE) */
  intermediate: number
  vocab: number
  tiedEmbeddings: boolean
  maxContext: number
  moe?: MoeConfig
  slidingWindow?: SlidingWindowConfig
  /** Published total parameter count, billions (for validation) */
  publishedParamsB: number
  publishedActiveB?: number
  /** Native-format quant to default to, e.g. MXFP4 for gpt-oss */
  defaultQuant?: string
  source: string
}

const hf = (repo: string) => `https://huggingface.co/${repo}/blob/main/config.json`

export const MODELS: Model[] = [
  {
    id: 'llama-2-7b', name: 'Llama 2 7B', family: 'Llama',
    layers: 32, hidden: 4096, heads: 32, kvHeads: 32, headDim: 128, intermediate: 11008,
    vocab: 32000, tiedEmbeddings: false, maxContext: 4096,
    publishedParamsB: 6.74, source: hf('meta-llama/Llama-2-7b-hf'),
  },
  {
    id: 'llama-3.1-8b', name: 'Llama 3.1 8B', family: 'Llama',
    layers: 32, hidden: 4096, heads: 32, kvHeads: 8, headDim: 128, intermediate: 14336,
    vocab: 128256, tiedEmbeddings: false, maxContext: 131072,
    publishedParamsB: 8.03, source: hf('unsloth/Meta-Llama-3.1-8B-Instruct'),
  },
  {
    id: 'qwen3-8b', name: 'Qwen3 8B', family: 'Qwen',
    layers: 36, hidden: 4096, heads: 32, kvHeads: 8, headDim: 128, intermediate: 12288,
    vocab: 151936, tiedEmbeddings: false, maxContext: 40960,
    publishedParamsB: 8.19, source: hf('Qwen/Qwen3-8B'),
  },
  {
    id: 'qwen3-14b', name: 'Qwen3 14B', family: 'Qwen',
    layers: 40, hidden: 5120, heads: 40, kvHeads: 8, headDim: 128, intermediate: 17408,
    vocab: 151936, tiedEmbeddings: false, maxContext: 40960,
    publishedParamsB: 14.77, source: hf('Qwen/Qwen3-14B'),
  },
  {
    id: 'gpt-oss-20b', name: 'gpt-oss 20B', family: 'gpt-oss',
    layers: 24, hidden: 2880, heads: 64, kvHeads: 8, headDim: 64, intermediate: 2880,
    vocab: 201088, tiedEmbeddings: false, maxContext: 131072,
    moe: { experts: 32, activeExperts: 4, expertIntermediate: 2880 },
    slidingWindow: { window: 128, fraction: 0.5 },
    publishedParamsB: 20.91, publishedActiveB: 3.61, defaultQuant: 'mxfp4',
    source: hf('openai/gpt-oss-20b'),
  },
  {
    id: 'mistral-small-3.1-24b', name: 'Mistral Small 3.1 24B', family: 'Mistral',
    layers: 40, hidden: 5120, heads: 32, kvHeads: 8, headDim: 128, intermediate: 32768,
    vocab: 131072, tiedEmbeddings: false, maxContext: 131072,
    // 24.0B published includes the ~0.4B vision encoder; text decoder is ~23.6B
    publishedParamsB: 23.57, source: hf('mistralai/Mistral-Small-3.1-24B-Instruct-2503'),
  },
  {
    id: 'gemma-3-27b', name: 'Gemma 3 27B', family: 'Gemma',
    layers: 62, hidden: 5376, heads: 32, kvHeads: 16, headDim: 128, intermediate: 21504,
    vocab: 262208, tiedEmbeddings: true, maxContext: 131072,
    slidingWindow: { window: 1024, fraction: 5 / 6 },
    // 27.4B published includes the ~0.4B SigLIP vision encoder
    publishedParamsB: 27.01, source: hf('unsloth/gemma-3-27b-it'),
  },
  {
    id: 'qwen3-30b-a3b', name: 'Qwen3 30B-A3B', family: 'Qwen',
    layers: 48, hidden: 2048, heads: 32, kvHeads: 4, headDim: 128, intermediate: 6144,
    vocab: 151936, tiedEmbeddings: false, maxContext: 40960,
    moe: { experts: 128, activeExperts: 8, expertIntermediate: 768 },
    // Qwen's "3.3B activated" also counts the input embedding table, which a token only reads one row of
    publishedParamsB: 30.53, source: hf('Qwen/Qwen3-30B-A3B'),
  },
  {
    id: 'qwen3-32b', name: 'Qwen3 32B', family: 'Qwen',
    layers: 64, hidden: 5120, heads: 64, kvHeads: 8, headDim: 128, intermediate: 25600,
    vocab: 151936, tiedEmbeddings: false, maxContext: 40960,
    publishedParamsB: 32.76, source: hf('Qwen/Qwen3-32B'),
  },
  {
    id: 'llama-3.3-70b', name: 'Llama 3.3 70B', family: 'Llama',
    layers: 80, hidden: 8192, heads: 64, kvHeads: 8, headDim: 128, intermediate: 28672,
    vocab: 128256, tiedEmbeddings: false, maxContext: 131072,
    publishedParamsB: 70.55, source: hf('unsloth/Llama-3.3-70B-Instruct'),
  },
  {
    id: 'qwen2.5-72b', name: 'Qwen2.5 72B', family: 'Qwen',
    layers: 80, hidden: 8192, heads: 64, kvHeads: 8, headDim: 128, intermediate: 29568,
    vocab: 152064, tiedEmbeddings: false, maxContext: 32768,
    publishedParamsB: 72.71, source: hf('Qwen/Qwen2.5-72B-Instruct'),
  },
  {
    id: 'gpt-oss-120b', name: 'gpt-oss 120B', family: 'gpt-oss',
    layers: 36, hidden: 2880, heads: 64, kvHeads: 8, headDim: 64, intermediate: 2880,
    vocab: 201088, tiedEmbeddings: false, maxContext: 131072,
    moe: { experts: 128, activeExperts: 4, expertIntermediate: 2880 },
    slidingWindow: { window: 128, fraction: 0.5 },
    publishedParamsB: 116.83, publishedActiveB: 5.13, defaultQuant: 'mxfp4',
    source: hf('openai/gpt-oss-120b'),
  },
]

export const modelById = (id: string) => MODELS.find(m => m.id === id)
