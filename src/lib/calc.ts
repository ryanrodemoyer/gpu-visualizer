import type { Gpu } from '../data/gpus'
import type { Model } from '../data/models'
import type { KvType, Quant } from '../data/quants'
import { LLAMA2_7B_Q4_0 } from '../data/benchmarks'
import { paramBreakdown } from './params'

export const GiB = 2 ** 30

// ---------- weights ----------

export function weightBytes(m: Model, q: Quant): number {
  const p = paramBreakdown(m)
  const embedBits = p.embed * (q.embedBpw ?? q.bpw)
  const blockBits = (p.attention + p.dense) * q.bpw
  const expertBits = p.experts * (q.expertBpw ?? q.bpw)
  return (embedBits + blockBits + expertBits) / 8
}

/** Weight bytes read to generate one token (output head + attention + active experts). */
export function activeWeightBytes(m: Model, q: Quant): number {
  const p = paramBreakdown(m)
  const bits =
    p.lmHead * (q.embedBpw ?? q.bpw) +
    (p.attention + p.dense) * q.bpw +
    p.activeExperts * (q.expertBpw ?? q.bpw)
  return bits / 8
}

// ---------- KV cache ----------

/** Bytes of K + V for one token in one layer */
export function kvBytesPerTokenPerLayer(m: Model, kv: KvType): number {
  return 2 * m.kvHeads * m.headDim * kv.bytes
}

export function layerSplit(m: Model) {
  const sliding = m.slidingWindow ? Math.round(m.layers * m.slidingWindow.fraction) : 0
  return { sliding, full: m.layers - sliding }
}

/**
 * KV cache size. Full-attention layers store every token; sliding-window layers only keep
 * the window per sequence (llama.cpp default without --swa-full, plus one ubatch of headroom).
 */
export function kvCacheBytes(m: Model, kv: KvType, ctxPerSeq: number, parallel: number, ubatch = 512): number {
  const perLayer = kvBytesPerTokenPerLayer(m, kv)
  const { sliding, full } = layerSplit(m)
  const totalTokens = ctxPerSeq * parallel
  const slidingTokens = m.slidingWindow
    ? Math.min(totalTokens, m.slidingWindow.window * parallel + ubatch)
    : 0
  return perLayer * (full * totalTokens + sliding * slidingTokens)
}

// ---------- engine overheads ----------

/** CUDA context, cuBLAS workspace and driver allocations seen in nvidia-smi before any model loads. */
export const RUNTIME_OVERHEAD_BYTES = 0.35 * GiB

/**
 * llama.cpp compute buffer. The peak is dominated by the f32 logits for one ubatch plus
 * a few hidden-size activations. Without flash attention, the f32 KQ score matrix for
 * one ubatch against the whole context is also materialized.
 */
export function llamaCppComputeBytes(m: Model, ctxPerSeq: number, parallel: number, ubatch: number, flashAttn: boolean): number {
  const logits = 4 * ubatch * m.vocab
  const acts = 4 * ubatch * 3 * m.hidden
  const kq = flashAttn ? 0 : 4 * ubatch * m.heads * ctxPerSeq * parallel
  return logits + acts + kq
}

/** vLLM activation peak from its profiling run (max_num_batched_tokens through one forward pass). */
export function vllmActivationBytes(m: Model, batchedTokens: number, tp: number): number {
  const ffn = m.moe ? m.moe.activeExperts * m.moe.expertIntermediate : m.intermediate
  return (batchedTokens * (2 * ffn + 6 * m.hidden) * 2) / tp
}

export const VLLM_CUDA_GRAPH_BYTES = 0.5 * GiB

// ---------- memory plan ----------

export interface Settings {
  engine: 'llama.cpp' | 'vllm'
  quant: Quant
  kvType: KvType
  /** Context length per sequence */
  context: number
  /** Concurrent sequences (llama.cpp -np / vLLM concurrent requests) */
  parallel: number
  /** llama.cpp -ub */
  ubatch: number
  flashAttn: boolean
  /** vLLM --gpu-memory-utilization */
  gpuMemUtil: number
  /** vLLM --max-num-batched-tokens */
  batchedTokens: number
  /** VRAM already used by the display / other apps, bytes */
  reservedBytes: number
}

export const DEFAULT_SETTINGS: Omit<Settings, 'quant' | 'kvType'> = {
  engine: 'llama.cpp',
  context: 8192,
  parallel: 1,
  ubatch: 512,
  flashAttn: true,
  gpuMemUtil: 0.9,
  batchedTokens: 8192,
  reservedBytes: 0,
}

export type SegmentKey = 'reserved' | 'runtime' | 'weights' | 'kv' | 'compute' | 'kvPool' | 'free'

export interface Segment {
  key: SegmentKey
  label: string
  bytes: number
}

export interface MemoryPlan {
  capacity: number
  segments: Segment[]
  used: number
  fits: boolean
  /** Bytes over capacity (0 if it fits) */
  overflow: number
  /** Longest per-sequence context that fits with the other settings unchanged */
  maxContext: number
  /** vLLM only: tokens the KV pool can hold, and how many sequences of `context` that is */
  kvPoolTokens?: number
  maxConcurrency?: number
}

export function memoryPlan(gpu: Gpu, m: Model, s: Settings): MemoryPlan {
  const capacity = gpu.vramGiB * GiB
  return s.engine === 'vllm' ? vllmPlan(capacity, m, s) : llamaCppPlan(capacity, m, s)
}

function llamaCppFixed(m: Model, s: Settings) {
  return s.reservedBytes + RUNTIME_OVERHEAD_BYTES + weightBytes(m, s.quant)
}

function llamaCppPlan(capacity: number, m: Model, s: Settings): MemoryPlan {
  const weights = weightBytes(m, s.quant)
  const kv = kvCacheBytes(m, s.kvType, s.context, s.parallel, s.ubatch)
  const compute = llamaCppComputeBytes(m, s.context, s.parallel, s.ubatch, s.flashAttn)
  const used = s.reservedBytes + RUNTIME_OVERHEAD_BYTES + weights + kv + compute
  const segments: Segment[] = [
    { key: 'reserved', label: 'Display / other apps', bytes: s.reservedBytes },
    { key: 'runtime', label: 'CUDA runtime', bytes: RUNTIME_OVERHEAD_BYTES },
    { key: 'weights', label: 'Model weights', bytes: weights },
    { key: 'kv', label: 'KV cache', bytes: kv },
    { key: 'compute', label: 'Compute buffer', bytes: compute },
    { key: 'free', label: 'Free', bytes: Math.max(0, capacity - used) },
  ]
  const fits = used <= capacity
  return {
    capacity, segments, used, fits,
    overflow: Math.max(0, used - capacity),
    maxContext: maxContextLlamaCpp(capacity, m, s),
  }
}

function maxContextLlamaCpp(capacity: number, m: Model, s: Settings): number {
  const fixed = llamaCppFixed(m, s)
  const usedAt = (ctx: number) =>
    fixed + kvCacheBytes(m, s.kvType, ctx, s.parallel, s.ubatch) + llamaCppComputeBytes(m, ctx, s.parallel, s.ubatch, s.flashAttn)
  if (usedAt(256) > capacity) return 0
  // Memory grows monotonically with context: binary search, then round down to 256.
  let lo = 256
  let hi = m.maxContext
  if (usedAt(hi) <= capacity) return hi
  while (hi - lo > 256) {
    const mid = Math.floor((lo + hi) / 2)
    if (usedAt(mid) <= capacity) lo = mid
    else hi = mid
  }
  return Math.floor(lo / 256) * 256
}

function vllmPlan(capacity: number, m: Model, s: Settings): MemoryPlan {
  const budget = capacity * s.gpuMemUtil
  const weights = weightBytes(m, s.quant)
  const act = vllmActivationBytes(m, s.batchedTokens, 1)
  const graphs = VLLM_CUDA_GRAPH_BYTES
  const kvPool = budget - s.reservedBytes - RUNTIME_OVERHEAD_BYTES - weights - act - graphs
  const perLayer = kvBytesPerTokenPerLayer(m, s.kvType)
  const { sliding, full } = layerSplit(m)
  // Sliding layers hold at most `window` tokens per sequence; charge that up front per sequence.
  const slidingPerSeq = m.slidingWindow ? perLayer * sliding * Math.min(m.slidingWindow.window, s.context) : 0
  const fullPerToken = perLayer * full
  const kvPoolTokens = kvPool > 0 ? Math.floor((kvPool - slidingPerSeq * s.parallel) / fullPerToken) : 0
  const needed = kvCacheBytes(m, s.kvType, s.context, s.parallel, 0)
  const used = s.reservedBytes + RUNTIME_OVERHEAD_BYTES + weights + act + graphs + Math.max(0, kvPool)
  const segments: Segment[] = [
    { key: 'reserved', label: 'Display / other apps', bytes: s.reservedBytes },
    { key: 'runtime', label: 'CUDA runtime', bytes: RUNTIME_OVERHEAD_BYTES },
    { key: 'weights', label: 'Model weights', bytes: weights },
    { key: 'compute', label: 'Activations + CUDA graphs', bytes: act + graphs },
    { key: 'kv', label: 'KV cache (requested)', bytes: Math.min(needed, Math.max(0, kvPool)) },
    { key: 'kvPool', label: 'KV pool (spare)', bytes: Math.max(0, kvPool - needed) },
    { key: 'free', label: 'Outside vLLM budget', bytes: Math.max(0, capacity - Math.max(used, budget)) },
  ]
  const fits = kvPool > 0 && kvPool >= needed
  const perSeqBytes = fullPerToken * s.context + slidingPerSeq
  const slidingMax = m.slidingWindow ? perLayer * sliding * m.slidingWindow.window : 0
  const maxContext = kvPool > 0
    ? Math.max(0, Math.min(m.maxContext, Math.floor((kvPool / s.parallel - slidingMax) / fullPerToken / 256) * 256))
    : 0
  return {
    capacity, segments, used, fits,
    overflow: fits ? 0 : needed - Math.max(0, kvPool),
    maxContext,
    kvPoolTokens: Math.max(0, kvPoolTokens),
    maxConcurrency: kvPool > 0 ? Math.floor(kvPool / perSeqBytes) : 0,
  }
}

// ---------- speed ----------

/**
 * Decode is memory-bandwidth bound: every generated token streams the active weights and the
 * KV cache through the GPU once. Time per token = bytes / (bandwidth * efficiency) + a fixed
 * per-layer cost (kernel launches, syncs) that dominates for small models on fast GPUs.
 * Constants are fit to the llama.cpp CUDA scoreboard (see speed.test.ts).
 */
export const DECODE_EFFICIENCY = 0.89
/** llama.cpp reaches a much smaller share of peak on HBM datacenter parts (fit to the A100 entry). */
export const DECODE_EFFICIENCY_HBM = 0.48
export const PER_LAYER_OVERHEAD_S = 34e-6

/** Bytes streamed from VRAM to generate one token with `depth` tokens already in context. */
export function bytesPerToken(m: Model, s: Settings, depth = 0): number {
  return activeWeightBytes(m, s.quant) + kvCacheBytes(m, s.kvType, Math.max(depth, 1), 1, 0)
}

export function decodeTokensPerSec(gpu: Gpu, m: Model, s: Settings, depth = 0): number {
  const eff = gpu.memoryType.startsWith('HBM') ? DECODE_EFFICIENCY_HBM : DECODE_EFFICIENCY
  const seconds = bytesPerToken(m, s, depth) / (gpu.bandwidthGBs * 1e9 * eff) + m.layers * PER_LAYER_OVERHEAD_S
  return 1 / seconds
}

/**
 * Prompt processing is compute bound, so it can't be derived from bandwidth. Where the GPU has
 * a measured llama.cpp pp512 result, scale it by active parameters relative to Llama 2 7B.
 */
export function prefillTokensPerSec(gpu: Gpu, m: Model): number | null {
  const ref = LLAMA2_7B_Q4_0.find(e => e.gpuId === gpu.id)
  if (!ref) return null
  return (ref.pp512 * 6.74e9) / paramBreakdown(m).active
}
