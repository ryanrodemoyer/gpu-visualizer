import { describe, expect, it } from 'vitest'
import { MODELS, modelById } from '../data/models'
import { gpuById } from '../data/gpus'
import { kvTypeById, quantById } from '../data/quants'
import { LLAMA2_7B_Q4_0 } from '../data/benchmarks'
import { paramBreakdown } from './params'
import {
  DEFAULT_SETTINGS, GiB, decodeTokensPerSec, kvCacheBytes, memoryPlan, weightBytes, type Settings,
} from './calc'

const model = (id: string) => modelById(id)!
const quant = (id: string) => quantById(id)!
const kv = (id: string) => kvTypeById(id)!
const relErr = (a: number, b: number) => Math.abs(a / b - 1)

const llamaSettings = (over: Partial<Settings> = {}): Settings => ({
  ...DEFAULT_SETTINGS, quant: quant('q4_k_m'), kvType: kv('f16'), ...over,
})

describe('parameter counts derived from config.json match published totals', () => {
  for (const m of MODELS) {
    it(m.name, () => {
      const p = paramBreakdown(m)
      expect(relErr(p.total / 1e9, m.publishedParamsB)).toBeLessThan(0.015)
      if (m.publishedActiveB) expect(relErr(p.active / 1e9, m.publishedActiveB)).toBeLessThan(0.03)
    })
  }
})

describe('weight size matches real GGUF files', () => {
  // [model, quant, file size in GB (1e9 bytes)] from bartowski / ggml-org Hugging Face repos
  const files: [string, string, number][] = [
    ['llama-3.1-8b', 'q8_0', 8.54], ['llama-3.1-8b', 'q6_k', 6.6], ['llama-3.1-8b', 'q5_k_m', 5.73],
    ['llama-3.1-8b', 'q4_k_m', 4.92], ['llama-3.1-8b', 'iq4_xs', 4.45], ['llama-3.1-8b', 'q3_k_m', 4.02],
    ['llama-3.1-8b', 'q2_k', 3.18], ['llama-3.1-8b', 'f16', 16.07],
    ['llama-3.3-70b', 'q8_0', 74.98], ['llama-3.3-70b', 'q6_k', 57.89], ['llama-3.3-70b', 'q5_k_m', 49.95],
    ['llama-3.3-70b', 'q4_k_m', 42.52], ['llama-3.3-70b', 'q4_0', 40.12], ['llama-3.3-70b', 'iq4_xs', 37.9],
    ['llama-3.3-70b', 'q3_k_m', 34.27], ['llama-3.3-70b', 'q2_k', 26.38], ['llama-3.3-70b', 'f16', 141.12],
    ['llama-2-7b', 'q4_0', 3.56 * GiB / 1e9],
    ['gpt-oss-20b', 'mxfp4', 12.1], ['gpt-oss-120b', 'mxfp4', 63.4],
  ]
  for (const [m, q, gb] of files) {
    it(`${m} ${q} ≈ ${gb.toFixed(2)} GB`, () => {
      expect(relErr(weightBytes(model(m), quant(q)) / 1e9, gb)).toBeLessThan(0.035)
    })
  }
})

describe('KV cache', () => {
  it('Llama 3.1 8B, 8192 ctx, f16 = 1024 MiB (llama.cpp log value)', () => {
    expect(kvCacheBytes(model('llama-3.1-8b'), kv('f16'), 8192, 1)).toBe(1024 * 2 ** 20)
  })
  it('q8_0 KV is 34/64 the size of f16', () => {
    const m = model('qwen3-32b')
    expect(kvCacheBytes(m, kv('q8_0'), 16384, 1) / kvCacheBytes(m, kv('f16'), 16384, 1)).toBeCloseTo(34 / 64)
  })
  it('parallel sequences multiply KV linearly for full-attention models', () => {
    const m = model('llama-3.1-8b')
    expect(kvCacheBytes(m, kv('f16'), 4096, 4)).toBe(4 * kvCacheBytes(m, kv('f16'), 4096, 1))
  })
  it('sliding-window layers cap Gemma 3 KV well below a full-attention estimate', () => {
    const m = model('gemma-3-27b')
    const full = 62 * 2 * 16 * 128 * 2 * 32768
    const actual = kvCacheBytes(m, kv('f16'), 32768, 1)
    expect(actual).toBeLessThan(full * 0.25)
    expect(actual).toBeGreaterThan(full / 6)
  })
})

describe('memory plan', () => {
  it('Llama 3.1 8B Q4_K_M at 8k fits a 12 GB 3060 with room to spare', () => {
    const plan = memoryPlan(gpuById('rtx-3060-12')!, model('llama-3.1-8b'), llamaSettings())
    expect(plan.fits).toBe(true)
    expect(plan.used / GiB).toBeGreaterThan(5.5)
    expect(plan.used / GiB).toBeLessThan(7.5)
  })
  it('Llama 3.3 70B Q4_K_M does not fit a single 24 GB card', () => {
    const plan = memoryPlan(gpuById('rtx-4090')!, model('llama-3.3-70b'), llamaSettings())
    expect(plan.fits).toBe(false)
    expect(plan.maxContext).toBe(0)
  })
  it('max context is the boundary: it fits, and 512 more does not', () => {
    const gpu = gpuById('rtx-3090')!
    const m = model('qwen3-32b')
    const s = llamaSettings()
    const max = memoryPlan(gpu, m, s).maxContext
    expect(max).toBeGreaterThan(8192)
    expect(memoryPlan(gpu, m, { ...s, context: max }).fits).toBe(true)
    expect(memoryPlan(gpu, m, { ...s, context: max + 512 }).fits).toBe(false)
  })
  it('turning off flash attention costs compute buffer memory', () => {
    const gpu = gpuById('rtx-4090')!
    const m = model('llama-3.1-8b')
    const on = memoryPlan(gpu, m, llamaSettings({ context: 32768 })).used
    const off = memoryPlan(gpu, m, llamaSettings({ context: 32768, flashAttn: false })).used
    expect(off - on).toBeGreaterThan(1 * GiB)
  })
  it('segments add up to capacity when the model fits', () => {
    const plan = memoryPlan(gpuById('rtx-5090')!, model('qwen3-32b'), llamaSettings())
    const sum = plan.segments.reduce((a, s) => a + s.bytes, 0)
    expect(relErr(sum, plan.capacity)).toBeLessThan(1e-9)
  })
  it('vLLM: KV pool fills the gpu_memory_utilization budget', () => {
    const s: Settings = { ...DEFAULT_SETTINGS, engine: 'vllm', quant: quant('vllm-bf16'), kvType: kv('auto') }
    const plan = memoryPlan(gpuById('h100-sxm')!, model('llama-3.1-8b'), s)
    expect(plan.fits).toBe(true)
    // ~72 GiB budget - ~15 GiB weights - overheads leaves ~54 GiB of KV at 128 KiB/token
    expect(plan.kvPoolTokens! / 1000).toBeGreaterThan(380)
    expect(plan.kvPoolTokens! / 1000).toBeLessThan(470)
    expect(plan.maxConcurrency).toBeGreaterThan(40)
  })
})

describe('decode speed vs the llama.cpp CUDA scoreboard (Llama 2 7B Q4_0, tg128)', () => {
  const m = model('llama-2-7b')
  const s = llamaSettings({ quant: quant('q4_0') })
  for (const r of LLAMA2_7B_Q4_0) {
    it(`${r.gpuId}: measured ${r.tg128} t/s`, () => {
      const pred = decodeTokensPerSec(gpuById(r.gpuId)!, m, s, 64)
      expect(relErr(pred, r.tg128)).toBeLessThan(0.1)
    })
  }
})
