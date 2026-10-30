// Measured llama.cpp results used to calibrate and test the speed model.
// Source: llama.cpp CUDA scoreboard, Llama 2 7B Q4_0 (3.56 GiB), flash attention on.
// https://github.com/ggml-org/llama.cpp/discussions/15013
// Entries come from different commits and contributors, so expect a few % of noise.

export interface ScoreboardEntry {
  gpuId: string
  pp512: number
  tg128: number
}

export const SCOREBOARD_MODEL_BYTES = 3.56 * 2 ** 30

export const LLAMA2_7B_Q4_0: ScoreboardEntry[] = [
  { gpuId: 'rtx-3060-12', pp512: 2407.67, tg128: 76.92 },
  { gpuId: 'rtx-3090', pp512: 5560.06, tg128: 161.89 },
  { gpuId: 'rtx-3090-ti', pp512: 6924.01, tg128: 172.26 },
  { gpuId: 'rtx-4060-ti-16', pp512: 3803.45, tg128: 64.03 },
  { gpuId: 'rtx-4080', pp512: 9205.93, tg128: 143.47 },
  { gpuId: 'rtx-4090', pp512: 14770.63, tg128: 188.96 },
  { gpuId: 'rtx-5080', pp512: 9487.7, tg128: 184.68 },
  { gpuId: 'rtx-5090', pp512: 14970.15, tg128: 300.4 },
  { gpuId: 'a100-80-sxm', pp512: 5285.96, tg128: 200.9 },
  { gpuId: 'rtx-6000-ada', pp512: 10576.85, tg128: 179.47 },
  { gpuId: 'rtx-pro-6000', pp512: 16618.98, tg128: 281.11 },
]
