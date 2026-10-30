export function Methodology() {
  return (
    <details className="card method">
      <summary>How these numbers are calculated</summary>
      <p>
        <b>Weights.</b> Parameter counts are derived from each model's Hugging Face <code>config.json</code> and
        checked against the published totals. GGUF bits-per-weight are measured from real files (bartowski's
        Llama 3.1 8B and 3.3 70B quants), so Q4_K_M means 4.86 bits on average, not 4. The test suite checks
        predicted sizes against 20 real GGUF files to within 3.5%.
      </p>
      <p>
        <b>KV cache.</b> <code>2 x layers x KV heads x head dim x bytes x tokens</code>. Models with
        sliding-window layers (Gemma 3, gpt-oss) only keep the window on those layers, as llama.cpp does by default.
      </p>
      <p>
        <b>Compute buffer and runtime.</b> llama.cpp's compute buffer is mostly f32 logits for one ubatch
        (<code>4 x ubatch x vocab</code>), plus the full attention score matrix when flash attention is off.
        The CUDA context costs about 0.35 GiB before anything loads. vLLM reserves
        <code> gpu_memory_utilization</code> of the card, subtracts weights, its profiled activation peak and
        CUDA graphs, and turns the rest into KV pool.
      </p>
      <p>
        <b>Generation speed.</b> Each token streams the active weights and the KV cache through memory once, so
        <code> time = bytes / (bandwidth x 0.89) + 34 µs per layer</code>. Both constants are fit to the{' '}
        <a href="https://github.com/ggml-org/llama.cpp/discussions/15013">llama.cpp CUDA scoreboard</a>, and
        predictions land within 10% of measured for every consumer and workstation card listed there.
        HBM datacenter cards get a lower efficiency (0.48) because llama.cpp reaches less of their peak.
      </p>
      <p>
        <b>Prompt speed</b> is compute bound, so it is only shown for GPUs with a measured llama.cpp result,
        scaled by active parameters.
      </p>
    </details>
  )
}
