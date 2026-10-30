# GPU VRAM Visualizer

Pick a model, a quantization and the settings you'd pass to llama.cpp or vLLM, then compare GPUs side by side: how VRAM splits into weights, KV cache, compute buffers and runtime overhead, the longest context that fits, and estimated tokens/sec.

Live site: https://ryanrodemoyer.github.io/gpu-visualizer/

## How it's grounded

- **GPUs** (`src/data/gpus.ts`): VRAM and bandwidth from vendor spec sheets / TechPowerUp, with a source per row.
- **Models** (`src/data/models.ts`): architectures copied from each model's Hugging Face `config.json`. Parameter counts are derived from the architecture and tested against published totals.
- **Quants** (`src/data/quants.ts`): GGUF bits-per-weight measured from real files, tested against 20 GGUF file sizes.
- **Speed** (`src/lib/calc.ts`): bandwidth-bound decode model fit to the [llama.cpp CUDA scoreboard](https://github.com/ggml-org/llama.cpp/discussions/15013) and tested to within 10% of measured results.

## Develop

```sh
npm install
npm run dev     # local dev server
npm test        # calculator test suite
npm run build   # static build in dist/
```

Pushes to `main` run the tests and deploy to GitHub Pages.

## License

MIT
