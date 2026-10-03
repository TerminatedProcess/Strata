# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

See also `AGENTS.md` (installing Strata for a user via `docs/AI_SETUP.md`; never expose the server beyond
`127.0.0.1` without `--api-key`).

## What this is

Strata runs the Qwen3.8-Flash-Next MoE model (plus Coder, Swift 1.5, Unsloth variants) on one consumer GPU
(NVIDIA CUDA or AMD HIP) + system RAM, Windows or Linux. Three layers:

1. **C++/CUDA/HIP engine** (`src/`, headers in `include/strata/`, same subdirectory layout). The main binary is
   `strata` (`src/program/generate.cpp`, the driver that composes everything into tokens). Experts are tiered:
   hottest on the GPU (expert cache), all in pinned RAM with a CPU expert pool working in parallel, a large lookup
   table (PLE / n-gram) read from SSD. Speculative decoding (MTP/draft, `src/spec`, `src/ngram`) gives the same
   output 1.6-1.8x sooner. `src/prefill` is the batched prompt path; `src/kernels` holds GPU/CPU kernels plus
   `*_parity.cpp` tests against reference implementations. ggml (llama.cpp) is pulled at a pinned commit
   (`third_party/ggml/VERSION.txt`) or from `-DSTRATA_GGML_DIR=<llama.cpp checkout>`.
2. **Python server** (`serve/server.py`, stdlib `http.server`, no web framework): OpenAI (`/v1/chat/completions`)
   and Anthropic (`/v1/messages`) APIs plus the web app (`serve/web`). It keeps one `strata --serve` process
   resident and talks to it over **stdin/stdout** with a line protocol (`GEN <ids>`, `GENI` for image embeddings,
   `STOP`, `QUIT`). Tokenization and chat templating happen in Python (`serve/frontend.py`,
   `serve/chat_template.jinja`, `tools/strata_tokenizer.py`) — the engine only sees token ids. One request at a
   time behind a FIFO; requests exceeding context are rejected with 400, never truncated. `MockEngine`
   (`--engine mock`) makes every API path testable without a GPU. MCP tool hub: `serve/mcp.py`; images go through
   the separate `strata-vision` binary (`tools/vision`).
3. **Installer** `setup.py` (started by `START-HERE.bat` / `setup.sh`; `UPDATE.bat` / `update.sh` update only):
   detects the GPU, picks the model, downloads it, downloads a prebuilt engine or builds one with CMake+Ninja, and
   writes the run config. It fingerprints engine sources (`ENGINE_SOURCES`) into `engine/BUILD.json` and rebuilds
   when they change. `tools/strata_mcp.py` exposes the same install/start/stop steps as an MCP server.

Model packing (`tools/strata_pack.py`, `pack_layer.py`, `iq_pack.py`, `mtp_pack.py`, ...) converts GGUF into the
pack format; the Python packer is the spec of record, and a pack must pass `verify` (bit-exact vs. ggml dequant).

## Commands

Python tests (stdlib `unittest`, no GPU, no downloads). Needs the pinned deps from `requirements.txt` (setup
puts them in `.venv`; e.g. `python -m venv .venv && .venv/bin/pip install -r requirements.txt`, then use
`.venv/bin/python`):

```sh
python -m unittest serve.test_server -v          # one module; also test_security, test_mcp, test_structured, ...
python -m unittest serve.test_server.MaxTokens.test_anthropic_thinks_only_when_asked
python tools/test_setup_amd.py                   # setup tests: tools/test_setup_<name>.py
python -m unittest discover -s tools -p 'test_*.py'
python -m serve.server --engine mock --port 8095 # run the API against the scripted engine
```

Engine build (CUDA; default arch 120, sm_80+ required unless `STRATA_EXPERIMENTAL_SM60`):

```sh
cmake -S . -B build -G Ninja -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=Release
cmake --build build --target strata
```

HIP (AMD): `-DSTRATA_ENABLE_HIP=ON` (exclusive with CUDA), see `docs/AMD_HIP.md` for the full command and
`ctest --test-dir build-hip --output-on-failure`. Most C++ tests/parity checks register with ctest only when
`STRATA_BUILD_TESTS=ON`, which defaults ON only if `tests/CMakeLists.txt` and `bench/micro/` exist — the
published tree omits them, so in this checkout the full C++ suite is not buildable.

## Conventions

- Docs style: plain words, measured numbers stated with the hardware they were measured on, no claims without a
  measurement. Detailed numbers, API and settings live in `docs/DETAILS.md`.
- Code comments explain *why* and cite issue numbers (`#31`, `issue #45`); keep that habit.
- Version is set in `CMakeLists.txt` (`project(strata VERSION ...)`).

## This machine's deployment

- Runs as systemd user unit `strata.service` (Manual Start, never enabled), PortHub lease `8891 strata`, in the
  Service Dashboard's `llm-gateways` group. The unit runs the same command as `run-iq2_xs.sh` minus `--open`.
- Model: `--family qwen --model IQ2_XS`, vision on, context 65536; config `strata-iq2_xs.json`, model files in
  `/mnt/llm/Strata-data`. Change settings with `./setup.sh --setup ...`, then restart the unit.
- Holds ~39 GB RAM + ~8.5 GB VRAM while running: stop it before ComfyUI / llama-cpp work.
- Engine builds against its own pinned llama.cpp in `third_party/`, not the fork at `~/work/ai/llama.cpp`
  (319 commits ahead of the pin). If setup's GitHub zip download crawls, fetch it with curl into
  `third_party/llama.cpp-<sha7>.zip` and write a `.zip.done` mark; setup then skips it.
