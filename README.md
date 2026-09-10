# GLM-5.3 (uncensored) on 8x NVIDIA DGX Spark (GB10) with vLLM TP=8

Launchers, measurement scripts and ops helpers behind the field report
[dealignai/GLM-5.3-UNCENSORED-FP8, discussion #3](https://huggingface.co/dealignai/GLM-5.3-UNCENSORED-FP8/discussions/3).
Everything here was run on real hardware; numbers quoted are measurements, with the prompt and run count stated.

## Hardware and stack

- 8x DGX Spark GB10, 121.69 GiB unified memory each, 2x200G RoCE fabric, one GPU per node
- vLLM tensor parallel 8 across the nodes, multiprocess executor (`--nnodes 8 --node-rank N`, no Ray)
- Image: `ghcr.io/ciprianveg/gb10-glm-5.2:v19-vision` (ciprianveg's gb10-vllm, a vLLM 0.11.2 fork with the B12X sparse-MLA kernels for SM121)
- Models: `dealignai/GLM-5.3-UNCENSORED-NVFP4` (daily driver), `dealignai/GLM-5.3-UNCENSORED-FP8` (kept on disk)

## The config that runs (NVFP4, 512K context)

From `launchers/launch-nvfp4-tp8.sh`, each flag with the reason it is set that way:

| flag | value | why |
|---|---|---|
| `--gpu-memory-utilization` | 0.80 | On GB10 unified memory the limit is host headroom, not a GPU ceiling. Each +0.01 costs about 1.3 GB of host headroom; the head node idles at ~8 GB available at 0.80 and ~5 GB at 0.82. 0.86 OOM-killed the first boot. |
| `--max-model-len` | 524288 | KV pool at 0.80 is ~36 GiB/rank = ~695K tokens with `fp8_ds_mla`, i.e. 1.33x concurrency at 512K. |
| `--kv-cache-dtype` | `fp8_ds_mla` | `nvfp4_ds_mla` is ~1.5x denser and works at plain TP8 on this stack (Light Foundry: 944K-token pool), but it is a quality trade on 4-bit weights; untested here at time of writing. |
| `--speculative-config` | MTP, k=1, `B12X_MLA_SPARSE` draft backend | Works on this fork despite the FP8 card saying MTP is non-functional. +48% decode on realistic coding prompts (FP8, adaptive k up to 5); 1.70 tokens accepted per step at k=1 on NVFP4. |
| `--default-chat-template-kwargs` | `{"reasoning_effort":"low"}` | The template honours only `low` and `high`; anything else (including `off`) falls through to `max`. See the trap below. |
| `--compilation-config` | `{"cudagraph_mode":"FULL","max_cudagraph_capture_size":30}` | Graph capture is cheap here (~5 s, ~0.4 GiB). |
| `--decode-context-parallel-size` | 1 | DCP>1 is closed for GLM-5.3 on this stack: the DSA indexer cache is DCP-replicated while MLA KV is sharded and the KV page-size unifier cannot reconcile them (`page size is not divisible ... cannot be padded`), for both KV dtypes. |
| `--pipeline-parallel-size` | 1 | PP2 x TP4 profiles fine (42 GiB KV/rank) but the MTP draft class lacks `SupportsPP`, and without MTP the multi-node run hung after warmup under the mp executor. |

Measured envelope (single-stream unless stated, realistic coding prompts):

| | FP8 v2 | NVFP4 |
|---|---|---|
| weights per rank | 88 GiB | 50-57 GiB |
| KV pool per rank | 7-9 GiB (131K-160K context) | 35-36 GiB (695K tokens) |
| decode, no MTP | 15.2-15.6 tok/s | not re-measured |
| decode, MTP | 22.8 tok/s | 21.6 tok/s |
| reasoning `high` at a 16K budget | empty answer (budget exhausted thinking) | complete answer, 8.2K tokens |
| reasoning `max` at a 16K budget | empty answer | answer truncated at the cap after ~13K tokens of thinking |

## The traps, in one place

1. **`reasoning_effort`**: only `low` and `high` are honoured; `off`, `medium`, `max` or an absent kwarg all become `max`. In YAML a bare `off:` parses as boolean false and never matches a string lookup either. Use `low` for agent and tool-loop work.
2. **Reasoning field name**: the thinking comes back in `message.reasoning`, not `message.reasoning_content`.
3. **FP8 at `high`/`max`**: the model thinks until `max_tokens` is exhausted and returns no answer (`finish_reason: length`, empty content). Sampling parameters do not rescue it. A truncated turn fed back as history is what produces the long "TheTheThe" repetition people report.
4. **Benchmark prompts**: a "count to 300" prompt inflates MTP acceptance enormously (+140% versus +48% on real work). Prefill run-to-run variance on this cluster is ~34%; decode ~3%. Use repeated controls (A-B-C-A) before believing a prefill number.
5. **`VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR` is a no-op at TP>1** on this build: `kernel_warmup.py::flashinfer_autotune` disables the persistent cache when world size > 1, so every boot re-tunes (~5 min) and the KV pool varies by a few GiB between identical boots.
6. **Never run a bulk download on a serving rank.** Twice it starved sshd and killed the TP job. Pull weights while nothing is serving.
7. **`sudo -n` can lie.** A cached ticket makes a "passwordless" test pass while cron gets denied. Test with `sudo -k` first. The scoped NOPASSWD rule in `ops/prism-spark-ops.sudoers` covers exactly the page-cache drop and `nvidia-smi`.
8. **zsh does not word-split unquoted variables** (`S="ssh ..."; $S host cmd` is one "command not found"), and `pkill -f <pattern>` from an ssh string that contains the pattern kills your own shell. Ship a script file and kill by PID.

## What is in here

- `launchers/` -- the three node launchers: FP8 TP8 (131K, MTP), NVFP4 TP8 (512K, the live config), and the NVFP4 DCP8 variant kept as the record of a failed 1M attempt. Run on every node with its rank; workers 7..1 first, head 0 last.
- `tools/` -- measurement: `bench_decode_real.py` (single-stream decode on realistic prompts), `bench_concurrency.py`, `bench_clocks.py` (clock-lock A-B-C-A), `test_repetition.py` and `diag_rep_at_*.py` (long-generation repetition harness), `test_max_effort.py` and `diag_effort.py` (reasoning-effort at a fixed budget), `check_reasoning.py` and `diag_raw.py` (which field the reasoning lands in, raw completions), `single64k.py`, `test_longctx.py`, `test_depth_coherence.py`, `probe_1m.py` (needle at depth), `sweep.py` (overnight recipe sweep), `nccl_allreduce.py` and `run_nccl8.sh` (fabric sanity), `run_graphs_suite.sh`, `run_condition.sh`, `nvfp4_suite.sh`, `sample_telemetry.sh`, `gid_probe.sh`.
- `ops/` -- page-cache flushers (boot-time and persistent cron), the sudoers rule and installer, GB10 clock lock (`set_clocks.sh`, `gb10-clocklock.service`; `-lgc 0,2200` cuts cluster power 32-38% with decode unchanged), and `lib.sh`.

Scripts point at the head node by IP (`10.100.128.10`) and use `$USER@<node>` for ssh; edit those for your fabric. There are no credentials anywhere in this repository.

## Credits

dealignai for the uncensored builds; ciprianveg (gb10-vllm) and the b12x kernel authors for the SM121 stack; Light Foundry, Tech2wild, local-inference-lab and the rest of the GB10 crowd whose recipes this leaned on.
