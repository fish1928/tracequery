# TraceQuery

TraceQuery speeds up diffusion LLM decoding with a small learned router. During cached decoding, the router uses the attention trace of the token that was just unmasked to predict which masked positions will unmask next. Only those positions, plus the periodically refreshed ones, are recomputed at each step.

This README walks through the basic pipeline for **LLaDA-8B-Base** on **GSM8K**:

1. Save the benchmark requests to a CSV
2. Collect oracle traces with full (uncached) denoising
3. Train the MLP router on the traces
4. Evaluate with the router in lm-eval, and compare against the dense baseline

## Setup

```bash
pip install torch transformers accelerate datasets lm_eval
# optional: pip install flash-attn
```

The model `GSAI-ML/LLaDA-8B-Base` downloads from the Hugging Face Hub on first use. Every step below needs one GPU.

## 1. Save the benchmark requests

lm-eval builds the real requests (few-shot context, stop strings), and a mock model writes them to a CSV without running anything. The CSV keeps only the **last** `percent` of documents. `lm_eval --limit N` evaluates the **first** N documents, so the training docs never overlap the eval docs.

```bash
python save_benchmark_mockup.py --tasks gsm8k --model mockup --num_fewshot 5 \
    --model_args percent=0.1,folder_output=benchmark_mockup,tag=5shot \
    --predict_only --output_path benchmark_mockup/lm_eval_logs
```

Output: `benchmark_mockup/mockup_gsm8k_5shot_p10.csv` and a `.meta.json` file next to it.

## 2. Collect oracle traces

This step runs full denoising with no cache over the CSV prompts. It saves the following for every sample:

- the attention rows of each step
- confidence, margin, and entropy
- the order in which positions unmask
- the generated text and whether the answer is correct

```bash
python run_collect_metrics_llada_base.py \
    --path_mockup benchmark_mockup/mockup_gsm8k_5shot_p10.csv \
    --folder_output stats_oracle/llada_base_gsm8k_b1 \
    --len_target 256 --num_blocks 1 --device cuda:0
```

Output: one folder per sample, `stats_oracle/llada_base_gsm8k_b1/<id>/`. The run is resume-safe: samples that already have `generated.json` are skipped. Add `--limit N` for a quick test.

## 3. Train the router

`run_train_mlp.py` can train a whole grid of routers. Use `GRID_FILTER` to train only the recommended setting:

- features: attention of the last unmasked token, plus position and mask density
- normalization: softmax
- loss: Plackett-Luce

The script reads both `FOLDER_DATA_GSM8K` and `FOLDER_DATA_IFEVAL`. For a GSM8K-only run, point both at the same folder. The IFEval folder is only read by `data-ifeval` and `data-mix` routers, which `GRID_FILTER` skips.

```bash
FOLDER_DATA_GSM8K=stats_oracle/llada_base_gsm8k_b1 \
FOLDER_DATA_IFEVAL=stats_oracle/llada_base_gsm8k_b1 \
SIZE_BLOCK_GSM8K=256 \
GRID_FILTER=feat-attn_last_geo__conf-none__norm-softmax_attn__loss-plackett_luce__data-gsm8k \
DEVICE=cuda:0 FOLDER_OUT=routers \
python run_train_mlp.py
```

Set `SIZE_BLOCK_GSM8K` to the `--len_target` you used in step 2, since that run used `num_blocks=1`.

Output: `routers/<name>.pt` (weights) and `routers/<name>.json` (feature spec and held-out recall@5). Check the log for `saved ...` lines. A version that fails prints `FAILED` and the script moves on to the next one.

## 4. Evaluate

Run lm-eval through `run_benchmark_main.py` with the router runner:

```bash
ROUTER=routers/feat-attn_last_geo__conf-none__norm-softmax_attn__loss-plackett_luce__data-gsm8k.pt

accelerate launch --num_processes=1 run_benchmark_main.py \
    --tasks gsm8k --num_fewshot 5 --limit 100 --model test --batch_size 1 --device cuda:0 \
    --output_path results/router_gsm8k \
    --model_args "id_model=GSAI-ML/LLaDA-8B-Base,size_batch=1,len_target=256,num_blocks=1,num_unmask_per_step=1,id_mask=126336,step_refresh_remainder=16,step_refresh_remainder_prompt=96,select_only_in_h=True,h=5,runner=run_llada_semi_mlp_v2_batch,path_router=$ROUTER,path_report=results/router_gsm8k__runner.json"
```

Key `model_args`:

| arg | meaning |
|---|---|
| `h` | number of future positions the router selects at each step |
| `step_refresh_remainder` | fully refresh the generation area every K steps |
| `step_refresh_remainder_prompt` | refresh the prompt KV every K steps (0 = never) |
| `path_router` | trained router bundle from step 3 (required) |
| `path_report` | JSON file with the time and finish flag for each sample |

**Dense baseline:** always compare against full denoising under the same settings. The dense runner is `run_llada_semi.py`, so swap the router args for `runner=run_llada_semi` and keep everything else:

```bash
accelerate launch --num_processes=1 run_benchmark_main.py \
    --tasks gsm8k --num_fewshot 5 --limit 100 --model test --batch_size 1 --device cuda:0 \
    --output_path results/dense_gsm8k \
    --model_args "id_model=GSAI-ML/LLaDA-8B-Base,size_batch=1,len_target=256,num_blocks=1,num_unmask_per_step=1,id_mask=126336,runner=run_llada_semi,path_report=results/dense_gsm8k__runner.json"
```

Keep `--limit` below the `limit_safe_max` recorded in the step 1 `.meta.json`. That keeps the eval docs disjoint from the oracle and training docs.

## File map

| file | role |
|---|---|
| `save_benchmark_mockup.py` | step 1: writes the lm-eval requests to a CSV |
| `run_collect_metrics_llada_base.py`, `collect_metrics_common.py` | step 2: oracle trace collection |
| `run_train_mlp.py`, `router_llada.py` | step 3: features, router, losses, trainer |
| `router_deploy.py` | saves and loads router bundles; builds features online at decode time |
| `run_benchmark_main.py` | step 4: lm-eval model wrapper (`--model test`) |
| `run_llada_semi_mlp_v2_batch.py`, `runner_mlp_common.py` | step 4: router-guided cached decoding |
| `run_llada_semi.py` | step 4: dense baseline |
| `modeling_llada_yukai_06.py`, `configuration_llada.py`, `plugins_llada.py` | LLaDA model with KV and attention cache plugins |
| `components_llada.py`, `tools_llada.py`, `configs_llada.py`, `dataprocess_llada.py`, `constants_llada.py` | shared utilities |
