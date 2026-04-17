[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/ulCFyiAb)
# Final Project - EE 508: Hardware Foundations of Machine Learning, Spring 2026

## University of Southern California

## Instructor: Arash Saifhashemi

A minimal Llama 3.2-1B codebase for exploring efficient LLM inference. Based on the [official Llama 3 implementation](https://github.com/meta-llama/llama3) from Meta.

## Project Overview

This year's project focuses on **efficient LLM inference**:

- **Phase 1:** Background knowledge questions — 5%
- **Phase 2:** INT4 weight quantization — 5%
- **Phase 3:** Mixture-of-Experts (slice mode + LoRA mode) — 10%

**👉 Read [`Efficient_LLM_Inference_Project.md`](Efficient_LLM_Inference_Project.md) first.** It is the authoritative project specification — it tells you what to implement, which tables to fill in, and where to write up your answers directly inside that markdown file. This README only covers how to set up and run the code.

`LLM_Foundations.pdf` is the background reading referenced by Phase 1.

## Repository Structure

```bash
├── Efficient_LLM_Inference_Project.md   # Project specification (READ THIS)
├── LLM_Foundations.pdf                   # Phase 1 background reading
├── README.md                             # Setup & how to run (this file)
├── requirements.txt                      # Python dependencies
├── inference.py                          # Basic inference / smoke test
├── benchmark_inference.py                # Single-config FP16 timing baseline
├── run_benchmark.py                      # Full Phase 2 + Phase 3 benchmark report
├── train_moe.py                          # Phase 3 fine-tuning driver
├── eval_moe.py                           # Phase 3 quantitative evaluation
├── prepare_data.py                       # (Optional) regenerate Alpaca / MT-bench data
├── llama/
│   ├── model.py                          # Llama architecture
│   ├── generation.py                     # Text generation loop
│   ├── tokenizer.py                      # Tokenizer
│   ├── quantize.py                       # [Phase 2] you implement this
│   └── moe.py                            # [Phase 3] you implement this
```

## Prerequisites

- The default scripts assume you are running on the class server with a CUDA-capable GPU.
- The intended environment is the class A40 setup referenced in the project specification.
- The default checkpoint and Phase 3 data paths are hardcoded to `/project2/saifhash_1190/...`.
- If you run on your own machine instead, you will need to download the model, regenerate or copy the data, and update the paths in the scripts.

## Setup

1. **Install packages** (PyTorch 2.6 + CUDA 12.4):

    ```bash
    pip install -r requirements.txt
    ```

2. **Model weights** — already on the class server at
    `/project2/saifhash_1190/llama/checkpoints/Llama3.2-1B/`.
    All scripts hardcode this path; no download needed.

    *Optional (running on your own machine):*

    ```bash
    pip install llama-stack
    llama model download --source meta --model-id Llama3.2-1B
    ```

    Then update `checkpoint_dir` at the top of each script.

3. **Phase 3 data** — already on the class server at
    `/project2/saifhash_1190/data/{alpaca_500.json, mt_bench_prompts.json}`.

    *Optional (regenerate elsewhere):*

    ```bash
    pip install datasets
    python prepare_data.py --output-dir ./data
    ```

    Then update the paths at the top of `train_moe.py` / `eval_moe.py`.

4. **Smoke test:** `python inference.py`

## What You Modify

For this project, you should only need to implement the student TODOs in:

- `llama/quantize.py` for Phase 2
- `llama/moe.py` for Phase 3

You do not need to rewrite the rest of the codebase. Use `inference.py`,
`train_moe.py`, `eval_moe.py`, and `run_benchmark.py` as drivers to test your
implementation.

## How to Run Each Phase

Refer to the project specification for the full task description; the commands below are just the quick-reference.

### Phase 2 — after completing `llama/quantize.py`

```bash
python inference.py       # qualitative: is the INT4 model coherent?
python run_benchmark.py   # quantitative: fills Phase 2 benchmark table
```

### Phase 3 — after completing `llama/moe.py`

```bash
# Slice mode first (expected to regress on small fine-tuning budgets — report honestly)
python train_moe.py --init-mode slice
python eval_moe.py  --init-mode slice

# LoRA mode second (should preserve dense-level quality with very few trainable params)
python train_moe.py --init-mode lora
python eval_moe.py  --init-mode lora

# Produces the Phase 3 comparison table (Dense / slice / LoRA)
python run_benchmark.py
```

Note: `slice` mode may produce worse generations and higher perplexity than the
dense baseline under our small fine-tuning budget. That is expected for this
assignment and is discussed in the project specification.

## Test & Benchmark Scripts

| Script                   | Purpose                                                                                                                                        |
|--------------------------|------------------------------------------------------------------------------------------------------------------------------------------------|
| `inference.py`           | Sanity-check: model loads and generates coherent text. Run at every stage.                                                                     |
| `benchmark_inference.py` | Single-config FP16 timing baseline (batch=16, in=256, out=64).                                                                                 |
| `run_benchmark.py`       | Main benchmarking driver — produces the numbers for the Phase 2 and Phase 3 tables in the project specification. You do not need to modify it. |
| `train_moe.py`           | Phase 3 fine-tuning; saves `checkpoints/moe_finetuned.pt`.                                                                                     |
| `eval_moe.py`            | Phase 3 quantitative eval (perplexity, next-token accuracy, tok/s); run after `train_moe.py`.                                                  |

## Expected Outputs

- `inference.py` prints sample generations for a quick smoke test.
- `run_benchmark.py` prints the benchmark numbers you will copy into the Phase 2 and Phase 3 tables in your report.
- `train_moe.py` writes `checkpoints/moe_finetuned.pt`, which `eval_moe.py` loads for Phase 3 evaluation.
- All written deliverables go directly into `Efficient_LLM_Inference_Project.md` (answers, filled-in tables, writeups, embedded figures). You submit by committing that file — no separate PDFs.

## Submission

This assignment has two phases with separate deadlines:

- **Phase 1 & 2 due:** May 1, 2026 (end of day)
- **Final submission (Phase 3) due:** May 5, 2026 (end of day)

Phase 1 & 2 is identified by a **Git tag**, not a commit message. We grade the commit that the `phase1-2` tag points to — `git commit -m "xxxxxx"` is a commit message, not a tag, and will not be counted.

### Phase 1 & 2 (due May 1)

1. Commit and push your Phase 1 and Phase 2 work:

    ```bash
    git add .
    git commit -m "Phase 1 & 2 submission"
    git push
    ```

2. Create the `phase1-2` tag and push it to GitHub:

    ```bash
    git tag phase1-2
    git push origin phase1-2
    ```

3. Verify the tag is present (should list `phase1-2`):

    ```bash
    git tag
    ```

### Final Submission — Phase 3 (due May 5)

Continue in the same repository and push your Phase 3 commits before the deadline:

```bash
git add .
git commit -m "Final submission"
git push
```

No additional tag is required unless announced otherwise.

### Notes

- Do not delete or move the `phase1-2` tag after May 1 — its target commit determines your Phase 1 & 2 grade.
- Grading uses GitHub commit timestamps and tag history to verify deadlines; late or modified tags may incur penalties.
- If you tagged the wrong commit, contact the instructor **before the deadline**. Do not silently delete and re-create the tag after May 1.

## Optional Extensions

The starter code keeps a `kv_caching` flag on both the model and `generate()`.
The KV-cache-disabled code path in `llama/generation.py` is left as an
optional exercise — implementing it lets you compare cached (O(n) per step)
vs uncached (O(n²) per step) decoding, and opens the door to a bonus
INT8/INT4 quantization of the KV cache itself. See the comments in
`generation.py` for pointers.
