# Efficient LLM Inference: INT4 Quantization & LoRA Mixture-of-Experts for Llama 3.2-1B

Post-training **INT4 weight-only quantization** and a **LoRA-based Mixture-of-Experts**, implemented from scratch in PyTorch for Llama 3.2-1B — with end-to-end benchmarks (memory, throughput, perplexity) and an analysis of why naive INT4 dequantization is a *bandwidth* win but not (yet) a *latency* win.

**Author: Krishna Prasad Deshpande** · [linkedin.com/in/krishna-prasadd](https://www.linkedin.com/in/krishna-prasadd/)

> Built on the EE 508 (USC, Hardware Foundations of ML) course scaffold, which is based on [Meta's Llama 3 reference implementation](https://github.com/meta-llama/llama3). **My implementation:** [`llama/quantize.py`](llama/quantize.py), [`llama/moe.py`](llama/moe.py), the training/evaluation runs, benchmarks, figures, and the full write-up in [`Efficient_LLM_Inference_Project.md`](Efficient_LLM_Inference_Project.md).

---

## What I built

### 1. INT4 weight-only quantization — [`llama/quantize.py`](llama/quantize.py)
- **Per-group asymmetric quantization**: each group of 128 weights shares an FP16 scale and zero-point.
- **Nibble packing**: two INT4 values packed per `uint8` buffer; dequantized to FP16 at matmul time.
- **`QuantizedLinear`** drop-in module + automated conversion of every `nn.Linear` in the model.
- **Result: model storage −60.6% (2,858 → 1,127 MB)** with coherent generation preserved.

### 2. Mixture-of-Experts FFN — [`llama/moe.py`](llama/moe.py)
Two variants behind one interface:
- **Slice MoE**: split the pretrained dense SwiGLU FFN into N smaller experts (hidden_dim/N each) with a learned top-k router.
- **LoRA MoE**: freeze the dense FFN and attach N lightweight LoRA adapters as experts — far more parameter-efficient; **zero-init so the converted model is logit-identical to the base model at step 0**.
- Fine-tuned on Alpaca instructions: **held-out perplexity 9.66 → 9.49**, with expert-load balance tracked across training ([figures](figures/)).

### 3. Benchmarks & analysis
Measured on an A40 (PyTorch 2.6, CUDA 12.4), end-to-end including prefill:

| input=256, output=32 | batch=1 | batch=8 | batch=16 |
|---|---|---|---|
| FP16 throughput (tok/s) | 39.7 | 388.9 | 686.7 |
| INT4 throughput (tok/s) | 16.6 | 120.4 | 221.2 |
| FP16 peak mem (MB) | 3,072 | 4,495 | 6,134 |
| INT4 peak mem (MB) | 3,282 | 4,251 | 5,364 |

**The honest headline: naive INT4 is 2–3× *slower* than FP16.** The dequantize → materialize-FP16 → matmul path adds work per step, and at batch=1 the materialized FP16 transient even erases the peak-memory win (+6.8%); INT4's memory advantage only emerges as activations/KV-cache dominate (−5.4% at batch=8, −12.6% at batch=16).

**Why it still matters:** roofline / arithmetic-intensity analysis shows autoregressive **decode is memory-bandwidth-bound** — weights are re-streamed from HBM every token. INT4 cuts bytes-per-token ~4× for weight traffic, so a **fused INT4×FP16 kernel** (dequantizing in registers/shared memory, never materializing FP16 weights) converts the storage win into a real decode-throughput win. The write-up covers this, plus GPTQ, AWQ, KV-cache quantization (~256 MB at 8K context), and grouped-GEMM / expert-parallel MoE dispatch.

📄 **Full write-up:** [`Efficient_LLM_Inference_Project.md`](Efficient_LLM_Inference_Project.md) — implementation details, correctness verification, benchmark methodology, and analysis.

---

## Repository map

```
llama/
├── quantize.py     ★ my implementation — INT4 QuantizedLinear + model conversion
├── moe.py          ★ my implementation — slice & LoRA MoE + top-k router
├── model.py          Llama 3.2 architecture (scaffold / Meta reference)
├── generation.py     generation loop (scaffold)
└── tokenizer.py      tokenizer (scaffold)
train_moe.py          Phase-3 fine-tuning driver
eval_moe.py           perplexity / MT-bench evaluation
run_benchmark.py      memory + throughput benchmark harness
figures/              training-loss & expert-load plots
*.log                 raw benchmark / training logs from my runs
```

## Running it

```bash
pip install -r requirements.txt
# Download weights (Meta license applies):
pip install llama-stack && llama model download --source meta --model-id Llama3.2-1B
# Update checkpoint_dir at the top of each script, then:
python inference.py            # smoke test
python run_benchmark.py        # FP16 vs INT4 memory/throughput
python train_moe.py --init lora && python eval_moe.py
```

> Note: scripts default to the original course-server paths; point `checkpoint_dir`/data paths at your local copies.

## Attribution & license

Model architecture and generation code derive from Meta's Llama 3 reference implementation (Meta Llama license applies to code and weights). Course scaffold © USC EE 508. All quantization and MoE implementation, experiments, benchmarks, figures, and analysis are my own work.
