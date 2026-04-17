# EE 508 Final Project

## Efficient Inference of Large Language Models

**Instructor:** Arash Saifhashemi

**TA:** [Zeyu Liu, Haoyan Xu]

---

## 1 Introduction

Large language models (LLMs) based on transformer architecture, such as OpenAI's GPT-4, Meta's Llama, and DeepSeek-V3, have greatly improved the ability to generate and understand text. While training these models is enormously expensive, the dominant cost for most organizations is now **inference** — serving the trained model to users at scale. As LLMs are deployed in production systems serving millions of requests per day, optimizing inference efficiency has become one of the most critical challenges in the industry.

In this project, we will use Meta AI's open-source **LLaMA 3.2-1B** model to explore efficient inference techniques. Specifically, we will investigate:

- **INT4 weight quantization**: reducing the precision of model weights to lower memory usage and accelerate memory-bound inference
- **Mixture-of-Experts (MoE)**: an architectural approach to reducing computation while maintaining model capacity

---

## 2 Phase 1: Background Knowledge (5%)

In this phase, you will review the shortened version of the LLM Foundations paper provided in the project repository. After reading through the paper, please answer the questions listed below. You are welcome to use external resources like Google or ChatGPT to help with your understanding, but you are responsible for ensuring the accuracy and completeness of your responses.

1. What is language modeling?
2. What is self-supervised pretraining?
3. Why is pretraining more hardware-efficient for Transformer- or attention-based models compared to RNN-based models?
4. What is the difference between encoder-only and decoder-only models, and why are decoder-only models more popular?
5. Suppose the vocabulary consists of only three words: Apple, Banana, and Cherry. During decoder-only pretraining, the model outputs the probability distribution Pr(· | x₀, ..., xᵢ) = (0.1, 0.7, 0.2). If the correct next word is Cherry, represented by the one-hot vector (0, 0, 1), what is the value of the log cross-entropy loss? What is the loss value if the correct next word is Banana instead?
6. What are zero-shot learning, few-shot learning, and in-context learning?
7. What is tokenization? What is a word embedding layer?
8. What is position embedding? What kind of position embedding method is used in Llama models?
9. What is the difference between multi-head attention (MHA) and grouped-query attention (GQA)? Which type of attention mechanism is used in Llama 3.2? Why is GQA preferred over MHA for inference?
10. What is layer normalization? What kind of normalization does Llama use, and how does it differ from standard layer normalization?
11. What is the auto-regressive generation process, and how is a decoding strategy used during text generation?
12. What is instruction fine-tuning, and how does it differ from pretraining in terms of data, objective, and computational cost? During instruction fine-tuning, why is the loss typically computed only on the output/response tokens rather than on the entire prompt+response sequence? Illustrate with a short example.
13. What does it mean to align an LLM with human intentions? Briefly explain the roles of supervised fine-tuning (SFT) and reinforcement learning from human feedback (RLHF).
14. What is a prompt template? What are *system information* and *demonstrations* in a prompt, and how can they change model behavior without updating model parameters?
15. What is chain-of-thought prompting? Compare zero-shot, one-shot, and few-shot chain-of-thought prompting, and explain why chain-of-thought can improve reasoning on complex tasks.

**Deliverable:** Modify this markdown file directly to add your answers to the questions above.

---

## 3 Phase 2: INT4 Weight Quantization (5%)

We first describe why quantization matters for LLM inference, then provide guidance for this phase.

### 3.1 Why Quantization Matters

LLM inference consists of two stages: **prefill** and **decode**. During the decode stage, the model generates one token at a time. For each token, every weight in the model must be read from GPU high-bandwidth memory (HBM) to the compute units, but each weight participates in only a single multiply-add operation (when batch size = 1). This makes the decode stage **memory-bandwidth bound**: the bottleneck is not computation but the speed at which weights can be read from memory.

This observation has a powerful implication: **if we can reduce the size of each weight, we can directly speed up inference.** Reducing weights from FP16 (16 bits per weight) to INT4 (4 bits per weight) cuts memory traffic by 4×, which in theory translates to a 4× speedup in the memory-bound decode phase.

### 3.2 INT4 Weight-Only Quantization

In weight-only quantization, we quantize the model weights to a lower precision for storage, but dequantize them back to FP16 before performing the actual matrix multiplication. This approach preserves the precision of activations and is the most commonly used quantization strategy for LLM inference.

The quantization process for a group of weights works as follows:

**Quantize (offline, once):**

Given a group of FP16 weights w₁, w₂, ..., wₘ (where m is the group size, typically 128):

1. Compute the range: w_min = min(wᵢ), w_max = max(wᵢ)
2. Compute scale and zero_point:
   - scale = (w_max - w_min) / (2⁴ - 1) = (w_max - w_min) / 15
   - zero_point = round(-w_min / scale)
3. Quantize each weight: wᵢ_int4 = clamp(round(wᵢ / scale) + zero_point, 0, 15)
4. Pack two INT4 values into one uint8: packed = (w_even) | (w_odd << 4)

**Dequantize (at inference time, every forward pass):**

1. Unpack uint8 to two INT4 values: w_even = packed & 0x0F, w_odd = (packed >> 4) & 0x0F
2. Dequantize: wᵢ_fp16 = (wᵢ_int4 - zero_point) × scale
3. Perform the matrix multiplication in FP16

**Why per-group?** Using a single scale/zero_point for an entire weight matrix (per-tensor quantization) leads to large quantization errors because outlier values force the range to be wide. Per-group quantization (e.g., group_size=128) allows each group of 128 weights to have its own scale/zero_point, significantly reducing quantization error at the cost of storing a few extra parameters per group.

### 3.3 Implementation Guidance

In this phase, you will implement INT4 weight-only quantization for the Llama 3.2-1B model.

1. **Understand the baseline:** Run `inference.py` with the original FP16 model and observe the outputs. Run `benchmark_inference.py` to measure baseline performance.

2. **Implement `QuantizedLinear` module:** Complete the skeleton in `llama/quantize.py`. Fill in the `QuantizedLinear` class so that it:
   - Takes a pre-trained `nn.Linear` layer and quantizes its weights to INT4 with per-group scale and zero_point
   - Packs two INT4 values into one uint8 for storage
   - At inference time, unpacks and dequantizes weights to FP16, then performs the matrix multiplication
   - This is analogous to last year's LoRA module replacement. You are replacing `nn.Linear` with your custom module, similar to how LoRA replaced Q/V projections.

3. **Convert the model:** Write a function `quantize_model(model, group_size=128)` that replaces all `nn.Linear` layers in the Llama model with your `QuantizedLinear`. Report:
   - Number of original FP16 parameters
   - Size of INT4 quantized weights (in MB)
   - Percentage reduction in model size

4. **Test correctness:** Verify your implementation before trusting any benchmark numbers.

   - **Automated checks.** Run `python check_student.py`. It verifies (1) buffer / scale / zero-point shapes, (2) bounded `quantize → dequantize` round-trip error, (3) ≈60 % model-size reduction after `quantize_model`, and (4) coherent (non-gibberish) generation from the quantized model. Any FAIL message tells you which invariant your implementation breaks.
   - **Qualitative check.** Run inference with the quantized model using the same prompts as the FP16 baseline and compare outputs side-by-side. The quantized outputs will differ from FP16 (expected), but they should still be coherent and reasonable.

### 3.4 Benchmark Results and Analysis

After your implementation passes correctness checks, benchmark and analyze it.

1. **Benchmark:** Fill in the table below comparing FP16 vs INT4:

| input_len=256, output_len=32 | batch_size=1 | batch_size=8 | batch_size=16 |
|---|---|---|---|
| **FP16** Peak Mem (MB) | | | |
| **FP16** Runtime (s) | | | |
| **INT4** Peak Mem (MB) | | | |
| **INT4** Runtime (s) | | | |

2. **Required analysis:** In your write-up, answer the following questions:

   - In your naive implementation, the INT4 weights are stored compactly but dequantized to FP16 before each matmul. Why can this materialized FP16 intermediate negate part of INT4's peak-memory benefit, especially at small batch sizes?
   - Why can this same design also reduce or erase the expected runtime speedup, even though INT4 uses 4× fewer bits for weight storage? Tie your answer to the extra unpacking/dequantization work and total memory traffic during decode.
   - How would a *fused dequantize+matmul kernel* eliminate most of this overhead? Explain the difference in both peak memory and memory traffic.
   - Under what workload conditions would INT4 be more likely to outperform FP16?

3. **Required conceptual extensions (implementation optional):** Briefly discuss the following ideas, even if you do not implement them:

   - **GPTQ:** Unlike round-to-nearest quantization, GPTQ uses a small calibration set of sample activations to iteratively adjust the quantized weights so that activation reconstruction error is minimized. Why does using calibration data reduce quantization error at the same bit-width?
   - **KV-cache quantization:** For Llama 3.2-1B (see `llama/model.py` for its config), at batch=1 and seq_len=8192, roughly estimate the FP16 KV-cache memory footprint. Why is quantizing the KV cache a complementary optimization to weight quantization rather than a replacement?
   - **AWQ and GGUF K-quants:** AWQ scales salient weight channels before quantization, and GGUF K-quants mix multiple bit-widths within a layer based on per-block importance. Why can each preserve model quality better than uniform round-to-nearest INT4 at the same average bit-width?
   - **Optional implementation:** If you want, use the `auto-gptq` library to produce an INT4-quantized model using calibration-based optimization (GPTQ algorithm), and compare the output quality of your naive round-to-nearest INT4 against GPTQ's INT4 at the same bit-width.

**Interpretation note.** A naive weight-only INT4 implementation at the 1B-parameter scale may show *higher* runtime than FP16, and sometimes *higher* peak memory at small batch sizes. Treat this as a result to explain from your `QuantizedLinear.forward()` design, not as automatic evidence of a bug.

### 3.5 Deliverable

Modify this markdown file directly to include: a summary of your `QuantizedLinear` implementation, the model size comparison, sample outputs from the quantized model, the filled-in benchmark table above, and concise written answers to the Phase 2 analysis questions (including fused-kernel reasoning and the GPTQ / KV-cache / AWQ / GGUF discussion). Embed any figures as Markdown images (e.g. `![](figures/phase2_xxx.png)`) with the image files committed to the repo.

---

## 4 Phase 3: Mixture-of-Experts (MoE) (10%)

### 4.1 Background

Mixture-of-Experts (MoE) is an architecture that replaces each dense feedforward network (FFN) with multiple smaller "expert" networks and a learned router. For each input token, only the top-K experts (typically K=2) are activated, while the rest are skipped. This means the model can have many more total parameters (capacity) without proportionally increasing the computation per token.

MoE has become the dominant architecture for frontier LLMs. DeepSeek-V3 (671B total, 37B active), Mixtral 8×7B, and Qwen3-MoE all use this approach. The key insight is that different tokens benefit from different "specialists" — a token about mathematics might activate different experts than a token about poetry.

The MoE feedforward layer works as follows:

Given input x ∈ ℝʰ and N experts E₁, E₂, ..., Eₙ, each being a smaller FFN:

1. **Router:** Compute gating scores g = softmax(W_router · x), where W_router ∈ ℝᴺˣʰ
2. **Top-K selection:** Select the K experts with the highest gating scores
3. **Expert computation:** Compute output from selected experts
4. **Weighted sum:** y = Σᵢ∈top-K gᵢ · Eᵢ(x)

A critical challenge in MoE is **load balancing** — if the router always sends tokens to the same few experts, the other experts never learn and the model degenerates. Production MoE systems use auxiliary loss terms to encourage balanced routing.

### 4.2 Two Initialization Modes

You will implement two ways to create MoE experts from a pretrained dense model:

**Slice mode:** Split the pretrained FFN into N non-overlapping slices. Each expert gets hidden_dim / N of the original weights. This preserves total parameter count but reduces each expert's capacity.

**LoRA mode:** Keep the original dense FFN **frozen** and attach N lightweight LoRA adapters as "experts". Each LoRA expert is a low-rank pair (A, B) where A ∈ ℝʳˣᵈ and B ∈ ℝᵈˣʳ with r << d. The router selects which LoRA adapters to activate per token:

    output = frozen_FFN(x) + Σᵢ∈top-K  gᵢ · LoRA_i(x)

By initializing B to zero, the converted model produces **identical outputs** to the original dense model before any training. This is far more memory-efficient than copying the full FFN for each expert.

### 4.3 Implementation Guidance

Your task is to convert the dense Llama 3.2-1B model into a sparse MoE model by replacing the FeedForward layers. You will implement **both** initialization modes.

1. **Implement slice mode:** Complete the `MoEFeedForward` skeleton in `llama/moe.py` so that it:
   - Contains N smaller expert FFNs (each with hidden_dim / N)
   - Has a linear router: `nn.Linear(dim, num_experts)`
   - Implements top-K gating (K=2)
   - Returns the weighted sum of the top-K expert outputs

2. **Implement LoRA mode:** In the same file, implement the `LoRAMoEFeedForward` variant where:
   - The original dense FFN is stored as `self.base_ff` and frozen
   - Each expert is a `LoRAExpert` — two linear layers `A (d→r)` and `B (r→d)` with `B` initialized to zero
   - The router selects which LoRA adapters to activate per token
   - LoRA and router parameters are kept in float32 for training stability (cast inputs to float32 inside the adapter and cast outputs back)
   - Output: `base_ff(x) + Σ_k gₖ · LoRA_k(x)`

3. **Convert the model:** Write a single `convert_to_moe` function that supports both `init_mode="slice"` and `init_mode="lora"`. For slice mode, initialize expert weights by slicing the original FFN. For LoRA mode, wrap the original FFN with LoRA adapters.

4. **Test correctness:** Before spending GPU time on training or benchmarking, verify both MoE implementations. Run `python check_student.py`. It verifies (1) output-shape preservation, (2) top-K routing weights sum to 1, (3) exact parameter counts for both slice and LoRA modes, (4) `LoRAExpert` and `LoRAMoEFeedForward` zero-init identity (an untrained LoRA-MoE must produce identical logits to the dense model), (5) frozen base FFN and trainable LoRA adapters, and (6) balanced random-init routing. Any FAIL message tells you which invariant your implementation breaks.

### 4.4 Evaluation, Analysis, and Discussion

After your implementation passes correctness checks, train, evaluate, and compare the two modes.

1. **Fine-tune and evaluate slice mode:** Run `python train_moe.py --init-mode slice` and evaluate with `python eval_moe.py --init-mode slice`.

   **Interpretation note.** Slice mode is trained from a *cold start*: each expert only sees hidden_dim / N of the original FFN's capacity, and the randomly initialized router has not yet learned useful routing. With only a few hundred Alpaca samples and 2–3 epochs, you will likely see perplexity increase relative to the dense baseline and generations get worse. Report this result honestly; the limitation here is the small fine-tuning budget, not necessarily your implementation.

2. **Fine-tune and evaluate LoRA mode:** Run `python train_moe.py --init-mode lora` and evaluate with `python eval_moe.py --init-mode lora`.
   - **Optimizer:** SGD or AdamW
   - **Learning rate:** 1e-3 for router, 1e-4 for LoRA/experts (script defaults)
   - **Epochs:** 2–3 (should take ~15 minutes on A40)
   - Goal: observe that training loss decreases **and** that perplexity on the held-out set stays close to the dense baseline.

3. **Benchmark and analyze:** Fill in the following table and provide analysis:

| Metric | Dense (original) | MoE-slice (N=4, K=2) | MoE-LoRA (N=4, K=2, r=8) |
|---|---|---|---|
| Total parameters | | | |
| Trainable parameters | | | |
| Peak memory (MB) | | | |
| Inference time (s), batch=1, in=256, out=32 | | | |
| Perplexity (held-out set) | | | |
| Next-token accuracy (held-out set) | | | |

   Briefly discuss:

   - What intuition motivates **slice mode**? When you partition a pretrained dense FFN into `N` non-overlapping chunks and treat them as experts, what computational or representational structure are you hoping to exploit?
   - Why is **LoRA mode** motivated as a different approach? Explain why freezing the dense FFN and adding low-rank adapters preserves the dense model's behavior at initialization and is more suitable for small-data fine-tuning.
   - Why does LoRA mode preserve dense-level perplexity while slice mode degrades it?
   - What's the trade-off in trainable parameter count, and how does it affect fine-tuning stability on small data?
   - How does peak memory differ, and why?
   - Where does the extra wall-clock time come from in your `MoEFeedForward.forward()` implementation, and why do production MoE systems not pay this same overhead? Briefly describe what a better implementation would look like.

**Interpretation note.** A straightforward Python-level MoE forward pass may be several times slower than the dense baseline on the same prompt. Treat that as something to explain from the implementation structure, not as automatic evidence of a bug.

4. **Visualize expert load balance:** For a set of test prompts, record which experts are activated for each token. Plot the distribution of expert activation frequencies for both modes. Are all experts used roughly equally, or is there significant imbalance?

### 4.5 Deliverable

Modify this markdown file directly to include: a summary of both MoE implementations, training loss curves for slice and LoRA modes, the filled-in comparison table above, the expert load balance visualization, sample generations from each model, and a short comparison of the slice-vs-LoRA design intuition discussed above. Embed any figures as Markdown images (e.g. `![](figures/phase3_xxx.png)`) with the image files committed to the repo.

---

## 5 Summary

| Phase | Topic | Weight | GPU Time |
|---|---|---|---|
| Phase 1 | Background Knowledge | 5% | None |
| Phase 2 | INT4 Weight Quantization | 5% | ~30 min (benchmark only) |
| Phase 3 | Mixture-of-Experts (MoE) | 10% | ~30 min training + ~30 min benchmark |

**Total GPU time per group: ~1–2 hours (reference only, P100 may need longer time).**

**Hardware:** A40 (48 GB). Llama 3.2-1B in FP16 is ~2.5 GB; INT4 quantized is ~625 MB. MoE fits comfortably.

**Please check deadlines and deliverables in the course schedule.**
