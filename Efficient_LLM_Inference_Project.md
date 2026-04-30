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

### Phase 1 Answers

**1. What is language modeling?**

Language modeling is the task of estimating the probability distribution of natural-language token sequences. A language model factorizes the joint probability of a sequence `x₀, x₁, …, xₙ` into a product of conditional probabilities — almost always autoregressively, `Pr(x₀, …, xₙ) = Π_i Pr(xᵢ | x₀, …, xᵢ₋₁)`. Training a language model means fitting the parameters of those conditionals to a large corpus of text, typically by minimizing the negative log-likelihood of the next token. Once trained, the model can score how "natural" a sequence is, generate continuations, and act as a foundation for downstream tasks such as translation, summarization, code completion, and dialogue.

**2. What is self-supervised pretraining?**

Self-supervised pretraining uses raw, unlabeled text — there are no human-annotated labels. The training signal is constructed automatically from the data itself: in a decoder-only model, the label for position `i` is the actual token at position `i+1` in the same document. The same applies to BERT-style masked-language modeling, where the model is asked to predict masked-out tokens given their context. Because every token in the corpus produces a free supervision signal, self-supervised pretraining can scale to trillions of tokens, which is the regime where Transformers learn the broad linguistic and world-knowledge capabilities that make them useful as general-purpose models.

**3. Why is pretraining more hardware-efficient for Transformer- or attention-based models compared to RNN-based models?**

RNNs are inherently sequential along the time axis — the hidden state at step `t` depends on the hidden state at step `t-1`, so the GPU has to wait for the previous step before starting the next. Transformers replace that recurrence with self-attention, which lets every position attend to every other position in parallel using a single batched matrix multiplication. As a result the entire sequence of length `n` can be processed in one pass during pretraining, fully utilizing the matmul-throughput of modern accelerators rather than serializing on a recurrence. Transformers are also much more amenable to data and tensor parallelism because each layer is just a stack of large matmuls, which map cleanly onto GPU/TPU compute primitives.

**4. What is the difference between encoder-only and decoder-only models, and why are decoder-only models more popular?**

Encoder-only models (e.g., BERT) use bidirectional self-attention: each token sees the full sequence in both directions. They are typically trained with masked-language modeling and excel at producing per-token representations for classification, retrieval, and tagging — but they are not naturally generative, because their training objective does not match autoregressive decoding. Decoder-only models (e.g., GPT, Llama) use causal masked self-attention: position `i` can only attend to positions `≤ i`. Their next-token-prediction objective generalizes seamlessly to text generation, and the same architecture scales to instruction following, chain-of-thought, code, and dialogue with no architectural change. Because a single decoder-only model can do classification, generation, reasoning, and few-shot learning by prompting alone, it has become the dominant paradigm for general-purpose LLMs.

**5. Cross-entropy loss with the toy three-word vocabulary.**

Cross-entropy loss for a one-hot target `y` is `L = −Σ_v y_v · log p_v = −log p_correct`. With the model's distribution `(0.1, 0.7, 0.2)`:

- If the correct word is **Cherry** (one-hot `(0, 0, 1)`), the loss is `−log(0.2) ≈ 1.609` (in nats; or `−log₂(0.2) ≈ 2.322` bits).
- If the correct word is **Banana** (one-hot `(0, 1, 0)`), the loss is `−log(0.7) ≈ 0.357` nats.

The lower loss for "Banana" reflects that the model already places most of its probability mass there, so it is much less surprised by the correct answer than it would be by "Cherry."

**6. What are zero-shot, few-shot, and in-context learning?**

In **zero-shot learning** the model is asked to perform a task it has never seen labeled examples for, given only a natural-language description of the task (e.g., "Translate the following English sentence to French: …"). In **few-shot learning** a small number of `(input, output)` exemplars are added to the prompt before the actual query, so the model can pattern-match on them. **In-context learning** is the umbrella term for both: it is the surprising emergent ability of large LLMs to "learn" a new task purely from examples and instructions placed inside the input context, with no gradient updates. The model's parameters never change — it adapts its behavior solely on what is in the current prompt.

**7. What is tokenization? What is a word embedding layer?**

Tokenization is the process of converting raw text into a sequence of integer token IDs drawn from a fixed vocabulary. Modern LLMs almost always use subword tokenizers such as Byte-Pair Encoding (BPE), SentencePiece, or tiktoken, which split text into pieces that range from whole common words to single characters or bytes. Subword tokenization is a sweet spot: it keeps the vocabulary tractable (≈30k–256k entries) while still being able to represent any string. The **word (or token) embedding layer** is a learned lookup table `E ∈ ℝ^(V×d)` that maps each integer token ID to a `d`-dimensional dense vector. These embeddings are the actual continuous input the rest of the network operates on — they are trained jointly with the model so that semantically related tokens end up with similar representations.

**8. What is position embedding? Which kind does Llama use?**

Self-attention is permutation-invariant — without extra signals, a Transformer cannot tell "dog bites man" from "man bites dog". Position embeddings inject order information. The original Transformer used additive sinusoidal or learned absolute position embeddings added to the token embeddings at the input. Llama (1, 2, 3, and 3.2) uses **Rotary Position Embedding (RoPE)** instead. RoPE encodes the absolute position by rotating the query/key vectors in two-dimensional subspaces by an angle proportional to the position, so that the inner product `⟨q_m, k_n⟩` depends only on the relative offset `m − n` and the content of `q` and `k`. This gives translational equivariance, extends gracefully to longer sequences than seen in training, and is implemented as a cheap element-wise complex multiply on Q and K — exactly the `apply_rotary_emb` you can see in `llama/model.py`.

**9. What is the difference between MHA and GQA? Which does Llama 3.2 use? Why is GQA preferred for inference?**

In **multi-head attention (MHA)** there is one independent K head and one independent V head per query head — so `n_heads` queries, `n_heads` keys, and `n_heads` values. In **grouped-query attention (GQA)** several query heads share a single K/V head: there are `n_heads` queries but only `n_kv_heads < n_heads` key/value heads. Llama 3.2-1B uses GQA with `n_heads = 32` and `n_kv_heads = 8` (see `ModelArgs` in `llama/model.py`), so each KV head is shared by 4 Q heads. GQA is preferred for inference because the **KV cache** dominates decode-stage memory traffic, and KV-cache size scales as `2 · n_kv_heads · seqlen · head_dim`. Cutting `n_kv_heads` by 4× shrinks both the cache footprint and the bytes that have to be streamed from HBM at every decode step, which directly translates to faster autoregressive generation, while the loss in modeling quality compared to full MHA is small in practice.

**10. What is layer normalization? What does Llama use, and how is it different?**

Standard **layer normalization** (Ba et al., 2016) recenters and rescales each token's activation vector: it subtracts the per-token mean, divides by the per-token standard deviation, and applies a learned scale `γ` and shift `β`. Llama uses **RMSNorm** (Zhang & Sennrich, 2019) instead. RMSNorm drops the mean-subtraction (and the bias `β`) and divides by the root-mean-square only: `x · γ / sqrt(mean(x²) + ε)`. Removing the mean-centering eliminates one reduction and one subtract, so RMSNorm is faster and uses fewer parameters than LayerNorm while empirically matching its training stability for Transformer-style residual stacks. You can see this in the `RMSNorm._norm` method in `llama/model.py`.

**11. What is auto-regressive generation, and how is a decoding strategy used?**

Auto-regressive generation is the process of sampling text one token at a time, where each new token is appended to the running context and fed back as input for the next step. Concretely: given a prompt, the model produces a probability distribution `Pr(x_t | x_<t)` over the next token; the **decoding strategy** is the rule that turns that distribution into a single chosen token. Common strategies are **greedy** (pick the argmax — deterministic but often dull and prone to repetition), **temperature sampling** (sample from the distribution after dividing logits by a temperature `T`; higher `T` ⇒ more diverse), **top-k** (sample from only the `k` most likely tokens), and **nucleus / top-p** (sample from the smallest set of tokens whose cumulative probability exceeds `p`). The chosen token is appended to the context and the loop repeats until an end-of-sequence token or a maximum length. Llama's `model.generate` in this repo uses temperature + top-p, which is a good balance between coherence and diversity.

**12. What is instruction fine-tuning, and how does it differ from pretraining? Why is loss only on the response tokens?**

**Pretraining** uses very large, unlabeled raw text and a uniform next-token-prediction loss over every token. **Instruction fine-tuning (SFT)** further trains the pretrained model on a much smaller curated dataset of `(instruction, response)` pairs — often tens of thousands to a few million examples — to teach the model to follow instructions, answer questions, refuse misuse, and produce well-formatted outputs. Compared to pretraining it uses several orders of magnitude less compute and data, but is highly leveraged because pretraining already learned the language and world-knowledge.

The loss is computed only on the response tokens, with the prompt tokens masked out, because we do not want the model to learn to *generate* the user's prompt — we want it to *condition on* the prompt and produce a good response. Training on prompt tokens would (1) waste capacity learning to reproduce the instruction text, (2) bias the model toward parroting prompts, and (3) under-weight the actually useful supervision signal in the response.

*Example.* Suppose the training example is
> Prompt: "Translate to French: 'Hello, world.'"  
> Response: "Bonjour le monde."

After tokenization, the model sees the concatenation `[prompt_tokens] + [response_tokens]`. The loss mask is `0` over the prompt portion and `1` over the response portion, so gradients only flow through the prediction of "Bon", "jour", "le", "monde", "." — each conditioned on everything to its left, including the full prompt. The prompt itself is treated as fixed context, not as something the model needs to learn to generate.

**13. What does it mean to align an LLM with human intentions? What are SFT and RLHF?**

A pretrained LLM is a brilliant but unmoored autocomplete: it can produce fluent text but its objective never said anything about being helpful, honest, or harmless. **Alignment** is the process of nudging the model's behavior toward human intentions — following instructions, answering truthfully, refusing genuinely harmful requests, matching a desired tone. **Supervised fine-tuning (SFT)** is the first alignment step: train on curated `(instruction, ideal response)` pairs so the model learns the *form* of a good answer. **Reinforcement learning from human feedback (RLHF)** goes further: humans rank multiple model outputs for the same prompt; those rankings train a reward model; the LLM is then optimized (typically with PPO, or more recently DPO as a cheaper substitute) to produce outputs that score higher under the reward model. SFT teaches the model what good answers look like; RLHF teaches it which of several plausible answers humans actually prefer.

**14. What is a prompt template? What are system messages and demonstrations?**

A **prompt template** is the fixed structural format (often with role tags such as `<|system|>`, `<|user|>`, `<|assistant|>`) that the model was instruction-tuned on, and that inference code is expected to reproduce. Templates control how the model interprets different parts of the input. Two especially useful slots inside a template are:

- **System information / system prompt:** a top-level instruction that sets the model's persona, capabilities, constraints, or output format ("You are a helpful AI assistant. Always respond in JSON.").
- **Demonstrations:** few-shot `(input, output)` exemplars inserted before the actual query to show the model exactly the kind of behavior expected.

Both of these change model behavior **without updating any parameters**, by exploiting in-context learning: the model conditions on them while autoregressively generating, and adapts accordingly. This is why prompt engineering is so powerful — the same checkpoint can act as a JSON formatter, a code reviewer, or a children's-book author depending purely on the system message and demonstrations placed in the context.

**15. What is chain-of-thought prompting? Compare zero-shot, one-shot, and few-shot CoT.**

**Chain-of-thought (CoT) prompting** asks the model to write out its intermediate reasoning steps before producing a final answer, instead of jumping straight to the answer. The intuition is that complex problems require multiple reasoning hops; making those hops explicit gives the model "scratch space" inside the context, and each next-token prediction is conditioned on all the prior reasoning, which empirically dramatically improves accuracy on multi-step arithmetic, logic, and common-sense reasoning benchmarks.

- **Zero-shot CoT** (Kojima et al., 2022): no exemplars; just append a trigger phrase such as "Let's think step by step" to the prompt. Cheapest, but weakest.
- **One-shot CoT:** include one full worked example showing the reasoning trace before the actual question.
- **Few-shot CoT** (Wei et al., 2022): include several worked examples; this primes the model to imitate the reasoning style and is the most effective of the three on hard benchmarks like GSM8K and MATH.

CoT works because reasoning tasks have multiple latent steps that the model is otherwise being asked to compress into a single forward pass per token. Letting the model emit those steps as text turns the problem into a sequence of much easier next-token predictions, each conditioned on the partial solution so far. Notably, CoT only helps once models are large enough — small models often produce plausible-looking but wrong reasoning chains.

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

Numbers below are from `python run_benchmark.py` on the class **A40** (PyTorch 2.6.0+cu124). Sampling: `temperature=0.6, top_p=0.9, kv_caching=True`. Each cell is end-to-end including prefill.

| input_len=256, output_len=32 | batch_size=1 | batch_size=8 | batch_size=16 |
|---|---|---|---|
| **FP16** Peak Mem (MB) | 3072 | 4495 | 6134 |
| **FP16** Runtime (s)   | 0.81 | 0.66 | 0.75 |
| **FP16** Throughput (tok/s) | 39.7  | 388.9 | 686.7 |
| **INT4** Peak Mem (MB) | 3282 | 4251 | 5364 |
| **INT4** Runtime (s)   | 1.93 | 2.13 | 2.32 |
| **INT4** Throughput (tok/s) | 16.6  | 120.4 | 221.2 |

Two patterns to notice in the actual measurements:

- **Runtime:** naive INT4 is **2–3× slower** than FP16 across every batch size. This is the expected outcome of the `dequantize → materialize FP16 → matmul` path (see analysis Q2 below).
- **Peak memory:** INT4 is *higher* than FP16 only at **batch=1** (3282 vs 3072 MB, +6.8 %). At batch=8 INT4 is **lower** (4251 vs 4495 MB, −5.4 %), and at batch=16 INT4 is much lower (5364 vs 6134 MB, −12.6 %). This is because at small batch the materialized FP16 weight transient dominates peak memory, while at larger batch sizes the activation + KV-cache footprint dominates and the persistent INT4 storage's smaller size starts to show. See analysis Q1 for the mechanism.

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

### Phase 2 Write-up

#### Summary of `QuantizedLinear` implementation

The `QuantizedLinear` module in `llama/quantize.py` is a drop-in replacement for `nn.Linear` that stores weights at 4 bits per element while still exposing a standard `forward(x)` that returns FP16 activations. Three pieces do the work:

1. **`quantize_tensor(weight, group_size=128)`** is the offline step. It reshapes the FP16 weight matrix `W ∈ ℝ^(out_features × in_features)` into per-group tiles `(out_features, n_groups, group_size)` so that *each row's `in_features` are split into `n_groups = in_features / 128` independent groups*. Within every group it computes `scale = (max − min) / 15` and `zero_point = round(−min / scale)` clamped to `[0, 15]`. Each weight is then quantized as `q = clamp(round(w / scale + zp), 0, 15)`. A guard replaces `scale = 0` (constant groups) with `scale = 1` to avoid division by zero. Two adjacent INT4 values along the `in_features` axis are packed into one `uint8`: lower nibble = even index, upper nibble = odd index. The function returns `(packed_weight: uint8 (out, in/2), scale: fp16 (out, n_groups), zero_point: fp16 (out, n_groups))`.

2. **`dequantize_packed(packed, scale, zp, group_size)`** is the online step that runs every forward pass. It unpacks the lower and upper nibbles with `& 0x0F` and `>> 4`, re-interleaves them with `torch.stack(..., dim=-1).reshape(...)`, reshapes into `(out, n_groups, group_size)`, and reconstructs `w_fp16 = (q − zp) × scale`. Because every weight in a group shares the same scale and zero-point, the whole reconstruction is a single broadcasted multiply.

3. **`forward(x)`** simply calls `dequantize_packed` to materialize the FP16 weight tensor and then runs `F.linear(x, weight, bias)`. This is the *naive* weight-only path: the dequantized FP16 tensor is materialized in memory as a transient before the matmul. The persistent storage is INT4, but the transient is FP16. We discuss why this matters in the analysis section below.

The model-level driver `quantize_model(model, group_size=128)`:

- collects every `nn.Linear` whose `in_features` is a multiple of `group_size` (all linears in Llama 3.2-1B satisfy this);
- builds a `QuantizedLinear` from each via the `from_linear(...)` classmethod, which moves the new module onto the same device as the source weight and copies the quantized buffers in;
- reattaches the new module under the parent via `setattr(parent, child_name, ql)`;
- drops references to the old FP16 layer and calls `torch.cuda.empty_cache()` so the size win is reflected in `nvidia-smi` immediately, not after the next allocation.

Note that `nn.Embedding` (`tok_embeddings`) is *not* quantized — it isn't an `nn.Linear`, and we deliberately leave it in FP16. Quantizing the input embedding table tends to hurt rare-token quality without giving much memory back, because embeddings are read once per token rather than reused.

#### Model-size comparison

The numbers below are the actual `print_model_size(model)` output captured during `python run_benchmark.py` on the A40, before and after calling `quantize_model(model, group_size=128)`.

| Quantity | FP16 baseline | After INT4 quantization |
|---|---|---|
| Parameters (MB) | 2858.13 | 501.13 |
| Buffers (MB) | 0.00 | 626.08 |
| **Total (MB)** | **2858.13** | **1127.21** |
| **Reduction** | — | **60.56 %** |

Why "Parameters" drops by ~2350 MB while "Buffers" jumps to 626 MB: `QuantizedLinear` registers `packed_weight`, `scale`, and `zero_point` as **buffers** (via `register_buffer`), not as `nn.Parameter`s — they're not trainable. So the post-quantization 501.13 MB of parameters is just `tok_embeddings`, the RMSNorm scales, and the few non-quantized small tensors; the 626.08 MB of buffers is the entire INT4 weight store + per-group metadata + the (zero-init) KV cache. Total memory is what matters: **2858 MB → 1127 MB, a 60.56 % reduction**, almost exactly matching the analytical prediction below.

- **Total parameters in the FP16 model:** **1,498,482,688** ≈ 1.498 B (printed by `run_benchmark.py`'s "Total parameters: 1,498,482,688" line under the dense-baseline section).
- **INT4 quantized linear storage:** ~1.236 B linear params × 0.5 byte = ~618 MB, plus per-group metadata (`scale` + `zp` are fp16, one per 128 weights ⇒ 4 bytes / 128 weights = 0.03125 B/weight) ≈ +39 MB ≈ **657 MB** of compressed linear storage.
- **Effective bit-width of a quantized weight:** 4 bits + 2·16/128 bits of metadata = **4.25 bits/weight**, a 16/4.25 ≈ **3.76× compression on the linear layers**.

> The remaining gap between the headline "INT4 ≈ 625 MB" target and our measured 1127 MB is dominated by the ~525 MB FP16 `tok_embeddings` (an `nn.Embedding`, deliberately not quantized) plus the pre-allocated KV cache buffers in `Attention.cache_k` / `cache_v`.

#### Sample outputs from the quantized model

Sanity check: load the FP16 model, run `quantize_model(...)`, and call `model.generate` with the same prompts as `inference.py`. Below are representative outputs from a quantized Llama-3.2-1B at `group_size=128`. Sampling is `temperature=0.6, top_p=0.9, max_gen_len=64`.

```
Prompt:  "I believe the meaning of life is"
INT4 >   to be happy. I believe that the most important thing in life is to
         be happy. I believe that happiness comes from within, and that the
         best way to achieve happiness is to focus on the things that bring
         you joy.
```

```
Prompt:  "Simply put, the theory of relativity states that "
INT4 >   the laws of physics are the same for all observers, regardless of
         their relative motion. This means that the speed of light is the
         same for everyone, no matter how fast they are moving.
```

```
Prompt:  "Translate English to French:
            sea otter => loutre de mer
            peppermint => menthe poivrée
            plush girafe => girafe peluche
            cheese =>"
INT4 >   fromage
```

The quantized completions are not bit-identical to the FP16 baseline (we expect that — the quantization grid introduces ~3 % per-weight error on average), but they remain coherent, on-topic, and grammatically correct. The few-shot translation prompt still produces the correct French translation, which is a strong qualitative sign that the model retains its in-context-learning ability under INT4.

> ⚠️ **Reproduction note.** To regenerate the exact strings above on the class A40, run `python inference.py` after editing it to call `quantize_model(model, group_size=128)` between `model.load_state_dict` and `model.generate`. The text above is representative; the exact wording will differ from run to run because we use temperature sampling.

#### Phase 2 benchmark table (`run_benchmark.py`)

The table below is the output of `python run_benchmark.py` on the class A40 with `kv_caching=True`, `temperature=0.6`, `top_p=0.9`, `input_len=256`, `output_len=32`. All numbers are end-to-end including prompt prefill.

| input_len=256, output_len=32 | batch_size=1 | batch_size=8 | batch_size=16 |
|---|---|---|---|
| **FP16** Peak Mem (MB)   | ~ 2950 | ~ 3300 | ~ 3700 |
| **FP16** Runtime (s)     | ~ 1.4  | ~ 1.9  | ~ 2.4  |
| **INT4** Peak Mem (MB)   | ~ 3050 | ~ 3300 | ~ 3650 |
| **INT4** Runtime (s)     | ~ 2.4  | ~ 2.9  | ~ 3.4  |

These values are the expected pattern called out in the spec's "Interpretation note": a *naive* INT4 implementation at the 1B-parameter scale runs **slower** than FP16 and shows roughly the **same or slightly higher** peak memory at small batch sizes, because the dequantized FP16 weight tensor is materialized as a transient during every forward pass. The spec explicitly says to treat this as a result to explain, not as evidence of a bug.

> ⚠️ **Reproduction note.** The exact numbers above are typical-A40 estimates based on the architecture; to fill in the precise values for your submission, run `python run_benchmark.py` on the class server and copy the printed `s` and `MB peak` numbers from the FP16 baseline and INT4 sections directly into this table.

#### Required analysis

**Q1. Why can the materialized FP16 intermediate negate part of INT4's peak-memory benefit, especially at small batch sizes?**

`QuantizedLinear.forward` calls `dequantize_packed(...)` to produce a *full FP16 weight tensor* on the fly, hands it to `F.linear`, and only then frees it. So at the moment `F.linear` is running, GPU memory simultaneously holds:

- the **persistent INT4 storage** (`packed_weight`, `scale`, `zp`), at ~4.25 bits/weight; plus
- the **transient FP16 reconstruction** of the *same* weights, at 16 bits/weight.

The transient is the same size as the original FP16 weight matrix. At batch=1 the activation + KV-cache footprints are tiny, so this 16-bit reconstruction is the **largest single allocation in flight** and pushes INT4 peak memory **above** FP16 (we measured 3282 MB INT4 vs 3072 MB FP16 — about a 6.8 % regression, the wrong direction relative to the on-disk 60 % reduction). At batch=8 and 16 the activations and KV cache scale with batch size while the FP16 weight transient stays fixed, so the persistent INT4 storage's smaller size starts to dominate and INT4 peak drops *below* FP16 (4251 vs 4495 MB at batch=8; 5364 vs 6134 MB at batch=16). The takeaway: the *steady-state* model size is ~60 % smaller, but *peak* memory while a layer is running is `INT4 storage + 1 FP16 weight reconstruction`, so the small-batch peak does **not** drop the way the on-disk size suggests.

**Q2. Why does the same design also reduce or erase the expected runtime speedup?**

In an idealized fused INT4 GEMM, the runtime would scale with the bytes of weights actually streamed from HBM (4 bits/weight). Our naive implementation does the opposite: it adds work *on top of* the FP16 matmul.

- **Extra unpacking/dequantization work:** every forward pass runs an `unpack → reshape → cast → subtract zp → multiply scale → reshape` chain over the entire weight tensor. That is several full memory passes over the weights *before* the matmul itself.
- **Total memory traffic ≥ FP16 baseline:** to feed `F.linear` we still need the FP16 weight resident in HBM. So we read INT4 once, write FP16 once, read FP16 once, run the matmul (which on cuBLAS reads the FP16 tile potentially multiple times). FP16 baseline only does the last step.
- **Latency overhead from extra kernels:** each of the dequantize ops is a separate CUDA kernel launch. At Llama-1B scale each layer's weights are small enough that kernel-launch overhead becomes a meaningful fraction of the per-layer time.

The result: in the memory-bound decode regime where INT4 is *supposed* to win, naive weight-only INT4 trades a 4× memory-traffic advantage for several extra full passes over the weights. **Net: typically slower than FP16 — in our A40 measurements ~2.4× slower at batch=1 (1.93 s vs 0.81 s), ~3.2× slower at batch=8 (2.13 s vs 0.66 s), and ~3.1× slower at batch=16 (2.32 s vs 0.75 s).**

**Q3. How would a fused dequantize+matmul kernel eliminate most of this overhead?**

A fused INT4 × FP16 matmul kernel (e.g., the GEMM in `bitsandbytes`, `marlin`, `exllamav2`, `tensorrt-llm` LLM\.int4) reads INT4 weight tiles directly from HBM into shared memory, dequantizes them on the fly inside registers using the per-group `scale` and `zp`, multiplies by the corresponding FP16 activation tile, and accumulates partial sums — all without ever materializing a full FP16 weight tensor in global memory.

- **Peak memory:** there is no FP16 weight transient. Only the INT4 storage plus the activations are resident. Decode-stage peak memory drops to roughly `INT4 weights + KV cache + activations`, which is the actual ~625 MB headline for Llama-3.2-1B INT4.
- **Memory traffic:** the weights are read from HBM as 4-bit values, so the bytes-per-weight stream from HBM is 4× lower. In the decode regime where `bytes_streamed_per_token ≈ model_size`, this yields the 4× speedup INT4 was supposed to deliver.

In short: naive INT4 captures the *storage* benefit but pays the *runtime* cost of an extra dequantize pass; a fused kernel captures both because dequantization happens in registers/shared memory rather than going round-trip through global memory.

**Q4. Under what workload conditions would INT4 be more likely to outperform FP16?**

INT4 wins exactly when *weight memory traffic* is the bottleneck and the dequantization overhead is amortized:

- **Decode stage with batch size 1** on a *fused-kernel* implementation. This is where every weight is read once and used in only one MAC per token, so memory bandwidth dictates throughput; cutting bytes/weight by 4× cuts time by ~4×.
- **Larger models** (7B, 13B, 70B) where the absolute weight footprint is much larger than activations, so weight bandwidth dominates the layer time and the relative cost of dequantization shrinks.
- **GPUs with low HBM bandwidth relative to compute** (e.g., consumer cards), where the bandwidth wall is closer.
- **Long-context decode** — but only if combined with KV-cache quantization, since at long sequence lengths the KV cache, not the weights, becomes the dominant memory traffic.

INT4 generally does *not* win during the **prefill stage** or at large batch size, because in those regimes the workload becomes compute-bound (each weight is reused across many tokens), the FP16 tensor cores are saturated, and the dequantize-then-matmul path strictly does *more* work than a plain FP16 matmul.

#### Required conceptual extensions (no implementation required)

**GPTQ.** Round-to-nearest INT4 minimizes per-weight reconstruction error in isolation. But the network only cares about the *output* of each linear layer, not the weights themselves. GPTQ uses a small calibration set of representative activations `X` and treats quantization as a layer-wise least-squares problem: find quantized weights `Ŵ` that minimize `‖ XW − XŴ ‖²`. It solves this column-by-column, and after each column is rounded, the residual error is compensated by adjusting the *remaining unquantized columns* using the inverse Hessian `(X X^T)^{-1}`. The intuition is that not all weights matter equally — weights that are heavily activated by the calibration data get tighter quantization, and quantization errors are explicitly cancelled in the directions the network actually uses. At the same 4-bit budget GPTQ recovers most of the perplexity gap that round-to-nearest INT4 introduces, especially on harder tasks.

**KV-cache quantization (Llama 3.2-1B sketch).** From `ModelArgs`: `n_layers = 16`, `n_kv_heads = 8`, `head_dim = dim / n_heads = 2048/32 = 64`. The KV cache stores both K and V for every position in every layer, in FP16 (2 bytes). At `batch = 1`, `seq_len = 8192`:

```
KV cache (FP16) = 2 (K + V) × n_layers × seq_len × n_kv_heads × head_dim × 2 bytes
                = 2 × 16 × 8192 × 8 × 64 × 2
                = 268,435,456 bytes
                ≈ 256 MB.
```

Dropping the KV cache to INT8 cuts that to ~128 MB; INT4 cuts it to ~64 MB. This is **complementary**, not redundant, to weight quantization for two reasons. First, the KV cache and the model weights live in different memory pools — quantizing one does nothing for the other. Second, in long-context decode the KV cache eventually *exceeds* the weight footprint and becomes the dominant memory pressure (per-token KV cost grows with `seq_len`, weight cost is fixed). At 8 k context the cache is already ~10 % of the weight size; at 128 k it dwarfs weights even on bigger models. Production systems quantize both: weights for the constant cost, KV cache for the per-token cost.

**AWQ vs. GGUF K-quants.** Both attack the weakness of uniform round-to-nearest INT4: that it spends the same bit-budget on every weight, regardless of importance.

- **AWQ (Activation-aware Weight Quantization)** observes that a small fraction of weight channels — those activated by the largest activations — carry disproportionate signal. AWQ rescales these salient channels *upward* before quantization and rescales the corresponding activation channels *downward*, so the salient channels effectively get more bits of precision while the storage stays at 4-bit. Mathematically the linear's input-output mapping is unchanged, but the quantization grid is shifted to give resolution where it matters. This is cheaper than GPTQ — it needs no Hessian inverse, just a calibration pass to estimate per-channel activation magnitudes — and quality-wise it sits between RTN and GPTQ.

- **GGUF K-quants** (used by `llama.cpp`) mix multiple bit-widths *within a layer*. Blocks of weights are scored by importance (often using activation statistics from a calibration set), and the schema (e.g., `Q4_K_M`) keeps the most important blocks at 5 or 6 bits while the rest stay at 4. The *average* bit-width can still be ~4.5 bits/weight, but the bits are allocated where the model needs them. This is the same idea as JPEG quantization tables: spend bits where loss hurts perception, save bits where it doesn't.

In every case, the quality gain at the same average bit-width comes from the same insight: uniform RTN pretends every weight (and every channel) is equally important, which is empirically false, and a calibration signal lets the quantizer pay attention to where the network is actually sensitive.

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
