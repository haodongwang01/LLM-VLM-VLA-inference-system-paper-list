# LLM-VLM-VLA-inference-system-paper-list
Summarize foundational model inference system paper


## Preface

本仓库记录关于LLM/VLM/VLA模型的文章。



###  🔥 Updates

- 2025-06 主要关注LLM推理系统的优化
## Directory
*[Survey](#survey)
* [LLM](#llm) 
  * [Inference Serving Systems](#inference-serving-systems)
     * [GPU Systems](#GPU-systems)
     * [GPU-CPU Systems](#GPU-CPU-systems)
  * [On-device Inference Systems](#on-device-inference-systems)
  * [KV Cache Optimization](#KV-cache-optimization)
  * [Post-Training Quantization](#post-training-quantization)
    * [Weight-Only Quantization](#weight-only-quantization)
       *[Scale Quantization](#scale-quantization)
       *[Vector Quantization](#vector-quantization)
       *[Additive Quantization](#additive-quantization)
       *[Product Quantization](#product-quantization)
    * [Weight-Activation Quantization](#weight-activation-quantization)
    * [Quantization Kernel](#quantization-kernel)
    * [Quantization Serving System](#quantization-serving)
  * [Quantization Training](#quantization-training)
  * [Speculative Decoding](#speculative-decoding)
  * [Open Source Code](#open-source-code)
* [VLM](#vlm)
  * [Sparse Attention](#sparse-attention)
  * [Pruning](#pruning)
  * [Quantization](#quantization)



# Survey
1. **Towards Efficient Generative Large Language Model Serving: A Survey from Algorithms to Systems** (ACM Computing Surveys) [[paper]](https://arxiv.org/pdf/2312.15234) ⭐⭐
2. **Large Language Model Inference Acceleration: A Comprehensive Hardware Perepective** (ArXiV 2025.06) [[paper]](https://arxiv.org/pdf/2410.04466) ⭐⭐


# LLM

## Inference Serving Systems
### GPU Systems
<a id="GPU-systems"></a>
1. **Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve** (OSDI 2024) [[paper]](https://arxiv.org/abs/2403.02310) ⭐⭐
2. **DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving** (OSDI 2024) [[paper]](https://arxiv.org/pdf/2401.09670)



### GPU-CPU Systems
<a id="GPU-CPU-systems"></a>
**MoE-Lightning: High-Throughput MoE Inference on Memory-constrained GPUs** (ASPLOS 2025)  [[paper]](https://arxiv.org/pdf/2411.11217) ⭐⭐
**HybriMoE: Hybrid CPU-GPU Scheduling and Cache Management for Efficient MoE Inference** (DAC 2025) [[paper]](https://arxiv.org/abs/2504.05897)


## On-device Inference Systems

1. **PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU** (SOSP 2024) [[paper]](https://arxiv.org/pdf/2312.12456) ⭐⭐⭐
2. **PowerInfer-2: Fast Large Language Model Inference on a Smartphone** (Arxiv Dec 2024) [[paper]](https://arxiv.org/abs/2406.06282) ⭐⭐
3. **Fast On-device LLM Inference with NPUs** (ASPLOS 2025) [[paper]](https://arxiv.org/pdf/2407.05858) ⭐⭐
4. **FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU** [[paper]](https://arxiv.org/pdf/2303.06865) ⭐
5. **H2-LLM: Hardware-Dataflow Co-Exploration for Heterogeneous Hybrid-Bonding-based Low-Batch LLM Inference** [[paper]](https://dl.acm.org/doi/10.1145/3695053.3731008)



## KV Cache Optimization

1. **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness** (NeurIPS 2024) [[paper]](https://arxiv.org/abs/2205.14135) ⭐⭐⭐
2. **Efficient Memory Management for Large Language Model Serving with PagedAttention** (SOSP 2023) [[paper]](https://arxiv.org/pdf/2309.06180) ⭐⭐⭐
3. **Efficient Streaming Language Models with Attention Sinks** (ICLR 2024) [[paper]](https://arxiv.org/abs/2309.17453) ⭐⭐⭐
4. **InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management** (OSDI 2024) [[paper]](https://arxiv.org/abs/2406.19707) ⭐⭐⭐


## Post-Training Quantization
<a id="post-training-quantization"></a>

### Weight-Only Quantization
<a id="weight-only-quantization"></a>

#### Scale Quantization
<a id="scale-quantization"></a>
**GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers** (ICLR 2023) [[paper]](https://arxiv.org/abs/2210.17323) ⭐⭐

**AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration** (MLSys 2024 Best Paper) [[paper]](https://arxiv.org/abs/2306.00978) ⭐⭐⭐

**QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks** (ICML 2024) [[paper]](https://arxiv.org/pdf/2402.04396) 

#### Vector Quantization
<a id="vector-quantization"></a>
**GPTVQ: The Blessing of Dimensionality for LLM Quantization** (ICML 2025) [[paper]](https://arxiv.org/abs/2402.15319)
**VPTQ: Extreme Low-bit Vector Post-Training Quantization for Large Language Models** (EMNLP 2024) [[paper]](https://arxiv.org/abs/2409.17066)

#### Additive Quantization
<a id="additive-quantization"></a>
**AQLM: Extreme Compression of Large Language Models via Additive Quantization** (ICML 2024) [[paper]](https://arxiv.org/abs/2401.06118)


#### Product Quantization
<a id="product-quantization"></a>
**PQCache: Product Quantization-based KVCache for Long Context LLM Inference** (SIGMOD 2025) [[paper]](https://arxiv.org/abs/2407.12820)

---

### Weight-Activation Quantization
<a id="weight-activation-quantization"></a>

**SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models** (ICML 2023) [[paper]](https://arxiv.org/abs/2211.10438) ⭐⭐

**MixQ: Taming Dynamic Outliers in Mixed-Precision Quantization by Online Prediction** (SC 2024) [[paper]](https://dl.acm.org/doi/10.1109/SC41406.2024.00080)

**OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models** (ICLR 2024 Spotlight) [[paper]](https://arxiv.org/abs/2308.13137)

**QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs** (NeurIPS 2024) [[paper]](https://arxiv.org/abs/2404.00456) 

**Atom: Low-bit Quantization for Efficient and Accurate LLM Serving** (MLSys 2024) [[paper]](https://arxiv.org/abs/2310.19102) 

**SpinQuant: LLM quantization with learned rotations** (ICLR 2025) [[paper]](https://arxiv.org/abs/2405.16406)

**OstQuant: Refining Large Language Model Quantization with Orthogonal and Scaling Transformations for Better Distribution Fitting** (ICLR 2025) [[paper]](https://arxiv.org/abs/2501.13987)

**FlatQuant: Flatness Matters for LLM Quantization** (ICML 2025) [[paper]](https://arxiv.org/abs/2410.09426)

**GPTAQ: Efficient Finetuning-Free Quantization for Asymmetric Calibration** (ICML 2025) [[paper]](https://arxiv.org/abs/2504.02692)

---

### Quantization Kernel 
<a id="quantization-kernel"></a>

**MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models** (PPoPP 2025) [[paper]](https://arxiv.org/abs/2408.11743) 

**T-MAC: CPU Renaissance via Table Lookup for Low-Bit LLM Deployment on Edge** (EuroSys 2025) [[paper]](https://arxiv.org/abs/2407.00088)

**DecDEC: A Systems Approach to Advancing Low-Bit LLM Quantization** (OSDI 2025) [[paper]](https://arxiv.org/abs/2412.20185)

**LiquidGEMM: Hardware-Efficient W4A8 GEMM Kernel for High-Performance LLM Serving** (ArXiV 2025.09) [[paper]](https://arxiv.org/abs/2509.01229) 

**BitDecoding: Unlocking Tensor Cores for Long-Context LLMs with Low-Bit KV Cache** (ArXiV 2025.08) [[paper]](https://arxiv.org/abs/2503.18773) 

---

### Quantization Serving System
<a id="quantization-serving"></a>
**COMET: Towards Practical W4A4KV4 LLMs Serving** (ASPLOS 2025) [[paper]](https://arxiv.org/abs/2410.12168) 

**QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving** (MLSys 2025) [[paper]](https://arxiv.org/abs/2405.04532) 


## Quantization Training
1. **Scaling FP8 training to trillion-token LLMs** (ICLR 2025 Spotlight) [[paper]](https://arxiv.org/abs/2409.12517)
2. **Optimizing Large Language Model Training Using FP4 Quantization** (ICML 2025) [[paper]](https://arxiv.org/abs/2501.17116)
3. **FP4 All the Way: Fully Quantized Training of LLMs** (ArXiV 2025.05) [[paper]](https://arxiv.org/abs/2505.19115)
4. **Pretraining Large Language Models with NVFP4** (ArXiV 25.09) [[paper]](https://arxiv.org/abs/2509.25149v1)



## Speculative Decoding
1. **SpecInfer: Accelerating Generative LLM Serving with Speculative Inference and Token Tree Verification** (ASPLOS 2024) [[paper]](https://arxiv.org/pdf/2305.09781) ⭐⭐
2. **EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty** (ICML 2024) [[paper]](https://arxiv.org/pdf/2401.15077) ⭐⭐⭐
3. **Simple LLM Inference Acceleration Framework with Multiple Decoding Heads** (ICML 2024) [[paper]](https://arxiv.org/pdf/2401.10774) ⭐



## Open Source Code
1. **LLM inference engine: vllm** (https://github.com/vllm-project/vllm)
2. **LLM inference engine (omly use cpp): llama.cpp** (https://github.com/ggml-org/llama.cpp)
3. **Speculative Decoding: EAGLE-1/2/3** (https://github.com/SafeAILab/EAGLE)
4. **Attention: Flashattention-1/2/3** (https://github.com/Dao-AILab/flash-attention)


# VLM

## Sparse Attention
1. **MMIference: Accelerating Pre-filling for Long-Context VLMs via Modality-Aware Permutation Sparse Attention** (ICML 2025) [[paper]](https://arxiv.org/abs/2504.16083) 
2. **XAttention: Block Sparse Attention with Antidiagonal Scoring** (ICML 2025) [[paper]](https://arxiv.org/pdf/2503.16428)

## Pruning
1. **TopV: Compatible Token Pruning with Inference Time Optimization for Fast and Low-Memory Multimodal Vision Language Model** (CVPR 2025)[[paper]](https://arxiv.org/abs/2503.18278)

## Quantization
1. **Q-VLM:Post-training Quantization for Large Vision-Language Models** (NeurIPS 2024)[[paper]](https://arxiv.org/abs/2410.08119)
2. **MBQ: Modality-Balanced Quantization for Large Vision-Language Models** (CVPR 2025)[[paper]](https://arxiv.org/html/2412.19509v1)
