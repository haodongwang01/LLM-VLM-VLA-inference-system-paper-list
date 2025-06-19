# LLM-VLM-VLA-inference-system-paper-list
Summarize foundational model inference system paper


## Preface

本仓库记录关于LLM/VLM/VLA模型的文章。



###  🔥 Updates

- 2025-06 主要关注LLM推理系统的优化
## Directory

* [LLM](#llm) 
  * [Inference Serving Systems](#inference-serving-systems)
  * [On-device Inference Systems](#on-device-inference-systems)
  * [KV Cache Optimization](#KV-cache-optimization)
  * [Quantization](#quantization)
  * [Speculative Decoding](#speculative-decoding)
  * [Open Source Code](#open-source-code)
* [VLM](#vlm)


# LLM

## Inference Serving Systems


1. **Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve** (OSDI 2024) [[paper]](https://arxiv.org/abs/2403.02310) 

2. **DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving** (OSDI 2024) [[paper]] (https://arxiv.org/pdf/2401.09670)


    


## On-device Inference Systems


1. **PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU** (SOSP 2024) [[paper]](https://arxiv.org/pdf/2312.12456) ⭐⭐⭐

2. **PowerInfer-2: Fast Large Language Model Inference on a Smartphone** (Arxiv Dec 2024) [[paper]](https://arxiv.org/abs/2406.06282) ⭐⭐

3. **Fast On-device LLM Inference with NPUs** (ASPLOS 2025) [[paper]](https://arxiv.org/pdf/2407.05858) ⭐⭐

4. **FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU** [[paper]](https://arxiv.org/pdf/2303.06865) ⭐

## KV Cache Optimization

1. **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness** (NeurIPS 2024) [[paper]](https://arxiv.org/abs/2205.14135) ⭐⭐⭐
2. **Efficient Memory Management for Large Language Model Serving with PagedAttention** (SOSP 2023) [[paper]](https://arxiv.org/pdf/2309.06180) ⭐⭐⭐



## Quantization
1. **GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers** (ICLR 2023) [[paper]] (https://arxiv.org/abs/2210.17323) ⭐⭐
2. **AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration** (MLSys 2024 Best Paper) [[paper]](https://arxiv.org/abs/2306.00978) ⭐⭐⭐
3. **SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models** (ICML 2023) [[paper]](https://arxiv.org/abs/2211.10438) ⭐⭐

## Speculative Decoding
1. **SpecInfer: Accelerating Generative LLM Serving with Speculative Inference and Token Tree Verification** (ASPLOS 2024) [[paper]](https://arxiv.org/pdf/2305.09781)
2. **EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty** (ICML 2024) [[paper]](https://arxiv.org/pdf/2401.15077)
3. **Simple LLM Inference Acceleration Framework with Multiple Decoding Heads** (ICML 2024) [[paper]](https://arxiv.org/pdf/2401.10774)

## Open Source Code
1. **LLM inference engine: vllm** (https://github.com/vllm-project/vllm)
2. **LLM inference engine (omly use cpp): llama.cpp** (https://github.com/ggml-org/llama.cpp)
3. **Speculative Decoding: EAGLE-1/2/3** (https://github.com/SafeAILab/EAGLE)
4. **Attention: Flashattention-1/2/3** (https://github.com/Dao-AILab/flash-attention)


# VLM

