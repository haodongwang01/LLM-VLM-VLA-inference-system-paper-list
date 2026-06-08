# DiT-WM-WAM
## Linear quantization

1. **VIDIT-Q: EFFICIENT AND ACCURATE QUANTIZATION OF DIFFUSION TRANSFORMERS FOR IMAGE AND VIDEO GENERATION** (ICLR 2025) [[paper]](https://arxiv.org/abs/2406.02540)
	* problems：one share scale for multiple variant dimensions(token-wise, input channel-wise, timestep-wise, CFG-wise variance)
	- methods: 
	    * Token-wise activation quantization
	    * Dynamic activation quantization
	    * Timestep-wise channel balancing
	    * W8A8+W4A8 mixed precision


## Sparse Attention/Attention Compression

1. **DiTFastAttn: Attention Compression for Diffusion Transformer Models** (NeurIPS 2024) [[paper]](https://arxiv.org/pdf/2406.08552)
    * problems: spatial/temporal/conditional redundancy in attention
    * methods: 
        * Window Attention with Residual Sharing
        * Attention Sharing across Timesteps
        * Attention Sharing across CFG
2. **DiTFastAttnV2: Head-wise Attention Compression for Multi-Modality Diffusion Transformers**(ICCV 2025) [[paper]](https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_DiTFastAttnV2_Head-wise_Attention_Compression_for_Multi-Modality_Diffusion_Transformers_ICCV_2025_paper.html)
    - problems: MMDiT
    - methods: 
        * Head-wise Arrow Attention
        * Efficient Fused Kernel
        * Head-wise Dynamic Caching

## Sparse attention+Attention quantization

1. **PAROAttention: Pattern-Aware ReOrdering for Efficient Sparse and Quantized Attention in Visual Generation Models** (NeurIPS 2025)[[paper]](https://openreview.net/pdf?id=UPELg2oUo3)
    * problems: irregular, diagonal, outlier-heavy attention patterns
    * methods: Pattern-Aware token Reordering, block-wise pattern

## Diffusion quantization correction

Consensus: error propagation in denoising chain

1. **Q-Sched: Pushing the Boundaries of Few-Step Diffusion Models with Quantization-Aware Scheduling** (ICML 2026) [[paper]](https://openreview.net/pdf?id=lqCvR0BDss)
     * problems: few-step DiT 
     * methods: adapts the diffusion sampler rather than the model weights
         * Quantization-aware preconditioning coefficients
         * JAQ loss reference-free

2. **Absorbing Quantization Error by Deformable Noise Scheduler for Diffusion Models** (ICML 2026) [[paper]](https://openreview.net/forum?id=KUXeiU7eNR)
     * methods: timestep shift


