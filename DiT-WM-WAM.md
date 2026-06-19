# DiT
## Linear quantization


1. **PTQ4DiT: Post-training Quantization for Diffusion Transformers**(NeurIPS 2024)[[paper]](https://arxiv.org/abs/2405.16005)
    - problems:
         - complex distribution patterns: salient channels in linear layers
         - temporal dynamics: salient activation channels change with timesteps (conditioning MLP)
    - methods:
         - Channel-wise Salience Balancing
         - salient channel varied-> Spearman’s ρ-guided Salience Calibration
         - Re-Parameterization (complete offline/static)
    - class-conditioned generation: image (DiT-XL/2 models)
    

2. **Q-DiT: Accurate Post-Training Quantization for Diffusion Transformers**(CVPR 2025) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/html/Chen_Q-DiT_Accurate_Post-Training_Quantization_for_Diffusion_Transformers_CVPR_2025_paper.html)
    * problems:
        - significant variance of weights and activations across input channels
        - varying activations across **different timesteps**
	* methods：
	    * Automatic Quantization Granularity , FID/FVD guiding 
	    * Sample-wise Dynamic Activation Quantization
	- class-conditioned generation: image (pre-trained DiT-XL/2), video(STDiT3 model from the Open-Sora )

3. **VIDIT-Q: EFFICIENT AND ACCURATE QUANTIZATION OF DIFFUSION TRANSFORMERS FOR IMAGE AND VIDEO GENERATION** (ICLR 2025) [[paper]](https://arxiv.org/abs/2406.02540)
	* problems：
		* text-to-image/video generation
		* token-wise, input channel-wise, timestep-wise, CFG-wise variance
		* MSE not enough
	- methods: 
	    * Fine-grained grouping and Dynamic Quantization
	    * Timestep-wise channel balancing
	    * (W8A8+W4A8) mixed precision **decouple metrics**
	- video(OpenSORA HPC-AI), image(PixArt-α model)

4. **Q-VDiT: Towards Accurate Quantization and Distillation of Video-Generation Diffusion Transformers**  (ICML2025)[[paper]](https://arxiv.org/pdf/2505.22167)
    - problems：video generation models require modeling additional temporal(not only spatial)
    - methods：
        - Token-aware Quantization Estimator
        - Temporal Maintenance Distillation
        - fused kernel(LoRunner Kernel?)
    - video(OpenSORA HPC-AI)

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

# WM
## Survey

1.  **An Empirical Study of World Model Quantization** [[paper]](https://arxiv.org/pdf/2602.02110)
      * DINO-WM (Action-conditioned ViT) = encoder+predictor
      * RTN/Smoothquant/AWQ/OMSE/Omniquant
 

