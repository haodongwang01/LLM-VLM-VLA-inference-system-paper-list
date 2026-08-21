# Efficient DiT / World Model / World-Action Model Inference

> 更新日期：2026-08-21。本文聚焦 Diffusion Transformer（DiT）、生成式世界模型（WM）与世界动作模型（WAM）的推理效率；同一论文只按其主要技术路线收录一次，会议尚未正式发表的条目统一标为 arXiv。
>
> 每篇论文下方仅保留一句“核心贡献”，便于快速浏览；速度数字均来自原论文，因模型、硬件、分辨率和质量约束不同，不宜直接横向比较。

<!-- 暂时隐藏 DiT 相关论文；需要恢复时删除本注释起止标记即可。

## 1. Diffusion Transformer（DiT）

### 1.1 模型量化

1. **PTQ4DiT: Post-training Quantization for Diffusion Transformers** — *NeurIPS 2024* · [[paper]](https://arxiv.org/abs/2405.16005)
   - 核心贡献：通过通道显著性平衡、时间步感知校准和离线重参数化处理 DiT 的通道离群值与时间动态，实现高精度训练后量化。

2. **HQ-DiT: Efficient Diffusion Transformer with FP4 Hybrid Quantization** — *arXiv 2024* · [[paper]](https://arxiv.org/abs/2405.19751)
   - 核心贡献：以 FP4 权重与激活配合裁剪范围选择和恒等变换，使 DiT-XL/2 在 W4A4 下仅产生很小的生成质量损失。

3. **ViDiT-Q: Efficient and Accurate Quantization of Diffusion Transformers for Image and Video Generation** — *ICLR 2025* · [[paper]](https://arxiv.org/abs/2406.02540)
   - 核心贡献：以细粒度分组、动态量化、时间步通道平衡和混合精度同时适配图像与视频 DiT 的 token、通道、时间步及 CFG 分布变化。

4. **Q-DiT: Accurate Post-Training Quantization for Diffusion Transformers** — *CVPR 2025* · [[paper]](https://openaccess.thecvf.com/content/CVPR2025/html/Chen_Q-DiT_Accurate_Post-Training_Quantization_for_Diffusion_Transformers_CVPR_2025_paper.html)
   - 核心贡献：通过自动搜索量化粒度与逐样本动态激活量化缓解跨通道和跨时间步方差，在图像与视频 DiT 上改善 PTQ 精度。

5. **TQ-DiT: Efficient Time-Aware Quantization for Diffusion Transformers** — *arXiv 2025* · [[paper]](https://arxiv.org/abs/2502.04056)
   - 核心贡献：以多区域量化处理不对称数值分布，并按时间步分组共享量化参数以降低激活随去噪过程变化造成的误差。

6. **FP4DiT: Towards Effective Floating Point Quantization for Diffusion Transformers** — *arXiv 2025* · [[paper]](https://arxiv.org/abs/2503.15465)
   - 核心贡献：将自适应舍入推广到浮点 PTQ，并用在线激活量化在 PixArt 和 Hunyuan 等模型上实现优于整数 PTQ 的 W4A6/W4A8 推理。

7. **Q-VDiT: Towards Accurate Quantization and Distillation of Video-Generation Diffusion Transformers** — *ICML 2025* · [[paper]](https://arxiv.org/abs/2505.22167)
   - 核心贡献：以 Token-aware Quantization Estimator 补偿双维量化误差，并用 Temporal Maintenance Distillation 保持视频帧间时空关联。

8. **S²Q-VDiT: Accurate Quantized Video Diffusion Transformer with Salient Data and Sparse Token Distillation** — *arXiv 2025* · [[paper]](https://arxiv.org/abs/2508.04016)
   - 核心贡献：以显著校准数据选择和稀疏 token 蒸馏聚焦视频中的关键信息，从而提高低比特视频 DiT 的量化保真度。

9. **VETA-DiT: Variance-Equalized and Temporally Adaptive Quantization for Efficient 4-bit Diffusion Transformers** — *NeurIPS 2025* · [[paper]](https://openreview.net/forum?id=z0BgfL1FRV)
   - 核心贡献：通过方差均衡变换和时间自适应校准抑制 DiT 的通道离群与时间分布漂移，实现图像、视频模型的 4-bit PTQ。

10. **DVD-Quant: Data-free Video Diffusion Transformers Quantization** — *ICLR 2026* · [[paper]](https://arxiv.org/abs/2505.18663)
    - 核心贡献：结合无数据网格细化、旋转量化和按时间差异切换位宽，在无需校准数据的条件下首次实现高保真的视频 DiT W4A4 PTQ。

11. **Timestep-Aware SVDQuant-GPTQ for W4A4 Quantization of Wan2.2-I2V** — *arXiv 2026* · [[paper]](https://arxiv.org/abs/2605.27003)
    - 核心贡献：针对 Wan2.2 双专家架构联合使用 SVD 离群补偿、GPTQ 残差量化与分专家分时间段裁剪搜索，将 W4A4 峰值显存降低 59.3%。

12. **OrbitQuant: Data-Agnostic Quantization for Image and Video Diffusion Transformers** — *arXiv 2026* · [[paper]](https://arxiv.org/abs/2607.02461)
    - 核心贡献：在随机置换分块 Hadamard 旋转后的统一分布上复用固定 Lloyd–Max 码本，无需数据校准即可跨图像、视频和时间步量化并将图像 DiT 推进至可用的 W2A4。

13. **Q-DiT4SR: Exploration of Detail-Preserving Diffusion Transformer Quantization for Real-World Image Super-Resolution** — *arXiv 2026* · [[paper]](https://arxiv.org/abs/2602.01273)
    - 核心贡献：以分层 SVD 和时空混合精度分配保护超分辨率局部纹理，使 DiT4SR 在 W4A4 下兼顾模型压缩与细节质量。

### 1.2 稀疏化、注意力压缩与低精度注意力

1. **DiTFastAttn: Attention Compression for Diffusion Transformer Models** — *NeurIPS 2024* · [[paper]](https://arxiv.org/abs/2406.08552)
   - 核心贡献：联合窗口注意力与残差共享、跨时间步注意力共享和 CFG 分支共享，系统消除 DiT 的空间、时间与条件冗余。

2. **Real-Time Video Generation with Pyramid Attention Broadcast** — *ICLR 2025* · [[paper]](https://arxiv.org/abs/2408.12588)
   - 核心贡献：PAB 根据空间、时间和交叉注意力在去噪过程中的不同稳定性，以金字塔式广播复用输出并结合序列并行实现实时视频生成。

3. **SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration** — *ICLR 2025* · [[paper]](https://arxiv.org/abs/2410.02367)
   - 核心贡献：设计近乎无损的 INT8 注意力内核，在语言、图像和视频生成模型上以即插即用方式显著提高注意力吞吐。

4. **Efficient-vDiT: Efficient Video Diffusion Transformers With Attention Tile** — *arXiv 2025* · [[paper]](https://arxiv.org/abs/2502.06155)
   - 核心贡献：从视频注意力中的重复 tile 模式构造随帧数线性扩展的稀疏 3D 注意力，并结合分段一致性蒸馏和序列并行获得端到端加速。

5. **SpargeAttn: Accurate Sparse Attention Accelerating Any Model Inference** — *ICML 2025* · [[paper]](https://arxiv.org/abs/2502.18137)
   - 核心贡献：用两阶段在线过滤器预测并跳过低贡献注意力块，同时融合量化内核，在多类生成模型上实现训练无关的稀疏注意力加速。

6. **SageAttention2: Efficient Attention with Thorough Outlier Smoothing and Per-thread INT4 Quantization** — *ICML 2025* · [[paper]](https://openreview.net/forum?id=nC8XliUxeg)
   - 核心贡献：通过彻底的离群值平滑、逐线程 INT4 QK 量化和 FP8 PV 计算，在维持精度的同时进一步提升注意力内核速度。

7. **DiTFastAttnV2: Head-wise Attention Compression for Multi-Modality Diffusion Transformers** — *ICCV 2025* · [[paper]](https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_DiTFastAttnV2_Head-wise_Attention_Compression_for_Multi-Modality_Diffusion_Transformers_ICCV_2025_paper.html)
   - 核心贡献：针对 MMDiT 引入逐头 Arrow Attention、动态缓存和融合内核，使不同注意力头可采用适配自身结构的压缩策略。

8. **EDiT: Efficient Diffusion Transformers with Linear Compressed Attention** — *ICCV 2025* · [[paper]](https://openaccess.thecvf.com/content/ICCV2025/html/Becker_EDiT_Efficient_Diffusion_Transformers_with_Linear_Compressed_Attention_ICCV_2025_paper.html)
   - 核心贡献：以卷积调制查询并空间聚合键值，再为多模态输入混合线性与标准注意力，将 DiT/MM-DiT 的注意力复杂度降为线性。

9. **PAROAttention: Pattern-Aware ReOrdering for Efficient Sparse and Quantized Attention in Visual Generation Models** — *NeurIPS 2025* · [[paper]](https://arxiv.org/abs/2506.16054)
   - 核心贡献：通过模式感知 token 重排把分散且不规则的视觉注意力转化为硬件友好的块结构，从而统一支持 20%–30% 密度与 INT8/INT4 计算。

10. **FPSAttention: Training-Aware FP8 and Sparsity Co-Design for Fast Video Diffusion** — *NeurIPS 2025 Spotlight* · [[paper]](https://openreview.net/forum?id=T62TYoF8R3)
    - 核心贡献：共同训练 FP8 量化与 3D tile 稀疏模式，并用时间步自适应策略和专用内核在视频扩散上获得最高 4.96 倍加速。

11. **Attention Surgery: An Efficient Recipe to Linearize Your Video Diffusion Transformer** — *arXiv 2025* · [[paper]](https://arxiv.org/abs/2509.24899)
    - 核心贡献：以少量蒸馏微调将预训练视频 DiT 的部分 softmax 注意力替换为线性/混合注意力，在 Wan2.1 上降低注意力 FLOPs 而保持视频质量。

12. **QuantSparse: Comprehensively Compressing Video Diffusion Transformer with Model Quantization and Attention Sparsification** — *ICLR 2026* · [[paper]](https://proceedings.iclr.cc/paper_files/paper/2026/hash/94359ca6e248af69b8b6854668ae9782-Abstract-Conference.html)
    - 核心贡献：用多尺度显著注意力蒸馏和二阶稀疏注意力重参数化联合抑制量化噪声与稀疏损失，在 HunyuanVideo-13B 上同时压缩存储并加速推理。

13. **Trainable Log-linear Sparse Attention for Efficient Diffusion Transformers** — *CVPR 2026* · [[paper]](https://openaccess.thecvf.com/content/CVPR2026/html/Zhou_Trainable_Log-linear_Sparse_Attention_for_Efficient_Diffusion_Transformers_CVPR_2026_paper.html)
    - 核心贡献：LLSA 以分层结构将超长视觉 token 的选择和注意力成本同时降至对数线性复杂度，并在高分辨率 DiT 上显著加速训练与推理。

14. **Just-in-Time: Training-Free Spatial Acceleration for Diffusion Transformers** — *arXiv 2026* · [[paper]](https://arxiv.org/abs/2603.10744)
    - 核心贡献：JiT 仅在动态挑选的锚 token 上计算空间近似 ODE，再用确定性 micro-flow 扩展回完整潜变量，从空间维度实现无需训练的稀疏加速。

15. **DiffSparse: Accelerating Diffusion Transformers with Learned Token Sparsity** — *ICLR 2026* · [[paper]](https://arxiv.org/abs/2604.03674)
    - 核心贡献：通过可微网络与动态规划学习逐层 token 缓存稀疏率，并以两阶段训练消除缓存方法依赖完整计算步骤的问题。

### 1.3 特征、层与时间步缓存

1. **Learning-to-Cache: Accelerating Diffusion Transformer via Layer Caching** — *NeurIPS 2024* · [[paper]](https://arxiv.org/abs/2406.01733)
   - 核心贡献：L2C 以层为缓存单元并通过可微优化学习随时间步变化的静态路由图，在几乎不损质量的前提下删除大量重复层计算。

2. **Δ-DiT: A Training-Free Acceleration Method Tailored for Diffusion Transformers** — *arXiv 2024* · [[paper]](https://arxiv.org/abs/2406.01125)
   - 核心贡献：根据前层负责轮廓、后层负责细节的阶段性规律，在早期缓存后部块、晚期缓存前部块，并以 Delta-Cache 修正输入偏差。

3. **Token Caching for Diffusion Transformer Acceleration** — *arXiv 2024* · [[paper]](https://arxiv.org/abs/2409.18523)
   - 核心贡献：TokenCache 联合学习 token 重要性、选择适合缓存的层，并用两阶段轮转调度决定时间步，从 token 粒度削减跨步重复计算。

4. **FasterCache: Training-Free Video Diffusion Model Acceleration with High Quality** — *ICLR 2025* · [[paper]](https://arxiv.org/abs/2410.19355)
   - 核心贡献：通过动态特征复用保持相邻步细微变化，并利用条件与无条件分支冗余设计 CFG-Cache，提高视频扩散缓存的质量—速度权衡。

5. **Timestep Embedding Tells: It’s Time to Cache for Video Diffusion Model** — *CVPR 2025* · [[paper]](https://arxiv.org/abs/2411.19108)
   - 核心贡献：TeaCache 用时间步嵌入调制的噪声输入廉价估计模型输出变化，据此自适应决定何时复用缓存。

6. **Adaptive Caching for Faster Video Generation with Diffusion Transformers** — *ICCV 2025* · [[paper]](https://arxiv.org/abs/2411.02397)
   - 核心贡献：AdaCache 根据每个样本的特征变化自适应安排缓存，并以运动正则化把更多计算分配给高运动视频区域或阶段。

7. **From Reusing to Forecasting: Accelerating Diffusion Models with TaylorSeers** — *arXiv 2025* · [[paper]](https://arxiv.org/abs/2503.06923)
   - 核心贡献：TaylorSeer 不再直接复用旧特征，而以历史特征的高阶有限差分和 Taylor 展开预测未来时间步特征，从而支撑更激进的跳步。

8. **Accelerating Diffusion Transformer via Gradient-Optimized Cache** — *ICCV 2025* · [[paper]](https://openaccess.thecvf.com/content/ICCV2025/html/Qiu_Accelerating_Diffusion_Transformer_via_Gradient-Optimized_Cache_ICCV_2025_paper.html)
   - 核心贡献：GOC 传播缓存特征的梯度差并感知轨迹拐点，以极低附加成本补偿高缓存比例下累积的生成误差。

9. **OmniCache: A Trajectory-Oriented Global Perspective on Training-Free Cache Reuse for Diffusion Transformer Models** — *ICCV 2025* · [[paper]](https://openaccess.thecvf.com/content/ICCV2025/html/Chu_OmniCache_A_Trajectory-Oriented_Global_Perspective_on_Training-Free_Cache_Reuse_for_ICCV_2025_paper.html)
   - 核心贡献：从完整采样轨迹而非局部相似度分配缓存位置，并动态估计和滤除复用噪声，避免缓存过度集中在难以纠错的后期步骤。

10. **FastCache: Fast Caching for Diffusion Transformer Through Learnable Linear Approximation** — *arXiv 2025* · [[paper]](https://arxiv.org/abs/2505.20353)
    - 核心贡献：联合空间显著 token 选择与 Transformer 级缓存，并用可学习线性近似和统计决策规则压缩 DiT 隐状态计算。

11. **DiCache: Let Diffusion Model Determine Its Own Cache** — *ICLR 2026* · [[paper]](https://arxiv.org/abs/2508.17356)
    - 核心贡献：利用浅层在线探针同时推断当前样本应在何时缓存以及如何组合多步缓存，使缓存策略由模型自身的实时特征轨迹驱动。

12. **HiCache: Training-free Acceleration of Diffusion Models via Hermite Polynomial-based Feature Caching** — *arXiv 2025* · [[paper]](https://arxiv.org/abs/2508.16984)
    - 核心贡献：根据 DiT 特征导数近似呈高斯特性的观察，以 Hermite 多项式和双缩放机制稳定预测未来特征。

13. **ERTACache: Error Rectification and Timesteps Adjustment for Efficient Diffusion** — *arXiv 2025* · [[paper]](https://arxiv.org/abs/2508.21091)
    - 核心贡献：将缓存误差拆为特征偏移与步长放大两部分，再通过离线残差分析、积分区间调整和闭式误差近似联合校正。

14. **CorGi: Contribution-Guided Block-Wise Interval Caching for Training-Free Acceleration of Diffusion Transformers** — *arXiv 2025* · [[paper]](https://arxiv.org/abs/2512.24195)
    - 核心贡献：CorGi 在区间内选择性复用低贡献 Transformer 块，并以跨注意力显著性触发局部更新来保护文本到图像中的关键对象细节。

15. **Evolutionary Caching to Accelerate Your Off-the-Shelf Diffusion Model** — *ICLR 2026* · [[paper]](https://openreview.net/forum?id=z9MlGsQbzR)
    - 核心贡献：使用遗传算法为具体模型自动搜索缓存计划，使现成扩散模型无需修改权重即可获得模型适配的质量—速度折中。

16. **Forecast the Principal, Stabilize the Residual: Subspace-Aware Feature Caching for Efficient Diffusion Transformers** — *CVPR 2026* · [[paper]](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_Forecast_the_Principal_Stabilize_the_Residual_Subspace-Aware_Feature_Caching_for_CVPR_2026_paper.html)
    - 核心贡献：SVD-Cache 将特征分解为平滑可预测的主子空间与振荡残差，以 EMA 预测前者并直接复用后者实现高倍率近无损缓存。

17. **OmniCache: Multidimensional Hierarchical Feature Caching For Diffusion Models** — *arXiv 2026* · [[paper]](https://arxiv.org/abs/2607.23844)
    - 核心贡献：以 Token、Frame、Block 和 Layered Cache 统一利用帧内、帧间、运动与跨去噪步四类冗余，并保持缓存特征的位置和时空顺序。

### 1.4 采样调度、误差校正与少步生成

1. **DOLLAR: Few-Step Video Generation via Distillation and Latent Reward Optimization** — *arXiv 2024* · [[paper]](https://arxiv.org/abs/2412.15689)
   - 核心贡献：结合变分分数蒸馏、一致性蒸馏与低显存潜空间奖励优化，将长视频生成压缩到一至数步并保持质量与多样性。

2. **Q-Sched: Pushing the Boundaries of Few-Step Diffusion Models with Quantization-Aware Scheduling** — *arXiv 2025 / ICLR 2026 Workshop* · [[paper]](https://arxiv.org/abs/2509.01624)
   - 核心贡献：在不改量化权重的情况下学习量化感知预条件系数，并以无需参考图的 JAQ 损失调整少步采样轨迹以抵消量化漂移。

3. **Absorbing Quantization Error by Deformable Noise Scheduler for Diffusion Models** — *ICML 2026* · [[paper]](https://openreview.net/forum?id=KUXeiU7eNR)
   - 核心贡献：将量化误差解释为扩散时间偏移，通过分布校准噪声补偿和可变形噪声调度器在不增加推理步骤的情况下保持目标分布。

4. **Multi-Resolution Flow Matching: Training-Free Diffusion Acceleration via Staged Sampling** — *arXiv 2026* · [[paper]](https://arxiv.org/abs/2607.01642)
   - 核心贡献：MrFlow 先以低分辨率快速确定结构，再经像素空间超分、低强度加噪和高分辨率细化实现无需训练的粗到细流匹配加速。

5. **DRiffusion: Draft-and-Refine Process Parallelizes Diffusion Models with Ease** — *arXiv 2026* · [[paper]](https://arxiv.org/abs/2603.25872)
   - 核心贡献：通过跳跃转移并行生成多个未来时间步草稿，再以标准去噪过程校正，使原本串行的扩散采样可跨设备并行。

### 1.5 动态架构、端侧部署与系统并行

1. **DistriFusion: Distributed Parallel Inference for High-Resolution Diffusion Models** — *CVPR 2024* · [[paper]](https://arxiv.org/abs/2402.19481)
   - 核心贡献：以 displaced patch parallelism 在多 GPU 间切分高分辨率输入，并复用上一时间步上下文来异步隐藏跨卡通信。

2. **PipeFusion: Patch-level Pipeline Parallelism for Diffusion Transformers Inference** — *arXiv 2024* · [[paper]](https://arxiv.org/abs/2405.14430)
   - 核心贡献：同时划分图像 patch 和模型层构造 patch 级流水线，并以一步陈旧特征补足上下文，从而降低 DiT 多卡推理通信和单卡显存。

3. **xDiT: an Inference Engine for Diffusion Transformers with Massive Parallelism** — *arXiv 2024* · [[paper]](https://arxiv.org/abs/2411.01738)
   - 核心贡献：将序列并行、PipeFusion 和 CFG 并行灵活组合成统一 DiT 推理引擎，使图像与视频模型可扩展到 NVLink 或以太网 GPU 集群。

4. **Taming Diffusion Transformer for Efficient Mobile Video Generation in Seconds** — *arXiv 2025* · [[paper]](https://arxiv.org/abs/2507.13343)
   - 核心贡献：联合高压缩 VAE、知识蒸馏引导的三级剪枝和四步对抗蒸馏，使视频 DiT 能在 iPhone 16 Pro Max 上达到约 15 FPS。

5. **Elastic Diffusion Transformer** — *arXiv 2026* · [[paper]](https://arxiv.org/abs/2602.13993)
   - 核心贡献：E-DiT 用轻量路由器按样本动态决定跳过整个块或缩减 MLP 宽度，并结合路由结果做块级缓存以提供弹性计算。

6. **Sol Video Inference Engine: Agent-Native Full-Stack Acceleration Framework for Efficient Video Generation** — *arXiv 2026* · [[paper]](https://arxiv.org/abs/2606.23743)
   - 核心贡献：Sol 将缓存、稀疏注意力、token 剪枝、量化和算子融合组织成面向具体模型—硬件—服务配置自动组合的全栈视频推理优化流程。

-->

## 2. World Model（WM）

### 2.1 量化与压缩分析

1. **An Empirical Study of World Model Quantization** — *arXiv 2026* · [[paper]](https://arxiv.org/abs/2602.02110)
   - 核心贡献：以 DINO-WM 系统比较多种权重/激活 PTQ，揭示编码器与预测器的非对称敏感性、低比特长程 rollout 失稳及规划目标失配等世界模型特有问题。

### 2.2 推理缓存、长期记忆与自回归生成

1. **WorldCache: Accelerating World Models for Free via Heterogeneous Token Caching** — *arXiv 2026* · [[paper]](https://arxiv.org/abs/2603.06331)
   - 核心贡献：以曲率引导的异构 token 预测和优先关注混沌 token 的自适应跳步处理多模态 token 异质性与非平稳动态，实现无需训练的扩散世界模型加速。

2. **X-Cache: Cross-Chunk Block Caching for Few-Step Autoregressive World Models Inference** — *arXiv 2026* · [[paper]](https://arxiv.org/abs/2604.20289)
   - 核心贡献：把缓存轴从去噪时间步改为相邻生成 chunk，以结构/动作感知门控决定逐块复用，并在 KV 写入 chunk 强制完整计算以截断误差传播。

3. **WorldKV: Efficient World Memory with World Retrieval and Compression** — *arXiv 2026* · [[paper]](https://arxiv.org/abs/2605.22718)
   - 核心贡献：根据相机位姿或动作检索历史 KV chunk，并以键相似度保留新颖 token，从而在有限 KV 预算下兼顾实时吞吐和重访视角的一致性。

4. **WorldDynCache: Risk-Controlled Latent Dynamics Approximation for Diffusion World Model** — *arXiv 2026* · [[paper]](https://arxiv.org/abs/2608.01845)
   - 核心贡献：用潜在转移风险估计器累计监控近似缺陷，并以条件和阶段感知的 lifted latent surrogate 预测演化，在 HunyuanVoyager 与 Aether 上实现受控高倍率缓存。

## 3. World-Action Model（WAM）与高效机器人扩散策略

> 本节优先收录严格意义上的 WAM；由于 WAM 尚属新兴命名，量化、缓存和 runtime 论文常以 VLA 或 Diffusion Policy 名义发表，因此用 `[WAM]`、`[VLA]`、`[Diffusion Policy]` 标明适用范围。

### 3.1 缓存与跨步复用

1. **Efficient Diffusion Transformer Policies with Mixture of Expert Denoisers for Multitask Learning** — *ICLR 2025 · Diffusion Policy* · [[paper]](https://openreview.net/forum?id=nDmwloEl3N)
   - 核心贡献：MoDE 使用噪声条件路由的稀疏去噪专家和专家缓存，在多任务扩散策略中减少活跃参数与约 90% 推理计算量。

2. **VLA-Cache: Towards Efficient Vision-Language-Action Model via Adaptive Token Caching in Robotic Manipulation** — *arXiv 2025 · VLA* · [[paper]](https://arxiv.org/abs/2502.02175)
   - 核心贡献：根据相邻观测的视觉变化自适应复用静态 token 的 KV，并按层注意力集中度重新计算任务关键 token，在实机控制中提高推理与控制频率。

3. **CDP: Towards Robust Autoregressive Visuomotor Policy Learning via Causal Diffusion** — *CoRL 2025 · Diffusion Policy* · [[paper]](https://openreview.net/forum?id=FwLMCbs47K)
   - 核心贡献：CDP 用历史动作条件化因果扩散策略，并缓存先前时间步的注意力 KV，以减少闭环自回归执行中的重复计算。

4. **EfficientVLA: Training-Free Acceleration and Compression for Vision-Language-Action Models** — *NeurIPS 2025 · VLA* · [[paper]](https://arxiv.org/abs/2506.10100)
   - 核心贡献：联合语言层剪枝、任务感知视觉 token 选择和扩散动作头的跨去噪步特征缓存，以无需训练的方式端到端压缩 VLA 推理。

5. **Faster-WAM: Efficient Inference-Time Future Conditioning for Robust World Action Models** — *arXiv 2026 · WAM* · [[paper]](https://arxiv.org/abs/2608.04404)
   - 核心贡献：以 SparseMoT 只在少数阶段融合视频与动作，并用 Interval KV-Fusion 复用多深度未来表征，在保留分布外鲁棒性的同时较 Joint-WAM 加速 2.21 倍。

6. **Neural Introspection Gating for Adaptive KV-Cache Reuse in Vision-Language-Action Models** — *arXiv 2026 · VLA* · [[paper]](https://arxiv.org/abs/2608.10824)
   - 核心贡献：Gated VLA-Cache 用动作 token 的 top-2 logit margin 作为零开销置信度信号，在模型不确定时使视觉 KV 缓存失效并触发完整重算。

### 3.2 量化与低比特部署

1. **BitVLA: 1-bit Vision-Language-Action Models for Robotics Manipulation** — *arXiv 2025 · VLA* · [[paper]](https://arxiv.org/abs/2506.07530)
   - 核心贡献：将 VLA 主体训练为三值 1-bit 参数并以蒸馏感知方式压缩视觉编码器，在接近 OpenVLA-OFT 4-bit 性能时仅使用约 29.8% 内存。

2. **QVLA: Not All Channels Are Equal in Vision-Language-Action Model’s Quantization** — *ICLR 2026 · VLA* · [[paper]](https://openreview.net/forum?id=TpL2nXanru)
   - 核心贡献：通过动作感知的通道重要性识别和非均匀量化保护控制关键通道，以 29.2% 原始显存保持 98.9% 性能并获得 1.49 倍加速。

3. **HBVLA: Pushing 1-Bit Post-Training Quantization for Vision-Language-Action Models** — *arXiv 2026 · VLA* · [[paper]](https://arxiv.org/abs/2602.13710)
   - 核心贡献：以策略感知 Hessian 识别动作关键权重，再对其余权重做稀疏正交变换和分组 1-bit PTQ，降低长程闭环中的量化误差累积。

4. **QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models** — *arXiv 2026 · VLA/DiT Action Head* · [[paper]](https://arxiv.org/abs/2602.20309)
   - 核心贡献：选择性整数化语言骨干与 DiT 动作头，并以注意力温度匹配和输出头平衡校准接口漂移，首次系统量化扩散式动作头。

### 3.3 异步执行、动作衔接与控制对齐

1. **VLASH: Real-Time VLAs via Future-State-Aware Asynchronous Inference** — *arXiv 2025 · VLA* · [[paper]](https://arxiv.org/abs/2512.01031)
   - 核心贡献：在执行上一动作 chunk 的同时推理下一 chunk，并用已承诺动作外推未来执行时刻状态，从而修正预测—执行时间错位且不增加模型开销。

2. **Xiaomi-Robotics-0: An Open-Sourced Vision-Language-Action Model with Real-Time Execution** — *arXiv 2026 · VLA* · [[paper]](https://arxiv.org/abs/2602.12684)
   - 核心贡献：通过面向异步执行的后训练与连续动作 chunk 时间戳对齐，使 VLA 在消费级 GPU 上实现平滑、无停顿的双臂实机 rollout。

3. **AsyncVLA: An Asynchronous VLA for Fast and Robust Navigation on the Edge** — *arXiv 2026 · VLA* · [[paper]](https://arxiv.org/abs/2602.13476)
   - 核心贡献：将远端大模型的低频语义推理与端侧轻量适配器的高频反应控制解耦，并以联合微调和轨迹重加权适应长通信延迟。

4. **AHA-WAM: Asynchronous Horizon-Adaptive World-Action Modeling with Observation-Guided Context Routing** — *arXiv 2026 · WAM* · [[paper]](https://arxiv.org/abs/2606.09811)
   - 核心贡献：让低频视频 DiT 以滚动 KV 记忆规划长程世界上下文、高频动作 DiT 闭环查询该上下文，并以观测路由避免每次动作更新都重跑视频分支。

5. **World Action Models in Real Time: An Empirical Study of Smooth Execution via Asynchronous Deployment** — *arXiv 2026 · WAM* · [[paper]](https://arxiv.org/abs/2608.01880)
   - 核心贡献：系统比较六种 WAM 同步/异步执行与动作融合方案，指出观测—预测—执行的精确时间对齐是避免 chunk 边界不连续的关键，并验证前缀条件生成最均衡。

### 3.4 轻量架构、未来分支裁剪与弹性计算

1. **Fast-WAM: Do World Action Models Need Test-time Future Imagination?** — *arXiv 2026 · WAM* · [[paper]](https://arxiv.org/abs/2603.16666)
   - 核心贡献：Fast-WAM 保留未来视频协同训练但在部署时取消显式未来生成，表明世界建模的主要收益可在训练期获得并把推理延迟降至约 190 ms。

2. **Light-WAM: Efficient World Action Models with State-Fusion Action Decoding** — *arXiv 2026 · WAM* · [[paper]](https://arxiv.org/abs/2606.08242)
   - 核心贡献：以紧凑视频骨干、低分辨率未来监督和一次前向的多层状态融合动作头取代重型生成式动作专家，实现 0.44B 参数和约 72 ms 推理。

3. **Efficient-WAM: A 1B-Parameter World-Action Model with Low-Cost Future Imagination** — *arXiv 2026 · WAM* · [[paper]](https://arxiv.org/abs/2606.10040)
   - 核心贡献：通过迁移紧凑视频专家、稀疏视频 latent 和视频/动作非对称去噪步数，把粗粒度未来作为动作指导信号并将每个动作 chunk 延迟降至约 100 ms。

4. **WAM4D: Fast 4D World Action Model via Spatial Register Tokens** — *arXiv 2026 · WAM* · [[paper]](https://arxiv.org/abs/2606.14048)
   - 核心贡献：训练时用可移除的空间 register token 将 4D 几何先验蒸馏进因果视频—动作 Transformer，部署时无需密集几何解码即可提高空间一致性。

5. **Learning 4D Geometric Priors for Inference-Efficient World Action Models** — *arXiv 2026 · WAM* · [[paper]](https://arxiv.org/abs/2607.05468)
   - 核心贡献：MECo-WAM 通过训练期 4D 专家、衰减读掩码和动作感知几何蒸馏注入动态几何知识，推理时移除辅助分支而不增加计算图。

6. **Faster-WAM: Do World Action Models Need Deep Action Modules?** — *arXiv 2026 · WAM* · [[paper]](https://arxiv.org/abs/2608.02365)
   - 核心贡献：基于 Dock of Transformer 将单层动作头接入 30 层视频骨干并聚合各层 KV，在保持动作性能的同时把端到端延迟降至 66.5 ms。

7. **SimWAM: A Simple World Action Model for End-to-End Autonomous Driving** — *arXiv 2026 · WAM* · [[paper]](https://arxiv.org/abs/2608.07468)
   - 核心贡献：以联合流匹配协同训练视频专家与轻量动作专家，并用隔离注意力让部署时可完全丢弃视频分支，从而直接低延迟预测驾驶轨迹。

8. **Flex-π: A Multi-Stream World-Action Model with Compute Flexibility** — *arXiv 2026 · WAM* · [[paper]](https://arxiv.org/abs/2608.10860)
   - 核心贡献：在共享潜空间联合去噪 RGB、3D 点图、DINO 语义与动作，并通过逐流 dropout 使同一检查点可在快速 action-only 与完整联合生成模式间弹性切换。

### 3.5 推理系统、runtime 与端侧部署

1. **LiteVLA-Edge: Quantized On-Device Multimodal Control for Embedded Robotics** — *arXiv 2026 · VLA/System* · [[paper]](https://arxiv.org/abs/2603.03380)
   - 核心贡献：以 4-bit GGUF、llama.cpp GPU 加速和 ROS 2 感知—推理—执行接口构建完全离线的 Jetson Orin VLA 部署流水线。

2. **vla.cpp: A Unified Inference Runtime for Vision-Language-Action Models** — *arXiv 2026 · VLA/System* · [[paper]](https://arxiv.org/abs/2606.08094)
   - 核心贡献：基于 llama.cpp/ggml 提供统一 C++ runtime，在同一协议和模型包格式下原生支持七类自回归、flow-matching 与扩散 VLA，并跨消费级 GPU、CPU 和嵌入式设备运行。

3. **Embodied.cpp: A Portable Inference Runtime of Embodied AI Models on Heterogeneous Robots** — *arXiv 2026 · VLA/WAM/System* · [[paper]](https://arxiv.org/abs/2607.02501) [[code]](https://github.com/SEU-PAISys/Embodied.cpp)
   - 核心贡献：将 VLA/WAM 的共享执行路径抽象为输入适配、序列构建、骨干执行、头插件与部署适配五层，以统一 C++ runtime 支持异构硬件上的多速率闭环执行和低延迟 batch-1 推理。

4. **Jetson-PI: Towards Onboard Real-Time Robot Control via Foresight-Aligned Asynchronous Inference** — *arXiv 2026 · VLA/System* · [[paper]](https://arxiv.org/abs/2607.12659)
   - 核心贡献：联合未来表征校正、置信度调度、CUDA Graph 复用、GPU 常驻缓冲与 flow unrolling，在 Jetson Orin 上同时处理异步错位和底层执行效率。

5. **PhyAI: Real-Time Physical AI at the Edge, Scalable Rollouts in the Cloud** — *arXiv 2026 · VLA/WAM/System* · [[paper]](https://arxiv.org/abs/2608.03682) [[code]](https://github.com/mingti-org/phyai)
   - 核心贡献：以模型适配器封装架构特有的条件、求解器、缓存和输出逻辑，同时共享图执行、内核、内存管理与并行服务，从而用同一 runtime 支撑 VLA/WAM 的端侧实时推理和云端多 GPU rollout。

## 4. 分类边界与阅读建议

- **量化**主要降低权重、激活或注意力的位宽，实际延迟收益依赖目标 GPU/NPU 是否提供对应 INT/FP 内核。
- **稀疏与注意力压缩**主要降低单次 DiT 前向成本，长视频和高分辨率任务通常受益最大，但结构化稀疏是否真正加速取决于内核实现。
- **缓存**利用相邻去噪步、层、token、生成 chunk 或历史 KV 的冗余，训练成本低，但需要特别关注误差累积和少步模型中冗余不足的问题。
- **少步/调度**减少网络函数评估次数，通常与量化、缓存和并行正交，但蒸馏方法需要额外训练且可能改变模型能力边界。
- **系统并行**不一定减少总 FLOPs，而是通过多卡切分、流水化、通信隐藏和内核融合降低墙钟延迟。
- **WM/WAM 专用优化**除生成质量外还必须评估长期 rollout、一致性、规划成功率、闭环控制频率和动作鲁棒性，不能只用 FID/VBench 判断效果。
