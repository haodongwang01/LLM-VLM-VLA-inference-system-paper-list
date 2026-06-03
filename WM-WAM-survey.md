 - **WorldKV: Efficient World Memory with World Retrieval and Compression**
(ArXiv 2026.05) [[paper]](https://arxiv.org/abs/2605.22718) WorldKV 提出了一个无需训练的框架，通过 World Retrieval（根据相机位姿或动作序列检索相关的历史 KV 缓存块）和 World Compression（利用键-键相似度剪枝冗余令牌）两大核心技术，在不增加计算负担的情况下实现了视频扩散模型在长期探索中的视点一致性。 

 - **X-cache：Cross-Chunk Block Caching for Few-Step Autoregressive World Models Inference**
(ArXiv 2026.04) [[paper]](https://arxiv.org/abs/2604.20289) X-Cache 创新性地通过在**跨生成块（Cross-Chunk）**而非跨去噪步维度缓存 DiT 模块残差，利用物理世界的时空连续性实现了少步数自回归世界模型的高效推理加速。

 - **Timestep Embedding Tells: It's Time to Cache for Video Diffusion Model**
(CVPR 2025) [[paper]](https://arxiv.org/abs/2411.19108) TeaCache 是一种无需训练的视频扩散模型加速方案，它创新性地利用时间步嵌入调制的噪声输入（Timestep Embedding Modulated Noisy Input）作为廉价指标来动态估计模型输出的差异，并据此实现自适应的特征缓存与复用，在几乎不损失视频质量的前提下大幅提升推理效率。

- **WorldCache: Accelerating World Models for Free via Heterogeneous Token Caching**
(ICML 2026) [[paper]](https://arxiv.org/abs/2603.06331) WorldCache 通过引入基于物理曲率的异构标记预测 (CHTP) 和以混沌标记为优先的自适应跳步策略 (CAS)，在无需训练的前提下有效解决了扩散世界模型中多模态标记异构性与时间动态非平稳带来的加速瓶颈。 

