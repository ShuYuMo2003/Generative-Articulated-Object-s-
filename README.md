# TODO.
 - [x] 处理所有测试数据，准备 ONet 数据集，做全部 ONet 训练
 - [x] 修改提示词，使得每个物体对应多个描述语。
    - [x] 考虑对同一个物体的不同 Joint State，不同角度，生成不同的描述语句。
    - [x] 可视化不同角度的物体吗，可以从原先数据集开始，也可以从最后的整体数据集开始。
 - [x] Tree Position Embedding.
 - [x] 验证薄片 mesh part 的 sdf 采样情况
 - [ ] 考虑特殊 Token 的赋值情况，需不需要根据每个字段的分布生成。
 - [ ] matrice 测试。
 - [x] 调试 sdf 监督 O Net 训练。
 - [x] 原训练数据集高质量渲染。
 - [x] 尝试使用 Vision (CNN / ViT)类预训练大模型做 Encoder。
    - [ ] vit, Swin Transformer, https://arxiv.org/abs/2103.14030
    - [x] vit, BLIP, image to text 模型, 可以取用训练完成的 Encoder, https://arxiv.org/abs/2201.12086
    - [ ] vit, nlpconnect/vit-gpt2-image-captioning https://huggingface.co/nlpconnect/vit-gpt2-image-captioning
    - [ ] vit, LRM, LARGE RECONSTRUCTION MODEL FOR SINGLE IMAGE TO 3D, https://yiconghong.me/LRM/


# Notes

GenSDF:
 - stilted-puddle-76, kl_weight=0.01, latent_dim=256, 测试 latent_dim 的下界，transformer 测试表明 kl_weight=1e-3的时候影响生成质量不错，kl_weight 可以考虑放宽。
 - true-sky-74, kl_weight=0.1, latent_dim=256, 中间效果不好，中止。
 - golden-cosmos-59, kl_weight=1e-3, latent_dim=768

Transformer:
 - bumbling-dust-26, image condition 首次测试，加入了 Post Encoder。Loss 表现优于 zesty-cloud-24
 - zesty-cloud-24, text condition, with golden-cosmos-59(latent_dim=768). 中止，训练不完全，怀疑 latent_dim 过大，且 GenSDF 训练时kl_weight=1e-3.