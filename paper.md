TransArticulate: Conditional Generation of Articulated Objects Using Transformer Models

highlight:
 - Use any customize encoder to encode the condition, and generate corresponding articulated object.
 - Novel tree structure generation approach based on transformer model.
    - Tree Position Embedding.
    - Parent-to-children encoding.

Experiments:
主要两个应用，从文字描述生成，从图像生成（重建）
CAGE 是 NAP + conditional generation 版本，参考 CAGE 的实验部分是怎么和 NAP 做对比的。
 - （从图像生成）vision condition for CAGE and NAP, compare with our model.
    - Human Search.
    - 参考 Diffusion-SDF: Conditional Generative Modeling of Signed Distance Functions
 - text condition for CAGE and NAP, compare with our model.
    - Human Search.
    - 参考：DREAMFUSION: TEXT-TO-3D USING 2D DIFFUSION

Ablations Study：
 - Tree Position Embedding.
 - Long Skip Connection.
 - Condition Encoder 的规模。
 - Vision Encoder 的规模。