# Efficient Transformer Architectures

This repository contains implementations for the [Simplified Transformer](http://arxiv.org/abs/2311.01906) block , as well as the architecture using a [single wide MLP](http://arxiv.org/abs/2309.01826) for all transformer blocks.
I intend to add further implementations such as multihead latent attention (MLA) as well as empirical results on small datasets in the future.

# Simplified Transformer Blocks
This paper suggests major prunings to the transformer architecture, resulting in fewer parameters while maintaining performance. Specifically,

* Instead of three projections (Q,K,V), only two are used (Q,K). The scaled dot-product attention operation is performed without multiplying by V at the end:
$$
scores = QK
$$