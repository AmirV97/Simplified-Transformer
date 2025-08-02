# Efficient Transformer Architectures

This repository contains implementations for the [Simplified Transformer](http://arxiv.org/abs/2311.01906) block , as well as the architecture using a [single wide MLP](http://arxiv.org/abs/2309.01826) for all transformer blocks.
I intend to add further implementations such as multihead latent attention (MLA) as well as empirical results on small datasets in the future. Obviously, improving transformer architecture efficiency is very wide and deep field of research.

## Simplified Transformer Blocks

This paper suggests major prunings to the transformer architecture, resulting in fewer parameters while maintaining performance. Specifically,

* Instead of three projections (Q,K,V), only two are used (Q,K). The scaled dot-product attention operation is performed without multiplying by V at the end: 
$$\text{scores}=\text{softmax} \left( \frac{QK^T}{\sqrt{ d }}  \right)$$

The final attention output is a weighted sum of the scores as well as two fixed matrices (known as **shaped attention**):
$$A = \alpha I+\beta \times \text{scores} + \gamma C$$
Where $\alpha,\beta$ and $\gamma$ are learned parameters, and $C$ is a *centering matrix*, equivalent to *uniform* attention over all the tokens in a sequence. The scalar parameters $\alpha,\beta,\gamma$ are initialized with values of $1$ each.

Finally, the shaped attention operation and the MLP are not arranged *sequentially*, but rather in parallel:
$$\text{output} = w_{\text{FF}}\text{MLP}(x)+ w_{A}\text{shaped attention}(x)$$
Where $w_\text{FF},w_{\text{A}}$ are learnable weights for the feed-forward network and the shaped attention operation, respectively.

## One Wide Feed-forward

The main idea of this paper is to use a shared MLP for *all* the transformer blocks in a model. Typically, the hidden layer in the MLP has a width of `4*embedding_dim`. The authors suggest using a shared MLP with a hidden layer of width `4 * embedding_dim * n_layers`, which maintains performance, while resulting in minimal parameter count reduction. The highlight result is Paretto-optimal parameter reductions when using $\sqrt{ 4 \times \text{embedding dim} \times \text{num layers} }$, resulting in minimal performance loss but also significant parameter count reduction.
