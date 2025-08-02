import torch.nn as nn
import torch.nn.functional as F

class SimplifiedTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        num_heads: int = 8,
        dropout_p: float = 0.1,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model, self.num_heads = d_model, num_heads
        self.d_k = d_model // num_heads

        # Projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_QK = nn.Linear(d_model, 2*d_model)

        # Feed-forward
        self.FF = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * d_model, d_model),
        )

        # Shaped-attention scalars
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.gamma = nn.Parameter(torch.tensor(1.0))
        ## FF gain parameters
        self.alpha_ff = nn.Parameter(torch.tensor(1.0))
        self.beta_ff  = nn.Parameter(torch.tensor(0.1))

        # Dropout
        self.dropout = nn.Dropout(dropout_p)

        # Zero-init W_q so that initial attention = I
        nn.init.zeros_(self.W_q.weight)
        if self.W_q.bias is not None:
            nn.init.zeros_(self.W_q.bias)

    def _proj(self, x: torch.Tensor, W: nn.Linear) -> torch.Tensor:
        return W(x).view(x.size(0), x.size(1), 2*self.num_heads, self.d_k).permute(0, 2, 1, 3).chunk(dim=1, chunks=2)

    def _make_centering(self, N: int, L: int, mask:torch.Tensor = None) -> torch.Tensor:
        # uniform attention over length L
        if mask is None:
          C = torch.zeros(N, self.num_heads, L, L, device=self.alpha.device)
          return F.softmax(C, dim=-1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        x: [N, L, d_model]
        mask: optional additive mask of shape [N, H, L, L]
                   (e.g. -inf where you want to block)
        """
        N, L, _ = x.size()

        # Prepare identity & centering
        I = torch.eye(L, device=x.device).unsqueeze(0).unsqueeze(0)  # [1,1,L,L]
        I = I.expand(N, self.num_heads, L, L)                        # [N,H,L,L]
        C = self._make_centering(N, L)

        # Q & K projections

        Q, K = self._proj(x, self.W_QK) # [N, L, D] -> [N, L, 2*D] -> [N, L, 2*H, d_k] -> [N, 2*H, L, d_k]

        # Scaled dot-product attention without V
        scores = (Q @ K.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            scores = scores + mask
        A = F.softmax(scores, dim=-1)

        # Shaped attention
        A = self.alpha * I + self.dropout(self.beta * A) - self.gamma * C

        # Apply to values (here using x as V)
        out = A @ x.view(N, L, self.num_heads, self.d_k).permute(0, 2, 1, 3) # [N,H,L,d_k]
        out = out.permute(0, 2, 1, 3).reshape(N, L, self.d_model)  # concat heads

        return self.alpha_ff * out + self.beta_ff * self.FF(x)
