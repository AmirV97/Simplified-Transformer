import torch
import torch.nn as nn
from simplified_transformer.block import SimplifiedTransformerBlock
import yaml

class S_GPTModel(nn.Module):
    def __init__(self, config_path="config.yaml"):
        super().__init__()
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        self.token_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.FF = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 2 * cfg["emb_dim"] * cfg["n_layers"]),
            nn.GELU(approximate='tanh'),
            nn.Linear(2 * cfg["emb_dim"] * cfg["n_layers"], cfg["emb_dim"]),
        )

        self.blocks = nn.Sequential(*[
            SimplifiedTransformerBlock(
                d_model=cfg["emb_dim"],
                context_length=cfg["context_length"],
                dropout_p=cfg["drop_rate"],
                num_heads=cfg["n_heads"],
                FF=self.FF
            ) for _ in range(cfg["n_layers"])
        ])

        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        N, L = in_idx.shape
        tok_embeds = self.token_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(L, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.blocks(x)
        logits = self.out_head(x)
        return logits