import torch
import torch.nn as nn
import torch.nn.functional as F


class CodebookAttention(nn.Module):
    def __init__(self,
        codebook_size: int,
        embed_dim: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        self.codebook_size = codebook_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.keys = nn.Parameter(torch.randn(self.codebook_size, embed_dim))
        self.codes = nn.Parameter(torch.randn(self.codebook_size, embed_dim))
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads,
            bias=False,
            batch_first=True
        )
    
    def forward(self, query, need_weights=False):
        B = query.size(0)
        return self.attn(
            query, 
            self.keys.unsqueeze(0).expand(B, -1, -1),
            self.codes.unsqueeze(0).expand(B, -1, -1),
            need_weights=need_weights,
            average_attn_weights=False
        )
