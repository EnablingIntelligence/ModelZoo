import math
from enum import Enum
from typing import Optional

import torch
import torch.nn as nn
from torch.nn.functional import softmax


class AttentionType(Enum):
    NON_CAUSAL = 1
    CAUSAL = 2


class SelfAttention(nn.Module):

    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float = .0):
        super(SelfAttention, self).__init__()
        self.q = nn.Linear(embed_dim, hidden_dim)
        self.k = nn.Linear(embed_dim, hidden_dim)
        self.v = nn.Linear(embed_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        query = self.q(x)
        key = self.k(x)
        value = self.v(x)

        scores = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        weights = softmax(scores, dim=-1)
        return torch.bmm(self.dropout(weights), value)


class CausalSelfAttention(nn.Module):

    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float = .0):
        super(CausalSelfAttention, self).__init__()
        self.attention = SelfAttention(embed_dim, hidden_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        seq_len = x.size(1)
        causal_mask = torch.tril(torch.ones((batch_size, seq_len, seq_len), device=x.device))
        return self.attention(x, causal_mask)


class MultiHeadAttention(nn.Module):

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = .0, head_dropout: float = .0,
                 attention_type: AttentionType = AttentionType.NON_CAUSAL):
        super(MultiHeadAttention, self).__init__()
        head_dim = embed_dim // num_heads
        attention = CausalSelfAttention if attention_type == AttentionType.CAUSAL else SelfAttention
        self.heads = nn.ModuleList(
            [attention(embed_dim, head_dim, head_dropout) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([head(x) for head in self.heads], dim=-1)
        x = self.linear(x)
        return self.dropout(x)
