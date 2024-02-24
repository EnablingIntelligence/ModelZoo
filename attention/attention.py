import math
from enum import Enum
from typing import Optional

import torch
from torch import nn
from torch.nn.functional import softmax


class AttentionType(Enum):
    NON_CAUSAL = 1
    CAUSAL = 2


class SelfAttention(nn.Module):
    """
    Transformer Self-Attention
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = .0):
        """
        :param input_dim: dimension of input
        :param hidden_dim: dimension of attention hidden layer
        :param dropout: probability of dropout
        """
        super(SelfAttention, self).__init__()
        self.q = nn.Linear(input_dim, hidden_dim)
        self.k = nn.Linear(input_dim, hidden_dim)
        self.v = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculates the transformer self-attention of the input. Optionally, a mask can be provided to mask out
        elements of the input sequence.
        :param x: input tensor
        :param mask: mask tensor applied on the scores (values of 0 are masked out)
        :return: attention tensor
        """
        query = self.q(x)
        key = self.k(x)
        value = self.v(x)

        scores = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        weights = softmax(scores, dim=-1)
        return torch.bmm(self.dropout(weights), value)


class CausalSelfAttention(nn.Module):
    """
    Transformer self-attention with causal mask
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = .0):
        """
        :param input_dim: dimension of input
        :param hidden_dim: dimension of attention hidden layer
        :param dropout: probability of dropout
        """
        super(CausalSelfAttention, self).__init__()
        self.attention = SelfAttention(input_dim, hidden_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the causal transformer self-attention of the input. All consequent elements are masked out.
        :param x: input tensor
        :return: attention tensor
        """
        batch_size = x.size(0)
        seq_len = x.size(1)
        causal_mask = torch.tril(torch.ones((batch_size, seq_len, seq_len), device=x.device))
        return self.attention(x, causal_mask)


class MultiHeadAttention(nn.Module):
    """
    Multi-head transformer self-attention
    """

    def __init__(self, input_dim: int, num_heads: int, dropout: float = .0, head_dropout: float = .0,
                 attention_type: AttentionType = AttentionType.NON_CAUSAL):
        """
        :param input_dim: dimension of input
        :param num_heads: number of attention heads
        :param dropout: probability of dropout of combined attention
        :param head_dropout: probability of dropout for each head
        :param attention_type: attention type (causal or non-causal)
        """
        super(MultiHeadAttention, self).__init__()
        head_dim = input_dim // num_heads
        attention = CausalSelfAttention if attention_type == AttentionType.CAUSAL else SelfAttention
        self.heads = nn.ModuleList(
            [attention(input_dim, head_dim, head_dropout) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the multi-head transformer self-attention of the input.
        :param x: input tensor
        :return: multi-head attention tensor
        """
        x = torch.cat([head(x) for head in self.heads], dim=-1)
        x = self.linear(x)
        return self.dropout(x)
