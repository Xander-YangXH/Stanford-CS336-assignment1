"""
Transformer Block（第二阶段·第三部分）：
把第一阶段的 RMSNorm、SwiGLU 和第二阶段的 MultiHeadSelfAttention 拼成一个
完整的 pre-norm Transformer block。

结构（pre-norm，跟原始 Transformer 论文的 post-norm 不同，训练更稳定）：
    x = x + MultiHeadSelfAttention(RMSNorm_1(x))
    x = x + SwiGLU(RMSNorm_2(x))

子模块命名（q_proj/k_proj/v_proj/output_proj、ln1/ln2、ffn.w1/w2/w3）故意跟
作业参考实现的 state_dict key 完全对齐，这样可以直接用
`transformer_block.load_state_dict(weights)` 把参考权重灌进来做数值对拍，
不用再手写一堆 `xxx.weight.data = ...` 的赋值。
"""

import torch
import torch.nn as nn

from cs336_basics.attention_core import MultiHeadSelfAttention
from cs336_basics.nn_utils import RMSNorm, SwiGLU


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            use_rope=True,
            max_seq_len=max_seq_len,
            theta=theta,
            device=device,
            dtype=dtype,
        )
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        token_positions: (batch, seq_len) 可选，不传就默认 0..seq_len-1。
        返回: (batch, seq_len, d_model)，形状不变——block 只做特征变换，不改变形状。

        两个残差连接都是"先归一化、再变换、再加回原始输入"（pre-norm），
        而不是"变换完再归一化"，这样残差路径上是干净的加法，梯度更容易
        传到浅层，训练大模型时明显更稳定。
        """
        x = x + self.attn(self.ln1(x), token_positions=token_positions)
        x = x + self.ffn(self.ln2(x))
        return x
