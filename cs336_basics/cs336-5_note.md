## CS336 作业一 —— 多头自注意力 + Transformer Block（MultiHeadSelfAttention / TransformerBlock）

日期：2026-09-04

---

## 一、知识点汇总（按作业文档的框架和公式）

### 1. Multi-Head Self-Attention（3.2.2）

```py
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
```

直觉：与其只用一组 Q/K/V 算一次 attention，不如把 `d_model` 切成 `num_heads` 份、每份 `d_k = d_model / num_heads` 维，让每个 head 各自独立地去关注序列里不同类型的模式（比如有的 head 关注语法结构，有的关注语义关联），最后把所有 head 的结果拼回 `d_model` 维，再过一次输出投影 `W^O` 把信息重新混合。

**工程要点（作业明确要求）**：不能真的写个 for 循环跑 `num_heads` 次小 attention，太慢。正确做法是先用**一次**大的矩阵乘法把 Q/K/V 投影到完整的 `d_model` 维，再把最后一维 reshape 成 `(num_heads, d_k)`，让"多头"变成一个额外的批量维度，一次性喂给已经写好的 `scaled_dot_product_attention`（复用第一部分的成果，"..." 通配前导维度这个设计这时候就体现出价值了）。

**权重排布**：`q_proj.weight` 形状 `(d_model, d_model)`，是把每个 head 的 `(d_k, d_model)` 权重按行拼接起来的（`q_proj.weight == cat([q_heads.0, ..., q_heads.N], dim=0)`），这跟"先整体投影再 reshape"的做法天然吻合，不需要特殊处理。

**Causal mask**：语言模型场景下，第 `i` 个 token 只能看到 `0..i`，用 `torch.tril` 生成下三角布尔矩阵，直接传给 `scaled_dot_product_attention` 的 `mask` 参数。

**RoPE 要作用在每个 head 的 `d_k` 维上，而不是完整的 `d_model` 维**——这点容易搞混，因为 RoPE 是"逐位置"旋转，跟"哪个 head"无关，所以旋转要在 split 成多头**之后**、算 attention **之前**做，且 `RotaryPositionalEmbedding` 初始化时 `d_k` 传的是 `head_dim`。另外还有一个广播上的坑：`token_positions` 形状是 `(..., seq_len)`，但 Q/K reshape 成多头后是 `(..., num_heads, seq_len, d_k)`，中间多插入了一个 head 维，如果不处理，`cos_cached[token_positions]` 查出来的表在 batch 有多个样本、head 数又大于 1 时会跟 Q/K 对不上号广播失败——需要先把 `token_positions` 在倒数第二维插入一个长度为 1 的维度（`unsqueeze(-2)`），让它排到 head 维的位置上，才能正确广播到每个 head。

### 2. Pre-Norm Transformer Block（3.5，Figure 2）

```py
x = x + MultiHeadSelfAttention(RMSNorm_1(x))
x = x + SwiGLU(RMSNorm_2(x))
```

跟原始 Transformer 论文的 post-norm（先算子层再归一化）不同，这里是 pre-norm：先归一化、再送进子层、最后把子层的输出**加回**没有被归一化过的原始输入。好处是残差路径上是一条干净的加法链，不会被归一化层打断，梯度能更顺畅地传回浅层，训练深层模型时明显更稳定，是目前主流 LLM（GPT、LLaMA 等）的标配写法。

两个残差连接分别包住 attention 子层和 FFN 子层，各自用独立的一份 RMSNorm（`ln1`、`ln2`），因为两个子层要归一化的分布不一样，不能共用一份参数。

---

## 三、代码详解

### 1. `cs336_basics/attention_core.py`：`MultiHeadSelfAttention`

```python
def _split_heads(self, x, seq_len):
    # (..., seq_len, d_model) -> (..., num_heads, seq_len, d_k)
    *batch, _ = x.shape[:-1]
    x = x.view(*batch, seq_len, self.num_heads, self.d_k)
    return x.transpose(-3, -2)
```

先把最后一维拆成 `(num_heads, d_k)`，这一步纯 reshape、不涉及数据搬动；再用 `transpose(-3, -2)` 把 `num_heads` 挪到 `seq_len` 前面，让它跟 batch 维排在一起，变成 `scaled_dot_product_attention` 眼里的又一个"前导维度"。`_merge_heads` 是它的逆操作，多了一步 `.contiguous()`，因为 `transpose` 之后张量在内存里不连续，后面的 `.view()` 要求内存连续。

```python
if self.use_rope:
    if token_positions is None:
        token_positions = torch.arange(seq_len, device=x.device)
    token_positions = token_positions.unsqueeze(-2)  # (..., 1, seq_len)
    Q = self.rope(Q, token_positions)
    K = self.rope(K, token_positions)
```

这里就是知识点里提到的广播坑的修复：`unsqueeze(-2)` 插入的这个长度为 1 的维度，会在 `cos_cached[token_positions]` 查表之后自动广播成 `num_heads`，从而正确对齐到 Q/K 的 `(..., num_heads, seq_len, d_k)` 形状。

```python
causal_mask = torch.tril(
    torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)
)
attn_out = scaled_dot_product_attention(Q, K, V, mask=causal_mask)
```

`torch.tril` 直接生成下三角矩阵，对角线及以下（含自身）是 `True`。因为 `scaled_dot_product_attention` 的 mask 语义是"`True` = 允许看到"，跟 `tril` 的语义正好一致，不用再取反。

**顺带做的一个重命名**：输出投影原来叫 `o_proj`，但作业参考实现的 state_dict key 是 `attn.output_proj.weight`。为了让后面 `TransformerBlock` 能直接 `load_state_dict(weights)` 整体加载参考权重（而不是像 `run_multihead_self_attention` adapter 里那样一个个字段手动赋值），把它改名成了 `output_proj`，跟参考实现的命名完全对齐。

### 2. `cs336_basics/transformer.py`（新文件）：`TransformerBlock`

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta, device=None, dtype=None):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(
            d_model=d_model, num_heads=num_heads, use_rope=True,
            max_seq_len=max_seq_len, theta=theta, device=device, dtype=dtype,
        )
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)

    def forward(self, x, token_positions=None):
        x = x + self.attn(self.ln1(x), token_positions=token_positions)
        x = x + self.ffn(self.ln2(x))
        return x
```

子模块命名（`ln1`/`attn.q_proj`/`attn.k_proj`/`attn.v_proj`/`attn.output_proj`/`ln2`/`ffn.w1`/`ffn.w2`/`ffn.w3`）故意跟作业参考实现的 state_dict key 一字不差地对齐，`forward` 里两行代码就是知识点里的两个残差公式的直接翻译，没有别的逻辑。

### 3. `tests/adapters.py`：接线

`run_multihead_self_attention` / `run_multihead_self_attention_with_rope` 分别构造一个不带/带 RoPE 的 `MultiHeadSelfAttention`，把参考权重手动塞进 `q_proj.weight.data` 等字段，再调用 forward。

`run_transformer_block` 因为子模块命名对齐了，直接：

```python
block = TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff,
                          max_seq_len=max_seq_len, theta=theta)
block.load_state_dict(weights)
return block(in_features)
```

一行 `load_state_dict` 顶替了一堆手动赋值，这也是这次特意做重命名的原因。

---

## 四、测试结果

`uv run pytest -k "test_multihead_self_attention or test_multihead_self_attention_with_rope or test_transformer_block" -v`：

```
collected 48 items / 45 deselected / 3 selected

tests/test_model.py::test_multihead_self_attention PASSED
tests/test_model.py::test_multihead_self_attention_with_rope PASSED
tests/test_model.py::test_transformer_block PASSED

3 passed, 45 deselected in 0.15s
```

**3 个全部通过**，`MultiHeadSelfAttention` 不带 RoPE、带 RoPE 两种情况，以及组装起来的 `TransformerBlock`（靠 `load_state_dict` 整体加载参考权重）都跟参考实现的输出对上了，中间提到的 `token_positions.unsqueeze(-2)` 广播修复和 `o_proj` → `output_proj` 的重命名也都没有引入问题。
