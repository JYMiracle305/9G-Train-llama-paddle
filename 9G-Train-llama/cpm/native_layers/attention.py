import math
from typing import Optional
from typing import Tuple

import paddle
from einops import rearrange

from .linear import Linear


class Attention(paddle.autograd.PyLayer):
    def __init__(
        self,
        dim_model: int,
        num_heads: int,
        num_kv_heads: int,
        dim_head: int,
        dtype: paddle.dtype = paddle.float16,
        dropout_p: Optional[float] = None,
        use_flash_attn: bool = False,
        scale: bool = True,
    ) -> None:
        super().__init__()

        self.dim_model = dim_model
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.num_kv_heads = num_kv_heads
        self.head_groups = num_heads // num_kv_heads

        self.project_q = Linear(self.dim_model, self.num_heads * self.dim_head, dtype=dtype, scale=scale)
        self.project_k = Linear(self.dim_model, self.num_kv_heads * self.dim_head, dtype=dtype, scale=scale)
        self.project_v = Linear(self.dim_model, self.num_kv_heads * self.dim_head, dtype=dtype, scale=scale)

        self.attention_out = Linear(self.num_heads * self.dim_head, self.dim_model, dtype=dtype, scale=scale)

        self.softmax = paddle.nn.Softmax(dim=-1)

        if dropout_p is not None:
            self.dropout = paddle.nn.Dropout(p=dropout_p)
        else:
            self.dropout = None

        # if use_flash_attn:
        #     self.core_attention_flash = FlashSelfAttention(causal=False, attention_dropout=0.0)
        # self.use_flash_attn = use_flash_attn

    def forward(
        self,
        hidden_q: paddle.Tensor,
        hidden_kv: paddle.Tensor,
        attention_mask: paddle.Tensor,
        position_bias: paddle.Tensor,
        use_cache: bool = False,
        past_kv: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None,
        pos_bias_type: Optional[str] = "relative",
        length_mask: Optional[paddle.Tensor] = None,
        context_mask: Optional[paddle.Tensor] = None,
    ):
        """
        Args:
            hidden_q (:obj:`paddle.Tensor` of shape ``(batch, len_q, dim_model)``): Indices of input sequence tokens. It will be embedded by model's internal embedding lookup matrix.
            hidden_kv (:obj:`paddle.Tensor` of shape ``(batch, len_k, dim_model)``): Length of input sequence before padding.
            attention_mask (:obj:`paddle.Tensor` of shape ``(batch, len_q, len_k)``): Used to avoid performing attention on padding token indices.
            position_bias(:obj:`paddle.Tensor` of shape ``(num_heads, len_q, len_k)`` or ``(1, num_heads, len_k, len_q)``): Provide positional information about tensor `key_value` and `query`.
        Return:
            out (:obj:`paddle.Tensor` of shape ``(batch, len_q, dim_model)``): The attention output.
        """  # noqa: E501

        batch_size = hidden_q.size(0)
        len_q = hidden_q.size(1)
        len_k = hidden_kv.size(1)

        h_q = self.project_q(hidden_q)
        h_k = self.project_k(hidden_kv)
        h_v = self.project_v(hidden_kv)

        h_q = h_q / math.sqrt(math.sqrt(self.dim_head))
        h_k = h_k / math.sqrt(math.sqrt(self.dim_head))

        h_q = h_q.view(batch_size, len_q, self.num_heads, self.dim_head).permute(0, 2, 1, 3)
        h_k = h_k.view(batch_size, len_k, self.num_kv_heads, self.dim_head).permute(0, 2, 1, 3)
        h_v = h_v.view(batch_size, len_k, self.num_kv_heads, self.dim_head).permute(0, 2, 1, 3)

        if pos_bias_type == "rotary":
            # b h s d
            h_q, h_k = position_bias(h_q, h_k, -2, offset=past_kv[0].size(-2) if past_kv is not None else 0)

        if past_kv is not None:
            h_k = paddle.cat([past_kv[0], h_k], dim=-2)
            h_v = paddle.cat([past_kv[1], h_v], dim=-2)
            len_k = h_k.size(-2)

            # (b, n_h, len_q, d_h) @ (b, n_h, d_h, len_k) -> (b, n_h, len_q, len_k)
        if self.head_groups == 1:
            score = paddle.matmul(h_q, h_k.transpose(-1, -2))  # / math.sqrt(self.dim_head) moved to line 75~76
        else:
            score = paddle.matmul(
                h_q.reshape(batch_size, self.num_kv_heads, self.head_groups * len_q, self.dim_head),
                h_k.transpose(-1, -2),
            ).view(batch_size, self.num_heads, len_q, len_k)

        if pos_bias_type == "relative":
            score = score + position_bias
        score = paddle.masked_fill(
            score,
            attention_mask.view(batch_size, 1, len_q, len_k) == False,
            paddle.scalar_tensor(float("-inf"), device=score.device, dtype=score.dtype),
        )

        score = self.softmax(score)

        score = paddle.masked_fill(
            score,
            attention_mask.view(batch_size, 1, len_q, len_k) == False,
            paddle.scalar_tensor(0, device=score.device, dtype=score.dtype),
        )

        if self.dropout is not None:
            score = self.dropout(score)

            # (b, n_kv_h, n_h_groups*len_q, len_k) @ (b, n_kv_h, len_k, d_h) -> (b, n_kv_h, n_h_groups*len_q, d_h) -> (b, n_h, len_q, d_h)
        score = paddle.matmul(score.view(batch_size, self.num_kv_heads, self.head_groups * len_q, len_k), h_v).view(
            batch_size, self.num_heads, len_q, self.dim_head
        )

        score = score.view(batch_size, self.num_heads, len_q, self.dim_head).permute(0, 2, 1, 3)
        score = score.contiguous().view(batch_size, len_q, self.num_heads * self.dim_head)

        score = self.attention_out(score)
        if use_cache:
            return score, (h_k, h_v)
        else:
            return score
