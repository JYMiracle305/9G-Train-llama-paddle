import math
from typing import Tuple
from typing import Union

import paddle
import paddle.nn.functional as F


class SegmentPositionEmbedding(paddle.autograd.PyLayer):
    def __init__(
        self,
        num_heads: int,
        num_segments: int = 1,
        num_buckets: int = 32,
        max_distance: int = 128,
        bidirectional: bool = False,
        dtype: paddle.dtype = paddle.float16,
        init_mean: float = 0.0,
        init_std: float = 1,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.bidirectional = bidirectional
        self.num_segments = num_segments

        self.relative_attention_bias = paddle.create_parameter(
            shape=[num_segments * num_segments + num_buckets, num_heads], dtype=dtype
        )

    def forward(
        self,
        key_pos: paddle.Tensor,
        query_pos: paddle.Tensor,
        key_segment: paddle.Tensor,
        query_segment: paddle.Tensor,
    ):
        with paddle.no_grad():
            batch = key_pos.size(0)
            keylen = key_pos.size(1)
            querylen = query_pos.size(1)

            assert key_pos.size(0) == query_pos.size(0)
            assert keylen == key_segment.size(1) and querylen == query_segment.size(1)

            key_pos = key_pos.view(batch, -1, keylen)
            query_pos = query_pos.view(batch, querylen, -1)
            key_segment = key_segment.view(batch, -1, keylen)
            query_segment = query_segment.view(batch, querylen, -1)

            relative_position_bucket = self._segment_relative_position_bucket(query_segment, key_segment)
            relative_position_bucket = relative_position_bucket + self.num_buckets  # 与相对位置编码区间不重叠

            # b*q*k
            absolute_position_bucket = self._position_bucket(
                paddle.arange(keylen, dtype=paddle.int32, device=relative_position_bucket.device)[None, :]
                - paddle.arange(querylen, dtype=paddle.int32, device=relative_position_bucket.device)[:, None],
                bidirectional=self.bidirectional,
                num_buckets=self.num_buckets,
                max_distance=self.max_distance,
            )
            relative_position_bucket = paddle.where(
                (key_segment == query_segment),
                absolute_position_bucket[None, :, :],
                relative_position_bucket,
            )
            # (batch, len_q, len_k)

        # (batch, len_q, len_k, num_heads)
        embeds = F.embedding(relative_position_bucket, self.relative_attention_bias)
        # (batch, num_heads, len_q, len_k)
        embeds = embeds.permute(0, 3, 1, 2).contiguous()
        return embeds

    def _segment_relative_position_bucket(self, query_segment, key_segment):
        return query_segment * self.num_segments + key_segment

    def _position_bucket(self, relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets = (relative_position > 0).to(paddle.int32) * num_buckets
            relative_position = paddle.abs(relative_position)
        else:
            relative_position = -paddle.min(relative_position, paddle.zeros_like(relative_position))
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        relative_postion_if_large = max_exact + (
            paddle.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(paddle.int32)
        relative_postion_if_large = paddle.min(
            relative_postion_if_large,
            paddle.full_like(relative_postion_if_large, num_buckets - 1),
        )
        relative_buckets += paddle.where(is_small, relative_position.to(paddle.int32), relative_postion_if_large)
        return relative_buckets


class BucketPositionBias(paddle.autograd.PyLayer):
    def __init__(
        self,
        num_heads: int,
        num_buckets: int = 32,
        num_segment_bucket: int = 32,
        max_distance: int = 128,
        dtype: paddle.dtype = paddle.float16,
        init_mean: float = 0.0,
        init_std: float = 1,
    ) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.num_segment_bucket = num_segment_bucket
        self.max_distance = max_distance

        self.relative_attention_bias = paddle.create_parameter(
            shape=[num_buckets + num_segment_bucket, num_heads], dtype=dtype
        )

    def forward(
        self,
        query_pos: paddle.Tensor,  # (batch, len_q)
        key_pos: paddle.Tensor,  # (batch, len_k)
        rel_buckets: paddle.Tensor,  # (batch, len_q, len_k)
    ):
        with paddle.no_grad():
            batch = key_pos.size(0)
            keylen = key_pos.size(1)
            querylen = query_pos.size(1)

            assert key_pos.size(0) == query_pos.size(0)
            assert rel_buckets.size(0) == batch and rel_buckets.size(1) == querylen and rel_buckets.size(2) == keylen

            relative_position_bucket = rel_buckets - 1 + self.num_buckets  # 与相对位置编码区间不重叠

            # b*q*k
            inner_segment_bucket = self._position_bucket(
                key_pos[..., None, :] - query_pos[..., :, None],
                num_buckets=self.num_buckets,
                max_distance=self.max_distance,
            )
            relative_position_bucket = paddle.where(
                rel_buckets == 0,
                inner_segment_bucket,
                relative_position_bucket,
            )
            # (batch, len_q, len_k)

        # (batch, len_q, len_k, num_heads)
        embeds = F.embedding(relative_position_bucket, self.relative_attention_bias)
        # (batch, num_heads, len_q, len_k)
        embeds = embeds.permute(0, 3, 1, 2).contiguous()
        return embeds

    def _position_bucket(self, relative_position, num_buckets=32, max_distance=128):
        relative_buckets = 0
        num_buckets //= 2
        relative_buckets = (relative_position > 0).to(paddle.int32) * num_buckets
        relative_position = paddle.abs(relative_position)

        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        relative_postion_if_large = max_exact + (
            paddle.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(paddle.int32)
        relative_postion_if_large = paddle.min(
            relative_postion_if_large,
            paddle.full_like(relative_postion_if_large, num_buckets - 1),
        )
        relative_buckets += paddle.where(is_small, relative_position.to(paddle.int32), relative_postion_if_large)
        return relative_buckets


class RotaryEmbedding(paddle.autograd.PyLayer):
    def __init__(
        self,
        dim,
        base=10000,
        distance_scale: Union[int, float] = 1,
        dtype: paddle.dtype = paddle.float16,
    ):
        super().__init__()
        inv_freq = 1.0 / (base ** (paddle.arange(0, dim, 2, device="cuda", dtype=paddle.float32) / dim))
        inv_freq = inv_freq.to(dtype)
        self.distance_scale = distance_scale
        self.dtype = dtype
        self.inv_freq = inv_freq

    def forward(self, x: paddle.Tensor, x_pos: paddle.Tensor):
        """
        Args:
            x (:obj:`paddle.Tensor` of shape ``(..., dim)``): Inputs.
            x_pos (:obj:`paddle.Tensor` of shape ``(...)``): Positions of inputs.
        """
        x_pos = x_pos * self.distance_scale
        freqs = x_pos[..., None].to(self.dtype) * self.inv_freq[None, :]  # (..., dim/2)

        # the same implementation as sat
        emb = paddle.cat((freqs, freqs), dim=-1)  # (..., dim)
        emb_cos = emb.cos()  # (..., dim)
        emb_sin = emb.sin()  # (..., dim)

        rotate_x = paddle.cat([-x[..., x.size(-1) // 2 :], x[..., : x.size(-1) // 2]], dim=-1)  # (..., dim)

        return x * emb_cos + rotate_x * emb_sin


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return paddle.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin, seq_dim, offset):
    if x.size(seq_dim) < cos.size(seq_dim):
        cos = cos.narrow(seq_dim, offset, x.size(seq_dim))
        sin = sin.narrow(seq_dim, offset, x.size(seq_dim))
    return (x * cos) + (rotate_half(x) * sin)


class RotaryEmbeddingESM(paddle.autograd.PyLayer):
    """
    Rotary position embeddings based on those in
    [RoFormer](https://huggingface.co/docs/transformers/model_doc/roformer). Query and keys are transformed by rotation
    matrices which depend on their relative positions.
    """

    def __init__(
        self,
        dim: int,
        base: Union[int, float] = 10000,
        distance_scale: Union[int, float] = 1,
        dtype=paddle.float16,
        persistent=True,
        mixed_precision=False,
    ):
        super().__init__()
        self.base = base
        self.distance_scale = distance_scale
        self.dtype = dtype

        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = 1.0 / (base ** (paddle.arange(0, dim, 2, device="cuda", dtype=paddle.float32) / dim))
        if mixed_precision:
            self.register_buffer("inv_freq", inv_freq, persistent=persistent)
        else:
            self.register_buffer("inv_freq", inv_freq.to(self.dtype), persistent=persistent)

        self._seq_len_cached = -1
        self._cos_cached = None
        self._sin_cached = None
        self.mixed_precision = mixed_precision

        self.apply_rotary_pos_emb = apply_rotary_pos_emb

    def _update_cos_sin_tables(self, x, seq_dim, offset):
        seq_len = x.size(seq_dim) + offset
        if seq_len > self._seq_len_cached or self._cos_cached.device != x.device:
            self._seq_len_cached = seq_len
            t = paddle.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = paddle.outer(t * self.distance_scale, self.inv_freq)
            emb = paddle.cat((freqs, freqs), dim=-1)
            for i in range(x.dim() - 1):
                if i != seq_dim:
                    emb = emb.unsqueeze_(i)
            if self.mixed_precision:
                self._cos_cached = emb.cos().to(self.dtype)
                self._sin_cached = emb.sin().to(self.dtype)
            else:
                self._cos_cached = emb.cos()
                self._sin_cached = emb.sin()
        return self._cos_cached, self._sin_cached

    def forward(self, q: paddle.Tensor, k: paddle.Tensor, seq_dim, offset=0) -> Tuple[paddle.Tensor, paddle.Tensor]:
        seq_dim = (seq_dim + k.dim()) % k.dim()
        self._cos_cached, self._sin_cached = self._update_cos_sin_tables(k, seq_dim, offset)
        return (
            self.apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached, seq_dim, offset),
            self.apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached, seq_dim, offset),
        )
