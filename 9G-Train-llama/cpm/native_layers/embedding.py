import math
from typing import Optional

import paddle
import paddle.nn.functional as F

from .position_embedding import RotaryEmbedding


class Embedding(paddle.autograd.PyLayer):
    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        dtype: paddle.dtype = paddle.float16,
        scale: bool = True,
        init_mean: float = 0.0,
        init_std: float = 1,
    ):
        super().__init__()

        self.dim_model = embedding_size
        self.weight = paddle.create_parameter(shape = [vocab_size, embedding_size], dtype=dtype)
        self.scale = scale

    def forward(self, ids: paddle.Tensor):
        """
        Args:
            ids (:obj:`paddle.Tensor` of shape ``(batch_size, seq_len)``): Indices of input sequence tokens.
        Return:
            :obj:`paddle.Tensor` of shape ``(batch_size, seq_len, embedding_size)``: The embedding output.
        """  # noqa: E501

        if self.scale:
            embeds = F.embedding(ids, self.weight) / math.sqrt(self.dim_model)
        else:
            embeds = F.embedding(ids, self.weight)
        return embeds.clone()

    def projection(self, x: paddle.Tensor):
        """
        Projection based on embedding's weight. For example, embedding map vocab_size to embed_size, than projection map embed_size back to vocab_size.
        Args:
            x (:obj:`paddle.Tensor` of shape ``(batch, seq_len, dim_model)``): Input of projection
        Returns:
            :obj:`paddle.Tensor` of shape ``(batch, seq_len, vocab_output_size)``: The projection output.
        """  # noqa: E501
        if self.scale:
            logits = F.linear(x / math.sqrt(self.dim_model), self.weight)
        else:
            logits = F.linear(x, self.weight)

        return logits


class EmbeddingExt(paddle.autograd.PyLayer):
    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        dtype: paddle.dtype = paddle.float16,
        init_mean: float = 0.0,
        init_std: float = 1,
        distance_scale: int = 16,
    ):
        super().__init__()

        self.dim_model = embedding_size
        self.rotary_emb = RotaryEmbedding(dim=embedding_size, distance_scale=distance_scale, dtype=dtype)

        self.weight = paddle.create_parameter(
            shape=[vocab_size, embedding_size], dtype=dtype
        )

    def forward(self, ids: paddle.Tensor, ids_sub: paddle.Tensor):
        """
        Args:
            ids (:obj:`paddle.Tensor` of shape ``(batch_size, seq_len)``): Indices of input sequence tokens.
            ids (:obj:`paddle.Tensor` of shape ``(batch_size)``): Subscript of input sequence tokens.
        Return:
            :obj:`paddle.Tensor` of shape ``(batch_size, seq_len, embedding_size)``: The embedding output.
        """  # noqa: E501

        embeds = F.embedding(ids, self.weight) / math.sqrt(self.dim_model)
        return self.rotary_emb(embeds, ids_sub)

    def projection(self, x: paddle.Tensor, ext_table: Optional[paddle.Tensor] = None):
        """
        Projection based on embedding's weight. For example, embedding map vocab_size to embed_size, than projection map embed_size back to vocab_size.
        Args:
            x (:obj:`paddle.Tensor` of shape ``(batch, seq_len, dim_model)``): Input of projection
            ext_table (:obj:`paddle.Tensor` of shape ``(ext_table_size, dim_model)``): Ext vocab table.
        Returns:
            :obj:`paddle.Tensor` of shape ``(batch, seq_len, vocab_size + ext_table_size)``: The projection output.
        """  # noqa: E501
        logits = F.linear(x / math.sqrt(self.dim_model), self.weight)
        if ext_table is not None:
            logits_ext = F.linear(x, ext_table)
            logits = paddle.cat([logits, logits_ext], dim=-1)
        return logits
