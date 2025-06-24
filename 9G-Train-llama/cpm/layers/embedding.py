import math
from typing import Optional

import bmtrain_paddle as bmt
import paddle
import paddle.nn.functional as F

from .position_embedding import RotaryEmbedding


class Embedding(bmt.DistributedModule):
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
        # self.weight = bmt.DistributedParameter(
        #     paddle.empty([embedding_size, vocab_size], dtype=dtype),
        #     init_method=paddle.nn.initializer.XavierNormal(),
        # )
        self.weight = self.create_parameter(
            shape=[embedding_size, vocab_size], dtype=dtype,
            default_initializer=paddle.nn.initializer.XavierNormal(),
        )
        self.scale = scale

    def forward(self, ids: paddle.Tensor):
        """
        Args:
            ids (:obj:`paddle.Tensor` of shape ``(batch_size, seq_len)``): Indices of input sequence tokens.
        Return:
            :obj:`paddle.Tensor` of shape ``(batch_size, seq_len, embedding_size)``: The embedding output.
        """  # noqa: E501
        # print("before------Embedding----", ids[:123], self.weight)
        if self.scale:
            embeds = F.embedding(ids, self.weight.T) / math.sqrt(self.dim_model)
        else:
            embeds = F.embedding(ids, self.weight.T)
        # print("------Embedding----", self.scale, ids)
        # print("------------------", self.weight)
        # print("~~~~~~~~~~~~~~~~~~~~", embeds)
        return embeds

    def projection(self, x: paddle.Tensor):
        """
        Projection based on embedding's weight. For example, embedding map vocab_size to embed_size, than projection map embed_size back to vocab_size.
        Args:
            x (:obj:`paddle.Tensor` of shape ``(batch, seq_len, dim_model)``): Input of projection
        Returns:
            :obj:`paddle.Tensor` of shape ``(batch, seq_len, vocab_output_size)``: The projection output.
        """  # noqa: E501
        if self.scale:
            # print("~~~~~~~~~~Embedding shape~~~~~~~~~~~~~", (x / math.sqrt(self.dim_model)).shape, self.weight.shape)
            logits = F.linear(x / math.sqrt(self.dim_model), self.weight)
        else:
            # print("~~~~~~~~~~Embedding project shape~~~~~~~~~~~~~", x.shape, self.weight.shape)
            logits = F.linear(x, self.weight)
        return logits


class EmbeddingExt(bmt.DistributedModule):
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

        self.weight = bmt.DistributedParameter(
            paddle.empty([vocab_size, embedding_size], dtype=dtype),
            init_method=paddle.nn.initializer.XavierNormal(),
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


class VocabParallelEmbedding(bmt.DistributedModule):
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
        assert vocab_size % config["tp_size"] == 0
        self.vocab_size_per_partition = vocab_size // config["tp_size"]
        self.start_index = config["tp_rank"] * self.vocab_size_per_partition
        self.end_index = (config["tp_rank"] + 1) * self.vocab_size_per_partition
        self.weight = bmt.DistributedParameter(
            paddle.empty([self.vocab_size_per_partition, embedding_size], dtype=dtype),
            init_method=paddle.nn.initializer.XavierNormal(),
            tp_split_dim=0,
            tp_mode=True,
        )

    def forward(self, ids: paddle.Tensor, gather_input=True):
        """
        Args:
            ids (:obj:`paddle.Tensor` of shape ``(batch_size, seq_len)``): Indices of input sequence tokens.
            gather_input (bool) : whether gather input is required between  tensor parallel group)
        Return:
            :obj:`paddle.Tensor` of shape ``(batch_size, seq_len, embedding_size)``: The embedding output.
        """  # noqa: E501

        if gather_input:
            ids = all_gather(ids, comm=config["tp_comm"])
        input_mask = (ids < self.start_index) | (ids >= self.end_index)
        ids = ids.clone() - self.start_index
        ids[input_mask] = 0

        embeds = F.embedding(ids, self.weight)

        embeds[input_mask, :] = 0.0
        embeds = all_reduce(embeds, op="sum", comm=config["tp_comm"])
        embed_list = embeds.chunk(config["tp_size"], dim=0)
        embeds = embed_list[config["tp_rank"]].flatten(0, 1)

        if self.scale:
            embeds = embeds / math.sqrt(self.dim_model)

        return embeds

    def projection(self, x: paddle.Tensor, gather_output=False, gather_input=True):
        """
        Projection based on embedding's weight. For example, embedding map vocab_size to embed_size, than projection map embed_size back to vocab_size.
        Args:
            x (:obj:`paddle.Tensor` of shape ``(batch, seq_len, dim_model)``): Input of projection
        Returns:
            :obj:`paddle.Tensor` of shape ``(batch, seq_len, vocab_output_size)``: The projection output.
        """  # noqa: E501
        if self.scale:
            x = x / math.sqrt(self.dim_model)
        out = bmt.nn.OpParallelLinear.apply(x, self.weight, None, gather_input, gather_output, False, None)
        return out
