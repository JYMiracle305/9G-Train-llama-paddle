from typing import Optional
from typing import Tuple
from typing import Union

import bmtrain_paddle as bmt
import paddle

from .attention import Attention
from .feedforward import FeedForward
from .layernorm import LayerNorm
from .position_embedding import RotaryEmbedding
from .position_embedding import RotaryEmbeddingESM


class SelfAttentionBlock(bmt.DistributedModule):
    """The whole cross-attention block. A sequence of operation. Consists of layernorm, self-attention and residual connection.

    Args:
        dim_model (int): main dimension of modules in transformer blocks.
        num_heads (int): num_heads used in :py:class:`model_center.layer.Attention`.
        dim_head (int): dim_head used in :py:class:`model_center.layer.Attention`.
        dtype (optional): Defaults to paddle.float16.
        eps (float, optional): eps used in :py:class:`model_center.layer.LayerNorm`. Defaults to 1e-5.
        dropout_p (float, optional): Defaults to 0.
    """  # noqa: E501

    def __init__(
        self,
        dim_model: int,
        num_heads: int,
        num_kv_heads: int,
        dim_head: int,
        dtype=paddle.float16,
        eps: float = 1e-5,
        dropout_p: Optional[float] = None,
        scale: bool = True,
        add_qkv_bias: bool = False,
        use_flash_attn: bool = False,
        tp: int = 0,
    ):
        super().__init__()

        self.layernorm_before_attention = LayerNorm(
            dim_model,
            dtype=dtype,
            eps=eps,
        )

        self.self_attention = Attention(
            dim_model=dim_model,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dim_head=dim_head,
            dtype=dtype,
            dropout_p=dropout_p,
            scale=scale,
            add_qkv_bias=add_qkv_bias,
            use_flash_attn=use_flash_attn,
            tp=tp,
        )

        if dropout_p:
            self.dropout = paddle.nn.Dropout(dropout_p)
        else:
            self.dropout = None

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: paddle.Tensor = None,
        position_bias: Union[paddle.Tensor, RotaryEmbedding, RotaryEmbeddingESM] = None,
        use_cache: bool = False,
        past_key_value: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None,
        pos_bias_type: Optional[str] = "relative",
        length_mask: Optional[paddle.Tensor] = None,
        attention_mask_bias: Optional[paddle.Tensor] = None,
        cu_seqlens: Optional[paddle.Tensor] = None,
        max_seqlen: int = None,
        position_ids: Optional[paddle.Tensor] = None,
    ):
        """
        Args:
            hidden_states (:obj:`paddle.Tensor` of shape ``(batch, seq_self, dim_model)``): Input of self-attention block. It can be the embedding of a batch of sequences.
            attention_mask (:obj:`paddle.Tensor` of shape ``(batch, seq_self, seq_self)``): Avoid invalid areas to participate in the calculation.
            position_bias (:obj:`paddle.Tensor` of shape ``(num_heads, seq_self, seq_self)``): Provide positional information to self-attention block.

        Return:
            :obj:`paddle.Tensor` of shape ``(batch, seq_self, dim_model)``: The output of attention block.

        """  # noqa: E501
        # print("-----before self.layernorm_before_attention(hidden_states)2------", hidden_states)
        x = self.layernorm_before_attention(hidden_states)
        # print("-----after self.layernorm_before_attention(hidden_states)------")
        x = self.self_attention(
            x,
            x,
            attention_mask,
            position_bias,
            use_cache,
            past_key_value,
            pos_bias_type=pos_bias_type,
            length_mask=length_mask,
            attention_mask_bias=attention_mask_bias,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            position_ids=position_ids,
        )
        if use_cache:
            x, current_key_value = x
        else:
            current_key_value = None

        if self.dropout is not None:
            x = self.dropout(x)
        hidden_states = hidden_states + x  # / 1.05
        if use_cache:
            return hidden_states, current_key_value
        else:
            return hidden_states


class FFNBlock(paddle.nn.Layer):
    """The whole feed-forward block. A sequence of operation. Consists of layernorm, feed-forward and residual connection.

    Args:
        dim_model (int): main dimension of modules in transformer blocks.
        dim_ff (int): dim_ff used in :py:class:`model_center.layer.FeedForward`.
        dtype (optional): Defaults to paddle.float16.
        eps (float, optional): eps used in :py:class:`model_center.layer.LayerNorm`. Defaults to 1e-5.
        dropout_p (float, optional): Defaults to 0.
    """  # noqa: E501

    def __init__(
        self,
        dim_model: int,
        dim_ff: int,
        activate_fn: str,
        dtype=paddle.float16,
        eps: float = 1e-6,
        dropout_p: Optional[float] = 0,
        scale: bool = True,
        tp: int = 0,
    ):
        super().__init__()

        self.layernorm_before_ffn = LayerNorm(
            dim_model,
            dtype=dtype,
            eps=eps,
        )

        self.ffn = FeedForward(
            dim_model,
            dim_ff,
            activate_fn=activate_fn,
            dtype=dtype,
            dropout_p=dropout_p,
            scale=scale,
            tp=tp,
        )

        if dropout_p:
            self.dropout = paddle.nn.Dropout(dropout_p)
        else:
            self.dropout = None

    def forward(
        self,
        hidden_states: paddle.Tensor,
    ):
        """
        Args:
            hidden_states (:obj:`paddle.Tensor` of shape ``(batch, seq_self, dim_model)``): Hidden states before feed forward layer.

        Return:
            :obj:`paddle.Tensor` of shape ``(batch, seq_self, dim_model)``: The output of feed-forward block

        """  # noqa: E501
        x = self.layernorm_before_ffn(hidden_states)
        x = self.ffn(x)
        if self.dropout is not None:
            x = self.dropout(x)
        hidden_states = hidden_states + x  # / 1.05
        return hidden_states


class TransformerBlock(paddle.nn.Layer):
    """The whole transformer block. A sequence of operation. Consists of self-attention block[, cross-attention block] and feed-forward block.

    Args:
        dim_model (int): main dimension of modules in transformer blocks.
        dim_ff (int): dim_ff used in :py:class:`model_center.layer.FeedForward`.
        num_heads (int): num_heads used in :py:class:`model_center.layer.Attention`.
        dim_head (int): dim_head used in :py:class:`model_center.layer.Attention`.
        dtype (optional): Defaults to paddle.float16.
        eps (float, optional): eps used in :py:class:`model_center.layer.LayerNorm`. Defaults to 1e-5.
        dropout_p (float, optional): Defaults to 0.
    """  # noqa: E501

    def __init__(
        self,
        dim_model: int,
        dim_ff: int,
        num_heads: int,
        num_kv_heads: int,
        dim_head: int,
        activate_fn: str = "gelu",
        dtype=paddle.float16,
        eps: float = 1e-6,
        dropout_p: Optional[float] = None,
        scale: bool = True,
        add_qkv_bias: bool = False,
        mask_att: bool = False,
        mask_ffn: bool = False,
        use_flash_attn: bool = False,
        tp: int = 0,
    ):
        super().__init__()
        self.mask_att = mask_att
        self.mask_ffn = mask_ffn

        if not self.mask_att:
            self.self_att = SelfAttentionBlock(
                dim_model=dim_model,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                dim_head=dim_head,
                dtype=dtype,
                eps=eps,
                dropout_p=dropout_p,
                scale=scale,
                add_qkv_bias=add_qkv_bias,
                use_flash_attn=use_flash_attn,
                tp=tp,
            )

        if not self.mask_ffn:
            self.ffn = FFNBlock(
                dim_model=dim_model,
                dim_ff=dim_ff,
                activate_fn=activate_fn,
                dtype=dtype,
                eps=eps,
                dropout_p=dropout_p,
                scale=scale,
                tp=tp,
            )

    def forward(
        self,
        self_hidden_states: paddle.Tensor,
        self_attention_mask: paddle.Tensor = None,
        self_position_bias: Optional[paddle.Tensor] = None,
        use_cache: bool = False,
        past_key_value: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None,
        pos_bias_type: Optional[str] = "relative",
        length_mask: Optional[paddle.Tensor] = None,
        attention_mask_bias: Optional[paddle.Tensor] = None,
        cu_seqlens: Optional[paddle.Tensor] = None,
        max_seqlen: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
    ):
        """
        Args:
            self_hidden_states (:obj:`paddle.Tensor` of shape ``(batch, seq_self, dim_model)``): Input of transformer block(self-attention block). It can be the raw embedding of a batch of sequences.
            self_attention_mask (:obj:`paddle.Tensor` of shape ``(batch, seq_self, seq_self)``): Avoid invalid areas to participate in the calculation of self-attention.
            self_position_bias (:obj:`paddle.Tensor` of shape ``(num_heads, seq_self, seq_self)``): Provide positional information to self-attention block.

        Return:
            :obj:`paddle.Tensor` of shape ``(batch, seq_self, dim_model)``: The output of transformer block.

        """  # noqa: E501
        # (batch, dim_model, seq_self)
        current_key_value = None
        if not self.mask_att:
            # print("---------TransformerBlock mask_att-------------", self_hidden_states)
            hidden_states = self.self_att(
                self_hidden_states,
                attention_mask=self_attention_mask,
                position_bias=self_position_bias,
                use_cache=use_cache,
                past_key_value=past_key_value,
                pos_bias_type=pos_bias_type,
                length_mask=length_mask,
                attention_mask_bias=attention_mask_bias,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                position_ids=position_ids,
            )

            if use_cache:
                hidden_states, current_key_value = hidden_states
        else:
            hidden_states = self_hidden_states

        # (batch, dim_model, seq_self)
        if not self.mask_ffn:
            hidden_states = self.ffn(hidden_states)

        if use_cache:
            return hidden_states, current_key_value
        else:
            return hidden_states
