from typing import List
from typing import Optional
from typing import Tuple

import paddle

from .blocks import TransformerBlock
from .layernorm import LayerNorm


class Encoder(paddle.autograd.PyLayer):
    """Layers of encoder transformer blocks plus an final layernorm.

    Args:
        num_layers (int): number of layers.
        dim_model (int): main dimension of modules in transformer blocks.
        dim_ff (int): dim_ff used in :py:class:`model_center.layer.FeedForward`.
        num_heads (int): num_heads used in :py:class:`model_center.layer.Attention`.
        dim_head (int): dim_head used in :py:class:`model_center.layer.Attention`.
        dtype (optional): Defaults to paddle.float16.
        eps (float, optional): eps used in :py:class:`model_center.layer.LayerNorm`. Defaults to 1e-6.
        dropout_p (float, optional): Defaults to 0.
    """

    def __init__(
        self,
        num_layers: int,
        dim_model: int,
        dim_ff: int,
        num_heads: int,
        dim_head: int,
        num_kv_heads: int = -1,
        activate_fn: str = "gelu",
        dtype: paddle.dtype = paddle.float16,
        eps: float = 1e-6,
        dropout_p: Optional[float] = None,
        scale: bool = True,
        mask_modules: Optional[List[Tuple[bool, bool]]] = None,
        use_flash_attn: bool = False,
    ):
        super().__init__()
        if num_kv_heads == -1:
            num_kv_heads = num_heads
        self.num_layers = num_layers

        if mask_modules is not None:
            assert len(mask_modules) == num_layers, "The total number of masks should equal to num_layers"
            for mask_module in mask_modules:
                assert len(mask_module) == 2, "For encoder, each mask should be (mask_att, mask_ffn)"
        else:
            mask_modules = [(False, False)] * num_layers

        self.layers = paddle.nn.ModuleList(
            [
                TransformerBlock(
                    dim_model=dim_model,
                    dim_ff=dim_ff,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    dim_head=dim_head,
                    activate_fn=activate_fn,
                    dtype=dtype,
                    eps=eps,
                    dropout_p=dropout_p,
                    scale=scale,
                    mask_att=mask_modules[ith][0],
                    mask_ffn=mask_modules[ith][1],
                    use_flash_attn=use_flash_attn,
                )
                for ith in range(num_layers)
            ]
        )

        self.output_layernorm = LayerNorm(dim_norm=dim_model, dtype=dtype, eps=eps)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: paddle.Tensor,
        position_bias: paddle.Tensor,
        use_cache: bool = False,
        past_key_values: Optional[List[Tuple[paddle.Tensor, paddle.Tensor]]] = None,
        pos_bias_type: Optional[str] = "relative",
        length_mask: Optional[paddle.Tensor] = None,
        context_mask: Optional[paddle.Tensor] = None,
    ):
        """
        Args:
            hidden-states (:obj:`paddle.Tensor` of shape ``(batch, seq_enc, dim_model)``): Input of encoder, might be the embedding of a batch of sequences.
            attention_mask (:obj:`paddle.Tensor` of shape ``(batch, seq_enc, seq_enc)``): Avoid invalid areas to participate in the calculation
            position_bias(:obj:`paddle.Tensor` of shape ``(num_heads, seq_enc, seq_enc)``) Provides position information to attention mechanism.

        Return:
            :obj:`paddle.Tensor` of shape ``(batch, seq_enc, dim_model)``: The encoder output.

        """
        if not use_cache:
            for layer in self.layers:
                hidden_states = layer(
                    hidden_states,
                    attention_mask,
                    position_bias,
                    pos_bias_type=pos_bias_type,
                    length_mask=length_mask,
                    context_mask=context_mask,
                )
            # print("--------Encoder---------", hidden_states)
            hidden_states = self.output_layernorm(hidden_states)
            return hidden_states
        else:
            with paddle.no_grad():
                current_key_values = []
                current_hidden_states = []
                for i, module in enumerate(self.layers):
                    hidden_states = module(
                        hidden_states,
                        attention_mask,
                        position_bias,
                        past_key_value=past_key_values[i] if past_key_values else None,
                        use_cache=use_cache,
                        pos_bias_type=pos_bias_type,
                        length_mask=length_mask,
                        context_mask=context_mask,
                    )
                    if use_cache:
                        current_key_values.append(hidden_states[1])
                        current_hidden_states.append(hidden_states[0])
                        hidden_states = hidden_states[0]
                hidden_states = self.output_layernorm(hidden_states)
                if use_cache:
                    return hidden_states, current_key_values, current_hidden_states
                else:
                    return hidden_states
