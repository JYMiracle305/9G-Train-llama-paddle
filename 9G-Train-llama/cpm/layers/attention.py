import math
from typing import Optional
from typing import Tuple

try:
    from .flash_triton import FlashAttnFunc
except:
    FlashAttnFunc = None
import bmtrain_paddle as bmt
import paddle
import paddle.nn.functional as F
from einops import rearrange

from .linear import ColumnParallelLinear
from .linear import Linear
from .position_embedding import apply_chatglm_rotary_pos_emb

try:
    from flash_attn.flash_attn_interface import _flash_attn_varlen_backward
    from flash_attn.flash_attn_interface import _flash_attn_varlen_forward
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
except:
    flash_attn_varlen_func = None

try:
    from flash_attn.bert_padding import pad_input
    from flash_attn.bert_padding import unpad_input
except:
    pad_input = None
    unpad_input = None

class OpFlash(paddle.nn.Layer):
    @staticmethod
    def forward(ctx, self, record, q, k, v, cu_seqlens, max_seqlen, dropout_p, causal):
        ctx.self = self
        ctx.cu_seqlens = cu_seqlens
        ctx.max_length = max_seqlen
        ctx.dropout_p = dropout_p
        ctx.causal = causal
        ctx.softmax_scale = q.shape[-1] ** (-0.5)
        if not record and "out" in self._layer_dict:
            out = self._layer_dict.pop("out")
            softmax_lse = self._layer_dict.pop("softmax_lse")
            rng_state = self._layer_dict.pop("rng_state")
        else:
            out, _, _, _, _, softmax_lse, rng_state = _flash_attn_varlen_forward(
                q,
                k,
                v,
                cu_seqlens,
                cu_seqlens,
                max_seqlen,
                max_seqlen,
                dropout_p,
                ctx.softmax_scale,
                causal=causal,
                return_softmax=False,
                use_alibi=False,
                alibi_mode=0,
            )
            if record:
                self._layer_dict["out"] = out
                self._layer_dict["softmax_lse"] = softmax_lse
                self._layer_dict["rng_state"] = rng_state

        ctx.save_for_backward(q, k, v, out, softmax_lse, rng_state)
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, softmax_lse, rng_state = ctx.saved_tensors
        dq, dk, dv = paddle.empty_like(q), paddle.empty_like(k), paddle.empty_like(v)
        _flash_attn_varlen_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq,
            dk,
            dv,
            ctx.cu_seqlens,
            ctx.cu_seqlens,
            ctx.max_length,
            ctx.max_length,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            use_alibi=False,
            alibi_mode=0,
            #rng_state=rng_state,
        )
        return None, None, dq, dk, dv, None, None, None, None

def simple_attention(query, key, value, mask=None, dropout=0.0):
    """
    简洁高效的注意力机制实现
    输入:
        query: [batch_size, seq_len, num_heads, head_dim]
        key:   [batch_size, seq_len, num_heads, head_dim]
        value: [batch_size, seq_len, num_heads, head_dim]
        mask:  [batch_size, 1, seq_len, seq_len] (1表示有效位置)
    输出:
        [batch_size, seq_len, num_heads, head_dim]
    """
    batch_size, seq_len, num_heads, head_dim = query.shape
    query = query.transpose([0, 2, 1, 3])  # [bs, heads, q_len, dim]
    key = key.transpose([0, 2, 1, 3])     # [bs, heads, k_len, dim]
    value = value.transpose([0, 2, 1, 3]) # [bs, heads, v_len, dim]

    # 1. 计算点积注意力分数
    # print("simple shape:", query.shape, key.shape)
    scores = paddle.matmul(query, key.transpose([0, 1, 3, 2]))  # [bs, seq_len, num_heads, seq_len]
    
    # 2. 缩放因子
    d_k = query.shape[-1]
    scale_factor = 1.0 / paddle.sqrt(paddle.to_tensor(d_k, dtype=query.dtype))
    scores = scores * scale_factor
    
    # 3. 应用掩码（如果提供）
    if mask is not None:
        # 转换为FP掩码：1表示有效位置，0表示无效位置
        mask_value = paddle.to_tensor(-1e4, dtype=scores.dtype)
        scores = paddle.where(mask > 0.5, scores, mask_value)
    
    # 4. Softmax归一化
    attention_weights = F.softmax(scores, axis=-1)
    
    # 5. 应用Dropout
    if dropout > 0.0:
        attention_weights = F.dropout(attention_weights, p=dropout)
    
    # 6. 加权求和
    result = paddle.matmul(attention_weights, value)
    
    return result

def handle_variable_length(query, key, value, cu_seqlens):
    """
    处理变长序列的辅助函数
    返回填充后的张量和序列掩码
    """
    batch_size = cu_seqlens.shape[0] - 1
    # print("-------cu_seqlens----------", cu_seqlens.shape, cu_seqlens[1:], cu_seqlens[:-1])
    # 1. 计算最大序列长度
    seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    # print("--------seq_lens---------", seq_lens, seq_lens.max(), seq_lens.max().item())
    max_seqlen = seq_lens.max().item() if seq_lens.size > 0 else 0
    # print("--------max_seqlen---------", max_seqlen)
    # 2. 创建结果张量
    def pad_tensor(tensor):
        # print("--------------max_seqlen----------------", batch_size, max_seqlen, tensor.shape)
        padded = paddle.zeros([batch_size, max_seqlen, *tensor.shape[1:]], dtype=tensor.dtype)
        for i in range(batch_size):
            start = cu_seqlens[i].item()
            end = cu_seqlens[i+1].item()
            if start < end:  # 只处理非空序列
                # print("-----------------------", padded.shape, tensor.shape, padded[i, :end-start].shape, tensor[start:end].shape)
                # --------------max_seqlen---------------- 45 1 [16384, 32, 128]
                # ----------------------- [45, 32, 128] [16384, 32, 128] [32, 128] [349, 32, 128]
                padded[i, :end-start] = tensor[start:end]
        return padded
    
    # 3. 创建序列掩码
    padding_mask = paddle.zeros([batch_size, max_seqlen])
    for i in range(batch_size):
        start = cu_seqlens[i].item()
        end = cu_seqlens[i+1].item()
        padding_mask[i, :end-start] = 1.0
    
    # 因果掩码（下三角）
    causal_mask = paddle.tril(paddle.ones([max_seqlen, max_seqlen], dtype=padding_mask.dtype))
    
    # 组合掩码
    combined_mask = causal_mask * padding_mask.unsqueeze(1)  # [batch_size, max_seqlen, max_seqlen]
    
    return (
        pad_tensor(query),
        pad_tensor(key),
        pad_tensor(value),
        combined_mask.unsqueeze(1)  # 添加注意力头维度
    )

def prepare_flash_attention(h_q, h_k, h_v, cu_seqlens, max_seqlen, dropout_p=0.0):
    """
    完整的变长序列注意力实现
    输入:
        h_q, h_k, h_v: [total_tokens, num_heads, head_dim]
        cu_seqlens: 累计序列长度 [batch_size + 1]
    输出:
        [total_tokens, num_heads, head_dim]
    """
    # print("------------------cu_seqlens--------------------", cu_seqlens.shape, cu_seqlens)
    # 1. 获取维度信息
    total_tokens = h_q.shape[0]
    num_heads = h_q.shape[1]
    head_dim = h_q.shape[2]
    batch_size = cu_seqlens.shape[0] - 1
    
    # 2. 处理空批次
    if batch_size == 0:
        return paddle.zeros_like(h_q)
    
    # 3. 填充序列
    # q_padded, k_padded, v_padded, mask = handle_variable_length(
    #     h_q.reshape([total_tokens, num_heads, head_dim]),
    #     h_k.reshape([total_tokens, num_heads, head_dim]),
    #     h_v.reshape([total_tokens, num_heads, head_dim]),
    #     cu_seqlens
    # )
    # print("--------------handle_variable_length------------", q_padded.shape, k_padded.shape, v_padded.shape, mask.shape)
    # 初始化填充后的张量
    # print("--------------handle_variable_length------------", batch_size, max_seqlen, num_heads, head_dim)
    q_padded = paddle.zeros([batch_size, max_seqlen, num_heads, head_dim], dtype=h_q.dtype)
    k_padded = paddle.zeros_like(q_padded)
    v_padded = paddle.zeros_like(q_padded)
    
    # 创建键填充掩码 (key_padding_mask)
    key_padding_mask = paddle.zeros([batch_size, max_seqlen], dtype='bool')
    
    # 填充每个序列
    for i in range(batch_size):
        start = int(cu_seqlens[i].item())
        end = int(cu_seqlens[i+1].item())
        seq_len = end - start
        
        if seq_len > 0:
            # 提取序列数据
            seq_q = h_q[start:end].reshape([seq_len, num_heads, head_dim])
            seq_k = h_k[start:end].reshape([seq_len, num_heads, head_dim])
            seq_v = h_v[start:end].reshape([seq_len, num_heads, head_dim])
            
            # 填充到max_seqlen
            q_padded[i, :seq_len] = seq_q
            k_padded[i, :seq_len] = seq_k
            v_padded[i, :seq_len] = seq_v
            
            # 设置有效位置
            key_padding_mask[i, :seq_len] = True

    # 转置为 [batch_size, num_heads, max_seqlen, head_dim] [45, 32, 716, 128]
    q_padded = q_padded.transpose([0, 2, 1, 3])
    k_padded = k_padded.transpose([0, 2, 1, 3])
    v_padded = v_padded.transpose([0, 2, 1, 3])

    # 4. 应用注意力
    result = simple_attention(
        q_padded,
        k_padded,
        v_padded,
        mask=None,
        dropout=dropout_p
    )
    # print("-------------simple_attention-----------------", result.shape)
    # 5. 恢复原始序列格式
    # output = paddle.zeros([cu_seqlens[-1].item() - cu_seqlens[0].item(), num_heads, head_dim], 
    #                      dtype=result.dtype)
    # current_idx = 0
    # for i in range(batch_size):
    #     start = cu_seqlens[i].item()
    #     end = cu_seqlens[i+1].item()
    #     length = end - start
    #     if length > 0:
    #         print("---------output[i, :length]-----------", output[i, :length].shape, result[i, :length].shape)
    #         output[start:end] = result[i, :length]
    #     current_idx += length
    
    # # 6. 重塑为原始格式
    # print("--------output.shape----------", output.shape, total_tokens, num_heads, head_dim)
    # return output.reshape([total_tokens, num_heads, head_dim])

    # 恢复原始格式
    # print("attention result:", result.shape)   # [45, 716, 32, 128]
    attn_output = result.transpose([0, 2, 1, 3])
    
    # 提取非填充部分
    output = []
    for i in range(batch_size):
        start = cu_seqlens[i]
        end = cu_seqlens[i+1]
        seq_len = end - start
        if seq_len > 0:
            # output.append(attn_output[i, :seq_len].reshape([seq_len, num_heads, head_dim]))
            output.append(attn_output[i, :, :seq_len, :].transpose([1, 0, 2]))

    return paddle.concat(output, axis=0) if output else paddle.zeros([0, num_heads, head_dim])

class Attention(bmt.DistributedModule):
    def __init__(
        self,
        dim_model: int,
        num_heads: int,
        num_kv_heads: int,
        dim_head: int,
        dtype: paddle.dtype = paddle.float16,
        dropout_p: Optional[float] = None,
        scale: bool = True,
        add_qkv_bias: bool = False,
        use_flash_attn: bool = False,
        tp: int = 0,
    ) -> None:
        super().__init__()

        self.dim_model = dim_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_groups = num_heads // num_kv_heads
        self.dim_head = dim_head

        self.project_q = Linear(
            self.dim_model,
            self.num_heads * self.dim_head,
            bias=add_qkv_bias,
            dtype=dtype,
            scale=scale,
            tp=tp,
        )
        self.project_k = Linear(
            self.dim_model,
            self.num_kv_heads * self.dim_head,
            bias=add_qkv_bias,
            dtype=dtype,
            scale=scale,
            tp=tp,
        )
        self.project_v = Linear(
            self.dim_model,
            self.num_kv_heads * self.dim_head,
            bias=add_qkv_bias,
            dtype=dtype,
            scale=scale,
            tp=tp,
        )

        self.attention_out = Linear(
            self.num_heads * self.dim_head,
            self.dim_model,
            dtype=dtype,
            scale=scale,
            tp=tp * 2,
        )

        self.softmax = paddle.nn.Softmax(axis=-1)

        if dropout_p is not None:
            self.dropout = paddle.nn.Dropout(p=dropout_p)
            self.dropout_p = dropout_p
        else:
            self.dropout = None

        self.use_flash_attn = use_flash_attn
        self._layer_dict = {}

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
        attention_mask_bias: Optional[paddle.Tensor] = None,
        cu_seqlens: Optional[paddle.Tensor] = None,
        max_seqlen: int = None,
        position_ids: Optional[paddle.Tensor] = None,
    ):
        """This model inherits from bmt.DistributedModule.
        Args:
            hidden_q (:obj:`paddle.Tensor` of shape ``(batch, len_q, dim_model)``): Indices of input sequence tokens. It will be embedded by model's internal embedding lookup matrix.
            hidden_kv (:obj:`paddle.Tensor` of shape ``(batch, len_k, dim_model)``): Length of input sequence before padding.
            attention_mask (:obj:`paddle.Tensor` of shape ``(batch, len_q, len_k)``): Used to avoid performing attention on padding token indices.
            position_bias(:obj:`paddle.Tensor` of shape ``(num_heads, len_q, len_k)`` or ``(1, num_heads, len_k, len_q)``): Provide positional information about tensor `key_value` and `query`.
        Return:
            out (:obj:`paddle.Tensor` of shape ``(batch, len_q, dim_model)``): The attention output.
        """  # noqa: E501

        len_q = hidden_q.shape[1]
        len_k = hidden_kv.shape[1]

        if isinstance(self.project_q, ColumnParallelLinear):
            assert hidden_q.data_ptr() == hidden_kv.data_ptr()
            if self.project_q.scale and self.project_q.scale_before:
                hidden_q = hidden_q / math.sqrt(self.project_q.dim_in)
            hidden_q = bmt.nn.OpParallelLinear.apply(
                hidden_q,
                paddle.cat([self.project_q.weight, self.project_k.weight, self.project_v.weight], dim=0),
                paddle.cat([self.project_q.bias, self.project_k.bias, self.project_v.bias], dim=0)
                if self.project_q.bias is not None
                else None,
                True,
                False,
                False,
                None,
            )
            if self.project_q.scale and not self.project_q.scale_before:
                hidden_q = hidden_q / math.sqrt(self.project_q.dim_in)

            block_size = hidden_q.shape[-1] // (self.head_groups + 1 + 1)
            h_q = hidden_q[..., : block_size * self.head_groups]
            h_k = hidden_q[..., block_size * self.head_groups : block_size * (self.head_groups + 1)]
            h_v = hidden_q[..., block_size * (self.head_groups + 1) :]
        else:
            h_q = self.project_q(hidden_q)
            h_k = self.project_k(hidden_kv)
            h_v = self.project_v(hidden_kv)

        batch_size = h_q.shape[0]

        if not self.use_flash_attn:
            h_q = h_q / math.sqrt(math.sqrt(self.dim_head))
            h_k = h_k / math.sqrt(math.sqrt(self.dim_head))

            h_q = h_q.reshape([batch_size, len_q, -1, self.dim_head]).permute(0, 2, 1, 3)
            h_k = h_k.reshape([batch_size, len_k, -1, self.dim_head]).permute(0, 2, 1, 3)
            h_v = h_v.reshape([batch_size, len_k, -1, self.dim_head]).permute(0, 2, 1, 3)

            if pos_bias_type == "rotary":
                # b h s d
                h_q, h_k = position_bias(h_q, h_k, -2, offset=past_kv[0].shape[-2] if past_kv is not None else 0)
            elif pos_bias_type == "chatglm_rotary":
                h_q = apply_chatglm_rotary_pos_emb(h_q, position_bias)
                h_k = apply_chatglm_rotary_pos_emb(h_k, position_bias)

            if past_kv is not None:
                h_k = paddle.cat([past_kv[0], h_k], dim=-2)
                h_v = paddle.cat([past_kv[1], h_v], dim=-2)
                len_k = h_k.shape[-2]

            # (b, n_h, len_q, d_h) @ (b, n_h, d_h, len_k) -> (b, n_h, len_q, len_k)
            # (b, n_kv_h, n_h_groups*len_q, d_h) @ (b, n_kv_h, d_h, len_k) -> (b, n_kv_h, n_h_groups*len_q, len_k) -> (b, n_h, len_q, len_k)
            if self.head_groups == 1:
                score = paddle.matmul(h_q, h_k.transpose(-1, -2))  # / math.sqrt(self.dim_head) moved to line 75~76
            else:
                score = paddle.matmul(
                    h_q.reshape([batch_size, -1, self.head_groups * len_q, self.dim_head]),
                    h_k.transpose(-1, -2),
                ).reshape(
                    [batch_size, -1, len_q, len_k]
                )  # / math.sqrt(self.dim_head) moved to line 75~76
            if pos_bias_type == "relative":
                if len_q == 1:  # inference with cache
                    if len(position_bias.shape) == 4:
                        position_bias = position_bias[:, :, -1:, :]
                    else:
                        position_bias = position_bias[:, -1:, :]
                score = score + position_bias
            score = paddle.masked_fill(
                score,
                attention_mask.reshape([batch_size, 1, len_q, len_k]) == False,
                paddle.scalar_tensor(float("-inf"), device=score.device, dtype=score.dtype),
            )

            score = self.softmax(score)

            score = paddle.masked_fill(
                score,
                attention_mask.reshape([batch_size, 1, len_q, len_k]) == False,
                paddle.scalar_tensor(0, device=score.device, dtype=score.dtype),
            )

            if self.dropout is not None:
                score = self.dropout(score)

            # (b, n_h, len_q, len_k) @ (b, n_h, len_k, d_h) -> (b, n_h, len_q, d_h)
            # (b, n_kv_h, n_h_groups*len_q, len_k) @ (b, n_kv_h, len_k, d_h) -> (b, n_kv_h, n_h_groups*len_q, d_h) -> (b, n_h, len_q, d_h)
            score = paddle.matmul(score.reshape([batch_size, -1, self.head_groups * len_q, len_k]), h_v).reshape(
                [batch_size, -1, len_q, self.dim_head]
            )

            score = score.reshape([batch_size, -1, len_q, self.dim_head]).permute(0, 2, 1, 3)
            score = score.contiguous().reshape([batch_size, len_q, -1])

        else:
            if attention_mask_bias is not None:
                assert pos_bias_type == "rotary"
                h_q = h_q.reshape([batch_size, len_q, -1, self.dim_head])  # .permute(0, 2, 1, 3)
                h_k = h_k.reshape([batch_size, len_k, -1, self.dim_head])  # .permute(0, 2, 1, 3)
                h_v = h_v.reshape([batch_size, len_k, -1, self.dim_head])  # .permute(0, 2, 1, 3)
                h_q, h_k = position_bias(h_q, h_k, -3)
                score = FlashAttnFunc.apply(h_q, h_k, h_v, attention_mask_bias, False, None)
            else:
                if pos_bias_type == "chatglm_rotary":
                    raise NotImplemented("No FlashAttn version for ChatGLM at present!")
                h_q = h_q.reshape([batch_size * len_q, -1, self.dim_head])  # .permute(0, 2, 1, 3)
                h_k = h_k.reshape([batch_size * len_k, -1, self.dim_head])  # .permute(0, 2, 1, 3)
                h_v = h_v.reshape([batch_size * len_k, -1, self.dim_head])  # .permute(0, 2, 1, 3)
                h_q, h_k = position_bias(
                    h_q, h_k, -3, cu_seqlens=cu_seqlens, max_length=max_seqlen, position_ids=position_ids
                )
                score = prepare_flash_attention(
                    h_q, h_k, h_v, cu_seqlens, max_seqlen, self.dropout_p
                )
                # score = flash_attn_varlen_func(
                #     h_q, h_k, h_v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, self.dropout_p, causal=True
                # )
                # score = OpFlash.apply(
                #     self, not paddle.is_grad_enabled(), h_q, h_k, h_v, cu_seqlens, max_seqlen, self.dropout_p, True
                # )
            # print("batch_size len_q-----------", batch_size, len_q)
            score = score.reshape([batch_size, len_q, -1])
            
        score = self.attention_out(score)
    
        # print("score shape ----------------", score.shape, use_cache)
        if use_cache:
            return score, (h_k, h_v)
        else:
            return score
