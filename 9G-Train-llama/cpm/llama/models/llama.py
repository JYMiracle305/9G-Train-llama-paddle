from typing import List
from typing import Optional
from typing import Tuple

import bmtrain_paddle as bmt
import paddle
import paddle.nn.functional as F
from typing_extensions import TypedDict

from ...layers import Embedding
from ...layers import Encoder
from ...layers import RotaryEmbeddingESM
from ...utils import Config
from ...utils import gradient_shrink


class LlamaInferenceState(TypedDict):
    buffer_context: paddle.Tensor
    buffer_sample_ids: paddle.Tensor
    buffer: List[Tuple[paddle.Tensor, paddle.Tensor]]


class LlamaConfig(Config):
    def __init__(
        self,
        vocab_size=32000,
        dim_model=4096,
        num_heads=32,
        num_kv_heads=32,
        dim_head=128,
        dim_ff=11008,
        num_layers=32,
        dropout_p=0.0,
        activate_fn="silu",
        scale=False,
        eps=1e-5,
        half: bool = True,
        bf16: bool = False,
        mask_modules: Optional[List[Tuple[bool, bool]]] = None,
        use_flash_attn: bool = True,
        flash_attn_mask_shape="1d",
        flash_impl="cuda",
        base=10000,
        tp=0,
        disabled_checkpoint=None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.dim_head = dim_head
        self.dim_ff = dim_ff
        self.num_layers = num_layers
        self.dropout_p = dropout_p
        self.activate_fn = activate_fn
        self.scale = scale
        self.eps = eps
        if half:
            if bf16:
                self.dtype = paddle.bfloat16
            else:
                self.dtype = paddle.float16
        else:
            self.dtype = paddle.float32
        self.flash_impl = flash_impl
        self.mask_modules = mask_modules
        self.use_flash_attn = use_flash_attn
        self.flash_attn_mask_shape = flash_attn_mask_shape
        self.base = base
        self.tp = tp
        self.disabled_checkpoint = disabled_checkpoint


class Llama(bmt.DistributedModule):
    def __init__(self, config: LlamaConfig):
        super().__init__()

        self.encoder = Encoder(
            num_layers=config.num_layers,
            dim_model=config.dim_model,
            dim_ff=config.dim_ff,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            dim_head=config.dim_head,
            activate_fn=config.activate_fn,
            dtype=config.dtype,
            eps=config.eps,
            dropout_p=config.dropout_p,
            scale=config.scale,
            mask_modules=config.mask_modules,
            use_flash_attn=config.use_flash_attn,
            tp=config.tp,
            disabled_checkpoint=config.disabled_checkpoint,
        )

        self.input_embedding = Embedding(
            vocab_size=config.vocab_size,
            embedding_size=config.dim_model,
            scale=config.scale,
            dtype=config.dtype,
            init_std=0.02,
        )

        self.position_bias = RotaryEmbeddingESM(
            dim=config.dim_head, dtype=config.dtype, base=config.base, persistent=False, mixed_precision=True
        )

        self.lm_head = Embedding(
            vocab_size=config.vocab_size,
            embedding_size=config.dim_model,
            scale=config.scale,
            dtype=config.dtype,
            init_std=0.02,
        )
        self.flash_impl = config.flash_impl
        self.use_flash_attn = config.use_flash_attn
        self.flash_attn_mask_shape = config.flash_attn_mask_shape
        self.config = config

    def forward(
        self,
        input: paddle.Tensor,  # (batch, seqlen) int32
        length: paddle.Tensor = None,  # (batch) int32
        context: paddle.Tensor = None,  # (batch, seqlen) bool
        span: paddle.Tensor = None,  # (batch, seqlen) int32
        cu_seqlens: paddle.Tensor = None,  # (real_batch+2) int32
        max_seqlen: int = None,
        position_ids: paddle.Tensor = None,  # (batch, seqlen) int32
    ):
        # bmt.print_rank("----Llama forward start 1", input.shape, flush=True)
        # bmt.print_rank("----Llama forward start------", input, flush=True)
        batch = input.shape[0]
        seqlen = input.shape[1]
        if length is not None and length.dim() == 1:
            length = paddle.arange(seqlen)[None, :].repeat(batch, 1) < length[:, None]

        # processing masks and position bias bucket
        if not self.use_flash_attn or (self.flash_attn_mask_shape == "2d" and self.flash_impl == "triton"):
            with paddle.no_grad():
                # directional mask
                directional_mask_2d = paddle.arange(seqlen) <= paddle.arange(seqlen).reshape(
                    [-1, 1]
                )
                # context mask
                attention_mask = context[:, None, :] | (
                    context[:, :, None].logical_not() & directional_mask_2d.reshape([1, seqlen, seqlen])
                )
                # span mask
                attention_mask = attention_mask & (span[:, None, :] == span[:, :, None])
                # length mask
                attention_mask = length.reshape([batch, seqlen, 1]) & length.reshape([batch, 1, seqlen]) & attention_mask
        else:
            attention_mask = None

        hidden_states = self.input_embedding(input)

        # 添加嵌入层输出监控
        # def embedding_hook(grad):
        #     print(f"\n===== 嵌入层输出梯度 ====="
        #           f"\n形状: {grad.shape}"
        #           f"\n非零比例: {100*(grad != 0).cast('float32').mean().item():.2f}%"
        #           f"\n范围: [{grad.min().item():.6e}, {grad.max().item():.6e}]")
        #     return grad
            
        # hidden_states.register_hook(embedding_hook)

        # print("after enmbedding-----------", hidden_states)
        if self.config.tp:
            with paddle.no_grad():
                if length is not None:
                    length = bmt.distributed.all_gather(length, comm=bmt.config["tp_comm"]).flatten(0, 1)
                if attention_mask is not None:
                    attention_mask = bmt.distributed.all_gather(attention_mask, comm=bmt.config["tp_comm"]).flatten(
                        0, 1
                    )
                if cu_seqlens is not None:
                    lens = bmt.distributed.all_gather(
                        paddle.tensor([cu_seqlens.numel().item()]).cuda(), comm=bmt.config["tp_comm"]
                    )
                    mx = lens.max().item()
                    cu_seq = paddle.zeros(mx).int().cuda()
                    cu_seq[: cu_seqlens.numel().item()] = cu_seqlens
                    all_seq = bmt.distributed.all_gather(cu_seq, comm=bmt.config["tp_comm"])
                    cu_seqlens = [0]
                    for i, l in enumerate(lens):
                        for j in range(1, l - 1):
                            cu_seqlens.append(all_seq[i][j] + i * seqlen)
                        cu_seqlens.append((i + 1) * seqlen)
                    cu_seqlens = paddle.tensor(cu_seqlens, dtype=paddle.int32).cuda()
                if max_seqlen is not None:
                    max_seqlen = bmt.distributed.all_reduce(
                        paddle.tensor([max_seqlen]).cuda(), "max", comm=bmt.config["tp_comm"]
                    )[0].item()
                if position_ids is not None:
                    position_ids = bmt.distributed.all_gather(position_ids, comm=bmt.config["tp_comm"]).flatten(0, 1)

        if self.use_flash_attn:
            if self.flash_attn_mask_shape == "1d":
                hidden_states = self.encoder(
                    hidden_states,
                    attention_mask=None,
                    position_bias=self.position_bias,
                    pos_bias_type="rotary",
                    length_mask=length,
                )
            else:
                if self.flash_impl == "triton":
                    mask = attention_mask.unsqueeze(dim=1).contiguous()
                    attention_mask_bias = paddle.zeros_like(mask, dtype=paddle.float16).cuda()
                    attention_mask_bias[mask == False] -= paddle.inf
                else:
                    attention_mask_bias = None
                    assert cu_seqlens is not None, "cu_seqlens are needed in Flash Attention cuda impl"
                hidden_states = self.encoder(
                    hidden_states,
                    attention_mask=None,
                    position_bias=self.position_bias,
                    pos_bias_type="rotary",
                    length_mask=None,
                    attention_mask_bias=attention_mask_bias,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    position_ids=position_ids,
                )
                # print("Llama OK 1--------------", hidden_states)
        else:
            hidden_states = self.encoder(
                hidden_states, attention_mask=attention_mask, position_bias=self.position_bias, pos_bias_type="rotary"
            )

        logits = self.lm_head.projection(hidden_states)
        # print("Llama OK 2--------------", logits)

        # def logits_hook(grad):
        #     print(f"\n===== 模型输出logits梯度 ====="
        #           f"\n形状: {grad.shape}"
        #           f"\n非零比例: {100*(grad != 0).cast('float32').mean().item():.2f}%"
        #           f"\n范围: [{grad.min().item():.6e}, {grad.max().item():.6e}]")
        #     return grad
        
        # # 为输出注册梯度钩子
        # logits.register_hook(logits_hook)

        return logits, hidden_states

    def inference(
        self,
        input: paddle.Tensor,  # (batch, len_q) int32
        length: paddle.Tensor,  # (batch) int32
        context: paddle.Tensor,  # (batch, seqlen) int16
        span: paddle.Tensor,  # (batch, seqlen) int32
        past_key_values: Optional[LlamaInferenceState] = None,
    ) -> Tuple[paddle.Tensor, paddle.Tensor, LlamaInferenceState]:
        batch = input.size(0)
        len_q = input.size(1)
        len_buffer = 0
        if past_key_values is None:
            present_buffer = None
        else:
            present_buffer = past_key_values["buffer"]
            len_buffer = present_buffer[0][0].shape[-2]
        seqlen = len_buffer + len_q
        with paddle.no_grad():
            device = input.device
            if length.dim() == 1:
                length = paddle.arange(seqlen, device=device)[None, :].repeat(batch, 1) < length[:, None]
            directional_mask_2d = paddle.arange(seqlen, device=device) <= paddle.arange(seqlen, device=device).reshape([-1, 1])
            # context mask
            attention_mask = context[:, None, :] | (
                context[:, :, None].logical_not() & directional_mask_2d.reshape([1, seqlen, seqlen])
            )
            # span mask
            attention_mask = attention_mask & (span[:, None, :] == span[:, :, None])
            # length mask
            attention_mask = length.reshape([batch, seqlen, 1]) & length.reshape([batch, 1, seqlen]) & attention_mask

        hidden_states = self.input_embedding(input)

        hidden_states, present_key_values, _ = self.encoder(
            hidden_states,
            attention_mask=attention_mask[:, len_buffer:],
            position_bias=self.position_bias,
            use_cache=True,
            past_key_values=present_buffer,
            pos_bias_type="rotary",
        )

        logits = self.lm_head.projection(hidden_states)

        return (
            logits,
            hidden_states,
            {"buffer": present_key_values},
        )
