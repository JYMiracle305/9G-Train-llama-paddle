import paddle


def rms_layernorm(hidden: paddle.Tensor, weight: paddle.Tensor, eps: float):
    old_dtype = hidden.dtype
    variance = hidden.to(paddle.float32).pow(2).mean(dim=-1, keepdim=True)
    hidden = (hidden * paddle.rsqrt(variance + eps)).to(old_dtype)
    return hidden * weight


class LayerNorm(paddle.autograd.PyLayer):
    """RMS LayerNorm"""

    def __init__(
        self,
        dim_norm: int,
        dtype: paddle.dtype = paddle.float16,
        eps: float = 1e-6,
        init_var: float = 1.0,
    ):

        super().__init__()

        self.eps = eps
        self.dim_norm = dim_norm
        self.weight = paddle.nn.parameter.Parameter(paddle.full((dim_norm,), init_var, dtype=dtype))

    def forward(self, x: paddle.Tensor):
        """
        Args:
            x (:obj:`paddle.Tensor` of shape ``(batch_size, seq_len, dim_norm)``): Input tensor that need to be normalized.
        Return:
            :obj:`paddle.Tensor` of shape ``(batch_size, seq_len, dim_norm)``: The layernorm output.
        """
        assert x.size(-1) == self.dim_norm
        return rms_layernorm(x, self.weight, self.eps)
