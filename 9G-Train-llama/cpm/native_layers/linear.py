import math

import paddle
import paddle.nn.functional as F


class Linear(paddle.autograd.PyLayer):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        dtype: paddle.dtype = paddle.float16,
        init_mean: float = 0.0,
        init_std: float = 1,
        scale: bool = True,
        scale_before: bool = False,
    ):
        super().__init__()
        self.dim_in = self.in_features = dim_in
        self.dim_out = self.out_features = dim_out
        self.scale = scale
        self.scale_before = scale_before
        self.weight = paddle.create_parameter(shape = [dim_out, dim_in], dtype=dtype,
                                              default_initializer=paddle.nn.initializer.XavierNormal())

    def forward(self, x: paddle.Tensor):
        """
        Args:
            x (:obj:`paddle.Tensor` of shape ``(batch, seq_len, dim_in)``): The input of linear layer
        Returns:
            :obj:`paddle.Tensor` of shape ``(batch, seq_len, dim_out)``: The output of the linear transform y.
        """  # noqa: E501
        if self.scale:
            if self.scale_before:
                x = x / math.sqrt(self.dim_in)

                x = F.linear(x, self.weight)
            else:
                x = F.linear(x, self.weight)
                x = x / math.sqrt(self.dim_in)

        else:
            x = F.linear(x, self.weight)
        return x
