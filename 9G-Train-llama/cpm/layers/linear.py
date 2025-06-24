import inspect
import math

import bmtrain_paddle as bmt
import paddle
import paddle.nn.functional as F


def Linear(*args, **kwargs):
    tp = kwargs.pop("tp", 0)
    if tp == 0:
        return NormalLinear(*args, **kwargs)
    if tp == 1:
        return ColumnParallelLinear(*args, **kwargs)
    if tp == 2:
        return RowParallelLinear(*args, **kwargs)


class OpLastLinear(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, self, record, x, weight, bias=None):
        ctx.self = self
        magic_tensor = paddle.to_tensor([3051124])
        if not record and "r" in self._layer_dict:
            ctx.save_for_backward(x, weight, bias, magic_tensor)
            self._layer_dict.pop("r")
            return paddle.zeros((*x.shape[:-1], self.out_features), device=x.device, dtype=x.dtype)
        else:
            ctx.save_for_backward(x, weight, bias, magic_tensor)
            if record:
                self._layer_dict["r"] = True
            # print("----------OpLastLinear-----------", x.shape, weight.shape)
            return F.linear(x, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias, magic_tensor = ctx.saved_tensor()
        # print("magic_tensor is", magic_tensor)
        grad_x = grad_weight = grad_bias = None
        # print("OpLastLinear backward-------------", bias, x.shape, weight.shape)
        if not x.stop_gradient:
            grad_x = grad_output.matmul(weight.T)
        if not weight.stop_gradient:
            grad_weight = grad_output.reshape([-1, grad_output.shape[-1]]).t().matmul(x.reshape([-1, x.shape[-1]]))
        if bias is not None and not bias.stop_gradient:
            grad_bias = grad_output.reshape([-1, grad_output.shape[-1]]).sum(0)
        # print("---------梯度-------------", grad_x, grad_weight)
        if bias is None:
            # print(f"return 2 grad_x {grad_x} grad_weight {grad_weight}")
            return grad_x, grad_weight
        else:
            # print("return 3")
            return grad_x, grad_weight, grad_bias


class LastLinear(bmt.DistributedModule):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        bias: bool = False,
        dtype: paddle.dtype = paddle.float16,
        init_mean: float = 0.0,
        init_std: float = 1,
        scale: bool = True,
        scale_before: bool = False,
        tp: int = 0,
    ):
        super().__init__()
        self.dim_in = self.in_features = dim_in
        self.dim_out = self.out_features = dim_out
        self.scale = scale
        self.scale_before = scale_before

        if not scale:
            init_std = 1 / ((dim_in + dim_out) ** 0.5)

        self.weight = bmt.DistributedParameter(
            paddle.empty([dim_in, dim_out], dtype=dtype),
            init_method=paddle.nn.initializer.XavierNormal(),
        )
        self.bias = (
            bmt.DistributedParameter(
                paddle.empty([dim_out], dtype=dtype),
                init_method=paddle.nn.initializer.Constant(0.0),
            )
            if bias
            else None
        )
        self._layer_dict = {}

    def forward(self, x: paddle.Tensor):
        """
        Args:
            x (:obj:`torch.Tensor` of shape ``(batch, seq_len, dim_in)``): The input of linear layer
        Returns:
            :obj:`torch.Tensor` of shape ``(batch, seq_len, dim_out)``: The output of the linear transform y.
        """  # noqa: E501
        if self.scale and self.scale_before:
            x = x / math.sqrt(self.dim_in)
        # x = OpLastLinear.apply(self, not paddle.is_grad_enabled(), x, self.weight, self.bias)
        x = F.linear(x, self.weight, self.bias)
        if self.scale and not self.scale_before:
            x = x / math.sqrt(self.dim_in)
        return x


class NormalLinear(bmt.DistributedModule):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        bias: bool = False,
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
        if not scale:
            init_std = 1 / ((dim_in + dim_out) ** 0.5)

        self.weight = bmt.DistributedParameter(
            paddle.empty([dim_in, dim_out], dtype=dtype),
            init_method=paddle.nn.initializer.XavierNormal(),
        )
        self.bias = (
            bmt.DistributedParameter(
                paddle.empty([dim_out], dtype=dtype),
                init_method=paddle.nn.initializer.Constant(0.0),
            )
            if bias
            else None
        )

    def forward(self, x: paddle.Tensor):
        """
        Args:
            x (:obj:`torch.Tensor` of shape ``(batch, seq_len, dim_in)``): The input of linear layer
        Returns:
            :obj:`torch.Tensor` of shape ``(batch, seq_len, dim_out)``: The output of the linear transform y.
        """  # noqa: E501
        if self.scale and self.scale_before:
            x = x / math.sqrt(self.dim_in)
        if "tp_size" in inspect.signature(bmt.init_distributed).parameters:
            # print("NormalLinear---------", x, self.weight, self.bias)
            x = bmt.nn.OpLinear.apply(x, self.weight, self.bias)
        else:
            x = F.linear(x, self.weight, self.bias)
        if self.scale and not self.scale_before:
            x = x / math.sqrt(self.dim_in)
        return x


class ColumnParallelLinear(bmt.DistributedModule):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        bias: bool = False,
        dtype: paddle.dtype = paddle.float16,
        init_mean: float = 0.0,
        init_std: float = 1,
        scale: bool = True,
        scale_before: bool = False,
        gather_output=False,
        gather_input=True,
    ):
        super().__init__()
        assert dim_out % bmt.config["tp_size"] == 0
        if not scale:
            init_std = 1 / ((dim_in + dim_out) ** 0.5)
        dim_out = dim_out // bmt.config["tp_size"]
        self.dim_in = self.in_features = dim_in
        self.dim_out = self.out_features = dim_out
        self.scale = scale
        self.scale_before = scale_before
        self.gather_input = gather_input
        self.gather_output = gather_output

        self.weight = bmt.DistributedParameter(
            paddle.empty((dim_out, dim_in), dtype=dtype),
            init_method=paddle.nn.initializer.XavierNormal(),
            tp_split_dim=0,
            tp_mode=True,
        )
        self.bias = (
            bmt.DistributedParameter(
                paddle.empty(dim_out, dtype=dtype),
                init_method=paddle.nn.initializer.Constant(0.0),
                tp_split_dim=0,
                tp_mode=True,
            )
            if bias
            else None
        )

    def forward(self, x: paddle.Tensor):
        """
        Args:
            x (:obj:`torch.Tensor` of shape ``(batch, seq_len, dim_in)``): The input of linear layer
        Returns:
            :obj:`torch.Tensor` of shape ``(batch, seq_len, dim_out)``: The output of the linear transform y.
        """  # noqa: E501
        if self.scale and self.scale_before:
            x = x / math.sqrt(self.dim_in)
        x = bmt.nn.OpParallelLinear.apply(x, self.weight, self.bias, self.gather_input, self.gather_output, False, None)
        if self.scale and not self.scale_before:
            x = x / math.sqrt(self.dim_in)
        return x


class RowParallelLinear(bmt.DistributedModule):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        bias: bool = False,
        dtype: paddle.dtype = paddle.float16,
        init_mean: float = 0.0,
        init_std: float = 1,
        scale: bool = True,
        scale_before: bool = False,
        split_input=False,
        all_reduce_output=False,
    ):
        super().__init__()
        assert dim_in % bmt.config["tp_size"] == 0
        if not scale:
            init_std = 1 / ((dim_in + dim_out) ** 0.5)
        dim_in = dim_in // bmt.config["tp_size"]
        self.dim_in = self.in_features = dim_in
        self.dim_out = self.out_features = dim_out
        self.scale = scale
        self.scale_before = scale_before
        self.split_input = split_input
        self.all_reduce_output = all_reduce_output

        self.weight = bmt.DistributedParameter(
            paddle.empty((dim_out, dim_in), dtype=dtype),
            init_method=paddle.nn.initializer.XavierNormal(),
            tp_split_dim=1,
            tp_mode=True,
        )
        self.bias = (
            bmt.DistributedParameter(
                paddle.empty(dim_out, dtype=dtype),
                init_method=paddle.nn.initializer.Constant(0.0),
                tp_split_dim=-1,
                tp_mode=True,
            )
            if bias
            else None
        )

    def forward(self, x: paddle.Tensor):
        """
        Args:
            x (:obj:`torch.Tensor` of shape ``(batch, seq_len, dim_in)``): The input of linear layer
        Returns:
            :obj:`torch.Tensor` of shape ``(batch, seq_len, dim_out)``: The output of the linear transform y.
        """  # noqa: E501
        if self.scale and self.scale_before:
            x = x / math.sqrt(self.dim_in)
        x = bmt.nn.OpParallelLinear.apply(
            x, self.weight, None, self.split_input, False, self.split_input, 1 if self.all_reduce_output else 2
        )
        if self.bias is not None:
            x = x + self.bias
        if self.scale and not self.scale_before:
            x = x / math.sqrt(self.dim_in)
        return x
