import paddle
import paddle.nn.functional as F

# 定义自定义 PyLayer
class OpLastLinear(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, record, x, weight, bias=None):

        magic_tensor = paddle.to_tensor([3051124])
        # 检查条件判断是否记录

        ctx.save_for_backward(x, weight, bias, magic_tensor)

        print("----------OpLastLinear FORWARD-----------", x.shape, weight.shape)
        return F.linear(x, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        # 确保提取所有保存的张量
        x, weight, bias, magic_tensor = ctx.saved_tensor()
        
        print("magic_tensor is", magic_tensor.item())
        
        grad_x = grad_weight = grad_bias = None
        print(f"OpLastLinear BACKWARD----------- x.shape: {x.shape}, weight.shape: {weight.shape}")
        print(f"OpLastLinear BACKWARD-   {grad_output} {weight}")
        # 计算输入梯度
        if not x.stop_gradient:
            grad_x = paddle.matmul(grad_output, weight.T)
            print("grad_x", grad_x)
        
        # 计算权重梯度
        if not weight.stop_gradient:
            grad_weight = paddle.matmul(
                grad_output.reshape([-1, grad_output.shape[-1]]).T,
                x.reshape([-1, x.shape[-1]])
            )
            print("grad_weight", grad_weight)
        
        # 计算偏置梯度
        if bias is not None and not bias.stop_gradient:
            grad_bias = grad_output.reshape([-1, grad_output.shape[-1]]).sum(0)
            print("grad_bias", grad_bias)

        # print(f"---------梯度形状: grad_x={grad_x.shape if grad_x else None}, grad_weight={grad_weight.shape if grad_weight else None}")
        return grad_x, grad_weight

# 创建测试对象
class TestLayer:
    def __init__(self):
        self._layer_dict = {}
        self.out_features = 4096  # 设置输出维度

# 创建模拟输入 (float16)
input_tensor = paddle.randn([1, 16384, 11008], dtype='float16')
input_tensor.stop_gradient = False
weight = paddle.create_parameter([11008, 4096], dtype='float16')
bias = None  # 根据要求设置bias=None

# 创建测试对象实例
test_layer = TestLayer()

# 运行前向传播 (record=True)
print("\n===== Record=True 模式 =====")
output = OpLastLinear.apply(True, input_tensor, weight, bias)
print("输出形状:", output.shape)

# 计算损失
loss = output.sum()
print("-----------loss-----------", loss)
# 反向传播
loss.backward()

# print("梯度返回类型:", [g.shape if g is not None else None for g in [grad_input, grad_weight, grad_bias]])

# 测试record=False模式
# print("\n===== Record=False 模式 =====")
# output = OpLastLinear.apply(test_layer, False, input_tensor, weight, bias)
# print("输出形状:", output.shape)  # 应返回全零张量