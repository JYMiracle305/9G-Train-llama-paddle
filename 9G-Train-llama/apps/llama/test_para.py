import paddle
import paddle.nn as nn

# 创建一个嵌入层
embedding_dim = 10
vocab_size = 100
embedding = nn.Embedding(vocab_size, embedding_dim)

# 使用 XavierNormal 初始化权重
xavier_initializer = nn.initializer.XavierNormal()
xavier_initializer(embedding.weight)

# 查看初始化后的权重
print("使用 XavierNormal 初始化后的权重：")
print(embedding.weight.numpy())

# 创建一个输入索引
input_indices = paddle.to_tensor([1, 2, 3])

# 获取嵌入结果
embedded = embedding(input_indices)
print("嵌入结果：")
print(embedded.numpy())

# 将权重设置为全 0
embedding.weight.set_value(paddle.zeros_like(embedding.weight))

# 再次获取嵌入结果
embedded_zero = embedding(input_indices)
print("权重为全 0 时的嵌入结果：")
print(embedded_zero.numpy())