import paddle
import numpy as np

# 模拟输入数据
data = {
    "inputs": np.random.randint(0, 100, size=(10, 20)),  # 随机生成一些整数数据
    "length": np.random.randint(1, 20, size=(10,)),      # 随机生成一些长度数据
    "target": np.random.randint(0, 100, size=(10, 20)),  # 随机生成一些目标数据
    "task_ids": np.random.randint(0, 10, size=(10,)),    # 随机生成一些任务ID
    "task_names": ["task1", "task2"],                    # 示例任务名称
    "cu_seqlens": np.random.randint(0, 100, size=(10,)), # 随机生成一些序列长度数据
    "max_seqlen": 20,                                    # 最大序列长度
    "position_ids": np.random.randint(0, 20, size=(10, 20)), # 随机生成位置ID
    "spans": np.random.randint(0, 10, size=(10, 20))     # 随机生成跨度数据
}

# 模拟参数
args = {
    "flash": "cuda"  # 模拟参数，控制是否使用 flash 模式
}

# 检查 GPU 是否可用
if paddle.is_compiled_with_cuda():
    place = paddle.CUDAPlace(0)
    print("Using GPU device")
else:
    place = paddle.CPUPlace()
    print("Using CPU device")

# 打印输入数据的前几项
print("----------------", data["inputs"][:10], data["length"])
input_ids = paddle.to_tensor(data["inputs"], dtype=paddle.int32, place=place)
print("input_ids[:10]:", input_ids[:10])
input_length = paddle.to_tensor(data["length"], dtype=paddle.int32, place=place)
targets = paddle.to_tensor(data["target"], dtype=paddle.int32, place=place)
task_ids = paddle.to_tensor(data["task_ids"], dtype=paddle.int32, place=place)
task_names = data["task_names"]
print("OKKK1")

if args["flash"] == "cuda":
    cu_seqlens = paddle.to_tensor(data["cu_seqlens"], dtype=paddle.int32).cuda()
    print("cu_seqlens[:10]:", cu_seqlens[:10], cu_seqlens.dim())

    max_seqlen = data["max_seqlen"]
    print("max_seqlen:", max_seqlen)
    position_ids = paddle.to_tensor(data["position_ids"], dtype=paddle.int32, place=place)
    print("position_ids[:10]:", position_ids[:10])
else:
    input_ids = paddle.to_tensor(data["inputs"], dtype=paddle.int32, place=place)
    input_context = paddle.zeros_like(input_ids, dtype=paddle.bool, place=place)
    input_span = paddle.to_tensor(data["spans"], dtype=paddle.int32, place=place)

print("Test completed successfully!")