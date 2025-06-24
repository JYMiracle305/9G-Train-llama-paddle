import paddle

# 假设 seq_lens 是一个 PaddlePaddle 张量
seq_lens = paddle.to_tensor([2, 0, 881, 993, 1364, 1419, 1897, 2613, 2942, 3568,
                             4201, 4510, 4915, 5449, 5571, 6009, 6206, 6472, 6501, 6701,
                             7380, 7670, 8165, 8725, 9027, 9723, 10318, 10352, 10838, 11090,
                             11721, 11773, 12384, 12455, 12500, 12793, 13258, 13836, 14543, 14987,
                             15479, 15539, 15817, 15934, 16213], dtype='int32')

# 打印关键信息
print("seq_lens:", seq_lens)
print("seq_lens.shape:", seq_lens.shape)
print("seq_lens.dtype:", seq_lens.dtype)

# 计算最大值
max_value = seq_lens.max()
print("seq_lens.max():", max_value)

# 转换为 Python 标量
max_seqlen = max_value.item() if seq_lens.size > 0 else 0
print("max_seqlen:", max_seqlen)

seq_lens_cpu = seq_lens.cpu()
max_value = seq_lens_cpu.max()
print("seq_lens_cpu.max():", max_value)