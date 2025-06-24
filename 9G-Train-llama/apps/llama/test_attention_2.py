import paddle
import paddle.nn.functional as F
import math

def safe_simple_attention(Q, K, V, dropout=0.0):
    # 输入: Q/K/V [batch, seq_len, heads, head_dim]
    d = Q.shape[-1]
    
    # 三重保护机制
    scores = paddle.matmul(Q, K.transpose([0,1,3,2])) / math.sqrt(d)
    scores = scores.clip(-10.0, 10.0)  # 硬约束
    scores = scores - scores.max(axis=-1, keepdim=True)  # 数值平移
    
    # 稳定的Softmax
    attn = F.softmax(scores, axis=-1)
    if dropout > 0:
        attn = F.dropout(attn, p=dropout)
    
    return paddle.matmul(attn, V)

def prepare_flash_attention(h_q, h_k, h_v, cu_seqlens, dropout_p=0.0):
    """
    优化版的变长序列注意力实现
    避免显式的填充操作，直接处理变长序列
    """
    # 1. 获取维度信息
    total_tokens = h_q.shape[0]
    num_heads = h_q.shape[1]
    head_dim = h_q.shape[2]
    batch_size = cu_seqlens.shape[0] - 1
    
    # 2. 处理空批次
    if batch_size == 0:
        return paddle.zeros_like(h_q)
    
    # 3. 直接计算注意力，避免填充
    # 使用分组计算代替填充
    output = paddle.zeros_like(h_q)
    
    # 4. 按序列分组计算
    for i in range(batch_size):
        start, end = int(cu_seqlens[i].item()), int(cu_seqlens[i+1].item())
        if end <= start:
            continue
            
        # 添加安全约束
        seq_q = h_q[start:end].reshape([end-start, num_heads, head_dim]).unsqueeze(0)
        seq_k = h_k[start:end].reshape([end-start, num_heads, head_dim]).unsqueeze(0)
        seq_v = h_v[start:end].reshape([end-start, num_heads, head_dim]).unsqueeze(0)
        
        # 使用安全版注意力
        attn_result = safe_simple_attention(seq_q, seq_k, seq_v, dropout_p)
        output[start:end] = attn_result.squeeze(0).reshape([end-start, num_heads, head_dim])
    
    # 增强的归一化
    output = output.reshape([-1, num_heads * head_dim])
    output = paddle.nn.LayerNorm(output.shape[-1])(output)
    output = output.clip(-8.0, 8.0)

    return output.reshape([total_tokens, num_heads, head_dim])

# 测试极端输入
h_q = paddle.randn([100, 8, 64]) * 5  # 放大输入
output = prepare_flash_attention(h_q, h_q, h_q, paddle.to_tensor([0, 50, 100]))
print("输出范围:", output.min().item(), output.max().item())  # 应≈[-8,8]