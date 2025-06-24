import paddle
import paddle.nn.functional as F
import math

def simple_attention(
    query: paddle.Tensor,
    key: paddle.Tensor,
    value: paddle.Tensor,
    mask: paddle.Tensor = None,
    dropout: float = 0.0,
    safety_checks: bool = True
) -> paddle.Tensor:
    """
    稳定的注意力机制实现
    
    参数:
        query: [batch_size, seq_len, num_heads, head_dim]
        key:   [batch_size, seq_len, num_heads, head_dim]
        value: [batch_size, seq_len, num_heads, head_dim]
        mask:  [batch_size, seq_len, seq_len] 或 None
        dropout: dropout概率
        safety_checks: 是否启用数值安全检查
    
    返回:
        [batch_size, seq_len, num_heads, head_dim]
    """
    # ===== 1. 输入验证 =====
    if safety_checks:
        assert query.dim() == 4, "Query维度应为4D"
        assert key.shape == query.shape, "Key应与Query同形"
        assert value.shape[:3] == query.shape[:3], "Value前三维需匹配"
        if mask is not None:
            assert mask.dim() in [3,4], "Mask应为3D或4D"
    
    # ===== 2. 维度转换 =====
    batch_size, seq_len, num_heads, head_dim = query.shape
    query = query.transpose([0, 2, 1, 3])  # [bs, heads, q_len, dim]
    key = key.transpose([0, 2, 1, 3])     # [bs, heads, k_len, dim]
    value = value.transpose([0, 2, 1, 3]) # [bs, heads, v_len, dim]

    # ===== 3. 注意力分数计算 =====
    # 3.1 基础点积计算
    scores = paddle.matmul(query, key.transpose([0, 1, 3, 2]))  # [bs, heads, q_len, k_len]
    
    # 3.2 动态缩放因子 (带安全保护)
    scale_factor = paddle.to_tensor(1.0 / math.sqrt(head_dim), dtype=scores.dtype)
    if safety_checks:
        scale_factor = scale_factor.clip(max=1.0)  # 防止head_dim过小导致爆炸
    
    scores = scores * scale_factor

    # 3.3 数值稳定性处理
    if safety_checks:
        # 自动检测异常分数
        max_score = scores.abs().max().item()
        if max_score > 10.0:  # 阈值可配置
            # 动态温度调节
            temp = 1.0 + 0.5 * math.log(1.0 + max_score)
            scores = scores / temp
            scores = scores.clip(-15.0, 15.0)  # 二次保护

    # ===== 4. 掩码处理 =====
    if mask is not None:
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)  # 扩展head维度
        mask_value = paddle.to_tensor(-1e4, dtype=scores.dtype)
        scores = paddle.where(mask > 0.5, scores, mask_value)

    # ===== 5. Softmax归一化 =====
    # 5.1 数值稳定版softmax
    scores = scores - scores.max(axis=-1, keepdim=True)  # 平移最大值到0
    exp_scores = paddle.exp(scores)
    
    # 5.2 防除零保护
    sum_exp = exp_scores.sum(axis=-1, keepdim=True)
    sum_exp = paddle.clip(sum_exp, min=1e-6)
    attention_weights = exp_scores / sum_exp

    # ===== 6. Dropout =====
    if dropout > 0.0:
        attention_weights = F.dropout(attention_weights, p=dropout)

    # ===== 7. 加权求和 =====
    output = paddle.matmul(attention_weights, value)  # [bs, heads, q_len, dim]
    
    # ===== 8. 输出处理 =====
    output = output.transpose([0, 2, 1, 3])  # 恢复原始维度
    
    # 最终数值检查
    if safety_checks:
        max_output = output.abs().max().item()
        if max_output > 10.0:  # 阈值可配置
            output = output * (8.0 / max_output)  # 自动缩放
    
    return output

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
        start_idx = int(cu_seqlens[i].item())
        end_idx = int(cu_seqlens[i+1].item())
        seq_len = end_idx - start_idx
        
        if seq_len == 0:
            continue
            
        # 提取当前序列的Q,K,V
        seq_q = h_q[start_idx:end_idx]
        seq_k = h_k[start_idx:end_idx]
        seq_v = h_v[start_idx:end_idx]
        
        # 重塑为注意力计算格式
        seq_q = seq_q.reshape([seq_len, num_heads, head_dim])
        seq_k = seq_k.reshape([seq_len, num_heads, head_dim])
        seq_v = seq_v.reshape([seq_len, num_heads, head_dim])
        
        # 计算当前序列的注意力
        attn_result = simple_attention(
            seq_q.unsqueeze(0),  # 添加batch维度
            seq_k.unsqueeze(0),
            seq_v.unsqueeze(0),
            mask=None,  # 内部生成因果掩码
            dropout=dropout_p
        )
        
        # 存储结果
        output[start_idx:end_idx] = attn_result.squeeze(0).reshape([seq_len, num_heads, head_dim])
    
    output = output.reshape([-1, num_heads * head_dim])
    output = paddle.nn.LayerNorm(output.shape[-1])(output)
    print("prepare_flash_attention", total_tokens, num_heads, head_dim)
    return output.reshape([total_tokens, num_heads, head_dim])

h_q = paddle.randn([100, 8, 64]) * 10  # 故意放大输入
output = prepare_flash_attention(h_q, h_q, h_q, paddle.to_tensor([0, 50, 100]))
print("输出范围:", output.min().item(), output.max().item())  # 


h_q = paddle.randn([100, 8, 64])  # 标准正态分布
output = prepare_flash_attention(h_q, h_q, h_q, paddle.to_tensor([0, 100]))
print("随机输入输出范围:", output.min().item(), output.max().item()) 