import inspect
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from typing import Any
from typing import Dict
from typing import List
from typing import Union

import bmtrain_paddle as bmt
import paddle

sys.path.insert(0, "/home/jiyiming/code/BMTrain_paddle_9G/9G-Train-llama")
from cpm.arguments import get_args
from cpm.llama.models import Llama
from cpm.llama.models import LlamaConfig
from cpm.llama.tokenizers import LlamaTokenizer
from cpm.llama.training_tasks import MixedDataset
from cpm.utils import allgather_objects
from cpm.utils import exporter
from cpm.utils import logger
from cpm.utils import LogManager

import numpy as np

def get_tokenizer(args):
    # print("get_tokenizer--------------", args.tokenizer_path)
    tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_model(args):
    # print("get_model--------------", args.model_config)
    config = LlamaConfig.from_json_file(args.model_config)
    config.tp = 1 if args.tp != 1 else 0
    if args.flash == "none":
        config.use_flash_attn = False
    else:
        config.use_flash_attn = True
        if args.flash == "1d":
            config.flash_attn_mask_shape = "1d"
        else:
            config.flash_attn_mask_shape = "2d"
            if args.flash == "triton":
                config.flash_impl = "triton"
            elif args.flash == "cuda":
                config.flash_impl = "cuda"
    model = Llama(config)
    if args.load is not None:
        bmt.print_rank("args.load is not None, start to load checkpoints" + args.load)
        bmt.load(model, args.load)
    else:
        bmt.print_rank("args.load is None, start to initialize parameters")
        bmt.init_parameters(model)
    return model


def get_optimizer(args, model):
    # print("------------get_optimizer  OK---------------", args.offload)
    if args.offload:
        optimizer = bmt.optim.AdamOffloadOptimizer(
            model.parameters(), betas=(0.9, 0.95), weight_decay=args.weight_decay
        )
    else:
        optimizer = bmt.optim.AdamOptimizer(model.parameters(), betas=(0.9, 0.95), weight_decay=args.weight_decay)
    # print("-----------------bmt.optim OK")
    if args.load is not None and args.load_grad:
        start = time.time()
        print(
            sum([1 if re.search(r"-{}.rank-\d+.opt".format(args.start_step), i) else 0 for i in os.listdir(args.save)])
        )
        if (
            sum([1 if re.search(r"-{}.rank-\d+.opt".format(args.start_step), i) else 0 for i in os.listdir(args.save)])
            == bmt.world_size()
        ):
            file_name = os.path.join(
                args.save,
                args.save_name + "-{}.rank-{}.opt".format(args.start_step, bmt.rank()),
            )
            print(file_name)
            if os.path.exists(file_name):
                print("start to load grad ckpt {}".format(file_name))
                states = paddle.load(file_name)
                optimizer.load_state_dict(states)
        logger.info("load grad in {:.2f}s".format(time.time() - start))
    return optimizer


class Cosine(bmt.lr_scheduler.WarmupLRScheduler):
    r"""
    After a warmup period during which learning rate increases linearly between 0 and the start_lr,
    The decay period performs :math:`\text{lr}=\text{start_lr}\times \dfrac{1+\cos \left( \pi \cdot \dfrac{\text{num_iter}-\text{warmup_iter}}{\text{end_iter}-\text{warmup_iter}}\right)}{2}`
    """

    def get_lr_warmup(self, num_iter) -> float:
        return self.start_lr * num_iter / self.warmup_iter

    def get_lr_decay(self, num_iter) -> float:
        progress = (num_iter - self.warmup_iter) / max(1, (self.end_iter - self.warmup_iter))
        return max(self.start_lr * 0.1, self.start_lr * (0.1 + 0.45 * (1.0 + math.cos(progress * math.pi))))


def get_learning_rate_scheduler(args, optimizer):
    if args.lr_decay_iters is None:
        args.lr_decay_iters = args.train_iters
    # lr_scheduler = bmt.lr_scheduler.Noam(
    #     optimizer,
    #     start_lr=args.lr,
    #     warmup_iter=args.warmup_iters,
    #     end_iter=args.lr_decay_iters,
    #     num_iter=args.start_step,
    # )
    lr_scheduler = Cosine(
        optimizer,
        start_lr=args.lr,
        warmup_iter=args.warmup_iters,
        end_iter=args.lr_decay_iters,
        num_iter=args.start_step,
    )
    return lr_scheduler


def setup_model_and_optimizer(args):

    start = time.time()
    model = get_model(args)
    logger.info("load model in {:.2f}s".format(time.time() - start))

    start = time.time()
    tokenizer = get_tokenizer(args)
    bmt.synchronize()
    logger.info("load tokenizer in {:.2f}s".format(time.time() - start))

    start = time.time()
    optimizer = get_optimizer(args, model)
    # print("get_optimizer OK-----------------")
    lr_scheduler = get_learning_rate_scheduler(args, optimizer)
    print("setup_model_and_optimizer--------------", lr_scheduler)
    bmt.synchronize()
    logger.info("load lr_scheduler in {:.2f}s".format(time.time() - start))

    return tokenizer, model, optimizer, lr_scheduler


def initialize():
    args = get_args(pretrain=True)
    #bmt.init_distributed(seed=args.seed, zero_level=3)
    bmt.init_distributed(seed=args.seed)
    if args.save is not None:
        os.makedirs(args.save, exist_ok=True)
    if args.load is not None:
        if args.start_step == 0:
            args.start_step = (int)(re.search("(\d+).pt", args.load)[1])
    return args


def see_memory(detail=False):
    if detail:
        res = paddle.device.cuda.max_memory_info()
    else:
        res = (
            round(paddle.device.cuda.max_memory_reserved() / (1024 * 1024 * 1024), 2),
            round(paddle.device.cuda.max_memory_allocated() / (1024 * 1024 * 1024), 2),
        )
    # paddle.device.cuda.reset_memory_stats()   # TODO
    return res


def add_mem_time(info, mem_usage, tim_usage):
    paddle.device.cuda.synchronize()
    bmt.synchronize()
    mem_usage[info] = see_memory()
    tim_usage[info] = time.time()
    return mem_usage, tim_usage


class LossSpikeDetector:
    def __init__(self, log_path: str) -> None:
        self._last_loss: Dict[str, float] = {}
        self._last_data: List[Any] = [None]
        self._log_path = log_path

    def update_data(self, data: Any):
        self._last_data.append(data)
        if len(self._last_data) > 2:
            self._last_data = self._last_data[-2:]

    def update_loss(self, iteration: int, loss_map: Dict[str, float]):
        loss_spike_result = []
        for task, loss in loss_map.items():
            if task in self._last_loss:
                if loss > self._last_loss[task] * 3:
                    # loss spike!
                    loss_spike_result.append(
                        {
                            "prev": self._last_loss[task],
                            "curr": loss,
                            "task": task,
                        }
                    )
            self._last_loss[task] = float(loss)
        if len(loss_spike_result) > 0:
            self._write_log(iteration, self._last_data[-1], loss_spike_result)

    def _write_log(self, iteration: int, data: Any, result: List[Dict[str, Any]]):
        while True:
            try:
                with open(self._log_path, "a", encoding="utf-8") as fp:
                    fp.write("=" * 20)
                    fp.write("\nloss spike at {}\n".format(iteration))
                    fp.write("{}\n".format(json.dumps(result, indent=4, ensure_ascii=False)))
                    fp.write("data: \n")
                    for d in data:
                        fp.write("{}\n".format(json.dumps(d, indent=4, ensure_ascii=False)))
                    fp.write("\n\n")
                    break
            except Exception as e:
                print("cannot output log to the file {}", self._log_path)

def analyze_logits_anomaly(logits, float16_max_safe=10.0, float32_max_safe=80.0, verbose=False):
    """
    分析 Logits 中的异常值情况，特别关注导致 float16 溢出的危险值
    
    参数:
    logits (Tensor): 模型输出的 logits 张量
    float16_max_safe (float): float16 的安全上限（默认 10，超过此值 exp 可能溢出）
    float32_max_safe (float): float32 的安全上限（默认 80，超过此值可能导致下溢）
    verbose (bool): 是否输出详细统计信息
    
    返回:
    dict: 包含异常统计指标的字典
    """
    # 准备收集的结果字典
    stats = {}
    
    # 1. 转换为 float32 确保精度计算
    logits_f32 = logits.cast("float32")
    
    # 2. 计算整体统计量
    stats["mean"] = logits_f32.mean().item()
    stats["std"] = logits_f32.std().item()
    stats["min"] = logits_f32.min().item()
    stats["max"] = logits_f32.max().item()
    
    # 3. 绝对值的范围分析 (核心关注点)
    abs_logits = paddle.abs(logits_f32)
    
    # 标记异常值的掩码
    dangerous_mask = abs_logits > float16_max_safe
    extreme_mask = abs_logits > float32_max_safe
    
    # 异常值计数
    stats["total_elements"] = logits.size * 1.0
    stats["dangerous_count"] = dangerous_mask.sum().item()
    stats["extreme_count"] = extreme_mask.sum().item()
    
    # 异常值占比
    stats["dangerous_ratio"] = stats["dangerous_count"] / stats["total_elements"]
    stats["extreme_ratio"] = stats["extreme_count"] / stats["total_elements"]
    
    print("logits_f32.reshape([-1]) ", logits_f32.reshape([-1]))
    print("logits_f32.reshape([-1].max(0) ", logits_f32.reshape([-1]).max(0))
    # 额外诊断: 最大值的位置
    max_val = logits_f32.reshape([-1]).max(0).item()
    max_idx = logits_f32.reshape([-1]).argmax(0).item()

    # 多维索引计算
    shape = logits.shape
    indices = []
    dim_size = max_idx
    for dim in shape[::-1]:
        indices.append(dim_size % dim)
        dim_size //= dim
    
    stats["max_value"] = max_val
    stats["max_position"] = tuple(indices[::-1])
    
    # 危险区域特征值统计
    if stats["dangerous_count"] > 0:
        dangerous_vals = abs_logits[dangerous_mask]
        stats["dangerous_mean"] = dangerous_vals.mean().item()
        stats["dangerous_min"] = dangerous_vals.min().item()
        stats["dangerous_max"] = dangerous_vals.max().item()
        stats["dangerous_std"] = dangerous_vals.std().item()
    else:
        stats["dangerous_mean"] = 0.0
        stats["dangerous_min"] = 0.0
        stats["dangerous_max"] = 0.0
        stats["dangerous_std"] = 0.0
    
    # 如果 verbose 打开，打印详细报告
    if verbose:
        print(f"\n=== Logits 异常诊断报告 ===")
        print(f"统计范围: 总元素数 {stats['total_elements']:.1e}")
        print(f"整体范围: [{stats['min']:.3f}, {stats['max']:.3f}] | 均值: {stats['mean']:.3f} | 标准差: {stats['std']:.3f}")
        
        if stats["dangerous_count"] > 0:
            print(f"\n⚠️ 危险值检测 (超过安全阈值 {float16_max_safe})")
            print(f"数量: {stats['dangerous_count']} | 占比: {stats['dangerous_ratio']*100:.4f}%")
            print(f"危险值均值: {stats['dangerous_mean']:.3f} | 范围: [{stats['dangerous_min']:.3f}, {stats['dangerous_max']:.3f}]")
            
            if stats["extreme_count"] > 0:
                print(f"\n🔥 极端值检测 (超过极限阈值 {float32_max_safe})")
                print(f"数量: {stats['extreme_count']} | 占比: {stats['extreme_ratio']*100:.6f}%")
            
            print(f"\n最高值位置: {stats['max_position']} | 值: {stats['max_value']:.3f}")
            
            # 异常位置分布分析
            if len(shape) >= 2:
                # 按词汇维度统计异常值分布
                abnormal_token = dangerous_mask.any(axis=tuple(range(1, len(shape))))
                print(f"\n异常样本数: {abnormal_token.sum().item()}/{shape[0]}")

    # 生成危险等级评估
    if stats["extreme_ratio"] > 1e-6:
        stats["risk_level"] = "CRITICAL"
    elif stats["dangerous_ratio"] > 1e-5:
        stats["risk_level"] = "HIGH"
    elif stats["dangerous_ratio"] > 1e-7:
        stats["risk_level"] = "MODERATE"
    else:
        stats["risk_level"] = "SAFE"
    
    return stats

def analyze_logits(logits, threshold=10.0):
    logits_f32 = logits.astype('float32')
    max_val = logits_f32.max().item()
    min_val = logits_f32.min().item()
    mean_val = logits_f32.mean().item()
    
    print(f"📊 Logits 统计: max={max_val:.2f}, min={min_val:.2f}, mean={mean_val:.2f}")
    
    if max_val > threshold or min_val < -threshold:
        print(f"⚠️ Logits 超出安全范围 [-{threshold}, {threshold}]")
    
    if paddle.isnan(logits).any() or paddle.isinf(logits).any():
        print("💥 Logits 包含 NaN/Inf！")

import re

def register_activation_hooks(model):
    activation_stats = {}
    
    def hook(layer_name):
        def forward_hook(module, input, output):
            activation_stats[layer_name] = {
                "max": output.abs().max().item(),
                "mean": output.mean().item(),
                "std": output.std().item()
            }
            if output.abs().max() > 10.0:
                print(f"🔥 激活异常: {layer_name} | max={output.abs().max().item():.2f}")
        return forward_hook
    
    # 定义目标层（支持通配符 *）
    target_patterns = [
        "encoder.transformers.*.self_att.self_attention.project_*",
        "encoder.transformers.*.self_att.self_attention.attention_out",
        "encoder.transformers.*.ffn.ffn.w_*",
        "encoder.transformers.*.self_att.layernorm_before_attention",
        "encoder.transformers.*.ffn.layernorm_before_ffn",
        "encoder.output_layernorm",
        "input_embedding"
    ]
    
    # 遍历所有层，匹配目标模式
    for name, layer in model.named_sublayers():
        for pattern in target_patterns:
            # 将通配符 * 转换为正则表达式 .*
            regex = re.compile(pattern.replace("*", ".*"))
            if regex.fullmatch(name):
                layer.register_forward_post_hook(hook(name))
                break  # 匹配到一个模式即可
    
    return activation_stats

def pretrain(
    args,
    tokenizer: LlamaTokenizer,
    model: Llama,
    optimizer,
    lr_scheduler: bmt.lr_scheduler.WarmupLRScheduler,
):
    print("================start pretrain================")
    average_time = bmt.utils.AverageRecorder()
    loss_func = bmt.loss.FusedCrossEntropy(ignore_index=-100)
    optim_manager = bmt.optim.OptimManager(
        loss_scale=None if args.bf16 else args.loss_scale,
        loss_scale_steps=args.loss_scale_steps,
        loss_scale_factor=2,
        max_loss_scale=args.max_loss_scale,
        min_loss_scale=args.min_loss_scale,
    )
    optim_manager.add_optimizer(optimizer, lr_scheduler)

    start_step = args.start_step
    #lsd = LossSpikeDetector("/data/logs/debug/spile.%d.log" % bmt.rank())
    print("start pretrain 22")
    if args.tensorboard is not None and bmt.rank() == 0:
        import distutils.version  # noqa: F401

        from tensorboardX import SummaryWriter

        if not os.path.exists(args.tensorboard):
            os.makedirs(args.tensorboard)
        writer = SummaryWriter(log_dir=args.tensorboard)

    if args.log_dir is not None and bmt.rank() == 0:
        log_mgr = LogManager(args.log_dir)

    global_token_pass = 0.0
    global_world_size = bmt.world_size()

    dataloader = MixedDataset(args.dataset, args.batch_size, args.max_length, tokenizer, unpad=(args.flash == "cuda"))
    if args.load is not None:
        dataset_states_path = args.load.replace(".pt", ".data")
        if os.path.exists(dataset_states_path):
            start = time.time()
            bmt.print_rank("start to load data ckpt")
            dataset_states = paddle.load(dataset_states_path)
            logger.info("load data ckpt in {:.2f}s".format(time.time() - start))

            start = time.time()
            missing = dataloader.load_state_dict(dataset_states)
            logger.info("load state dict in {:.2f}s".format(time.time() - start))
            if len(missing) > 0:
                bmt.print_rank("Missing keys when loading dataset states: ", missing)
        else:
            bmt.print_rank("cannot find data ckpt {}".format(dataset_states_path))
    print("start pretrain 55")
    dataloader.start()
    bmt.print_rank("finish dataset start")
    try:
        total = 0
        hash = {}
        for iteration, data in enumerate(dataloader):
            iteration = iteration + start_step + 1
            print("----------------", data["inputs"], data["length"])
            print("data['inputs'].shape:", data["inputs"].shape)
            input_ids = paddle.to_tensor(data["inputs"], dtype=paddle.int32).cuda()
            # print("input_ids ", input_ids[:10])
            input_length = paddle.to_tensor(data["length"], dtype=paddle.int32).cuda()
            targets = paddle.to_tensor(data["target"], dtype=paddle.int32).cuda()
            print("targets shape:", targets.shape)  # 打印张量的形状
            # print("targets length:", len(targets))
            # print("------------targets--------------", targets[0, 5000:5101].numpy())
            task_ids = paddle.to_tensor(data["task_ids"], dtype=paddle.int32).cuda()
            task_names = data["task_names"]
            #lsd.update_data(data["raw_data"])
            if args.flash == "cuda":
                cu_seqlens = paddle.to_tensor(data["cu_seqlens"], dtype=paddle.int32).cuda()
                max_seqlen = data["max_seqlen"]
                position_ids = paddle.to_tensor(data["position_ids"], dtype=paddle.int32).cuda()
            else:
                input_ids = paddle.to_tensor(data["inputs"], dtype=paddle.int32).cuda()
                input_context = paddle.zeros_like(input_ids).cuda().bool()
                input_span = paddle.to_tensor(data["spans"], dtype=paddle.int32).cuda()

            # ===========
            optim_manager.zero_grad()
            # torch.cuda.empty_cache()
            mem_usage = {}
            tim_usage = {}
            mem_usage, tim_usage = add_mem_time("init", mem_usage, tim_usage)

            # bmt.print_rank(torch.cuda.max_memory_allocated())

            # ===========
            if args.flash == "cuda":
                # print("param 1 ", input_ids, cu_seqlens, max_seqlen, position_ids)
                logits, _ = model(
                    input_ids,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    position_ids=position_ids,
                )
            else:
                # print("param 2 ", input_ids)
                logits, _ = model(
                    input_ids,
                    input_length,
                    input_context,
                    input_span,
                )
                #tmp_cpu = logits.cpu().detach()
                #havNan = np.isnan(np.array(tmp_cpu)).any() 
            print("========================OKKKK4======================")
            mem_usage, tim_usage = add_mem_time("forward_1", mem_usage, tim_usage)
            # print("~~~~~~~~~logits.reshape([-1, logits.shape[-1]]), targets.reshape([-1])~~~~~~~~~", logits.reshape([-1, logits.shape[-1]]), targets.reshape([-1]))
            loss = loss_func(logits.reshape([-1, logits.shape[-1]]).astype(paddle.float32), targets.reshape([-1]))

            # print(f"loss_vec 形状: {loss.shape}")
            # print(f"loss_vec 均值: {loss.mean().item()}")
            # print(f"loss_vec: {loss}")
            # loss.register_hook(lambda grad: print(f"\n===== 损失梯度 ====="
            #                              f"\n形状: {grad.shape}"
            #                              f"\n非零值比例: {100*(grad != 0).cast('float32').mean().item():.2f}%"
            #                              f"\n均值: {grad.mean().item():.6f}, 最大值: {grad.max().item():.6f}"
            #                              f"\n是否全零: {paddle.all(grad == 0).item()}"))

            #tmp_cpu = loss.cpu().detach()
            #havNan = np.isnan(np.array(tmp_cpu)).any() 
            # bmt.print_rank("Iter: {} | logits: {} | loss: {} ".format(iteration, logits, loss))

            global_loss = bmt.sum_loss(loss).item()
            mem_usage, tim_usage = add_mem_time("forward", mem_usage, tim_usage)

            # bmt.print_rank(torch.cuda.max_memory_allocated())
            # ===========
            bmt.synchronize()
            # del logits
            # paddle.device.cuda.empty_cache()
            optim_manager.backward(loss)
            mem_usage, tim_usage = add_mem_time("backward", mem_usage, tim_usage)

            # bmt.print_rank(torch.cuda.max_memory_allocated())
            # ===========
            grad_norm = optim_manager.clip_grad_norm(optimizer._param_groups, args.clip_grad, norm_type=2)
            weight_before = model.input_embedding.weight.clone()
            optim_manager.step()
            weight_after = model.input_embedding.weight.clone()
            # print(f"权重变化: {(weight_after - weight_before).abs().sum().item()}")
            mem_usage, tim_usage = add_mem_time("optim", mem_usage, tim_usage)
            # bmt.print_rank(torch.cuda.max_memory_allocated())
            gradient_records = {}
            # for name, param in model.named_parameters():
                
            #     if param.grad is not None:
            #         if paddle.isnan(param.grad).any() or paddle.isinf(param.grad).any():
            #             print(f"⚠️ 异常梯度: {name}")
            #         grad_norm_1 = paddle.norm(param.grad.astype(paddle.float32)).item()
            #         gradient_records[name] = grad_norm_1
            #         if grad_norm_1 > 100:  # 梯度爆炸阈值
            #             print(f"💥 爆炸层: {name} | 梯度范数: {grad_norm_1:.2e}")

            print("========================OKKKK5======================")

            # if iteration % 50 == 0:  # 每50步检查一次
            #     # 基本调用
            #     logits_stats = analyze_logits_anomaly(logits)
                
            #     # 当检测到危险值时详细记录
            #     if logits_stats["dangerous_ratio"] > 0:
            #         print(f"检测到异常值! 位置: {logits_stats['max_position']}, 值: {logits_stats['max_value']:.2f}")
            #         # 详细报告
            #         analyze_logits_anomaly(logits, verbose=True)

            # if iteration % 50 == 0:
            #     # 假设logits形状为 [1, 16384, 32000]
            #     logits = logits.squeeze(0)  # 移除batch维度 -> [16384, 32000]

            #     # 找出每列(每个vocab)的最大值
            #     col_max = logits.max(axis=0)  # [32000]
            #     abnormal_cols = paddle.where(col_max > 5.0)[0]  # 阈值根据情况调整

            #     print(f"异常词汇ID: {abnormal_cols.tolist()[:10]}")  # 打印前10个异常词
            #     print(f"对应最大值: {col_max[abnormal_cols].tolist()[:10]}")
            # 获取关键参数
            # key_params = {
            #     "input_embedding": model.input_embedding.weight,
            #     "lm_head": model.lm_head.weight,
            # }

            # 打印当前所有监控参数的状态
            # print("\n===== 本轮梯度总结 =====")
            # for name, param in key_params.items():
            #     if param.grad is None:
            #         status = "无梯度"
            #     elif paddle.all(param.grad == 0):
            #         status = "全零"
            #     else:
            #         status = f"正常 (范数={paddle.norm(param.grad).item():.4e})"
            #     print(f"{name:20} | 状态: {status}")

            # ==========
            iter_time = tim_usage["optim"] - tim_usage["init"]
            average_time.record(iter_time)

            with paddle.no_grad():
                task_num = len(task_names)
                targets_tmp = targets.expand([task_num, -1, -1])
                task = paddle.arange(task_num, dtype=paddle.int32).cuda()[:, None, None]
                targets_tmp = paddle.where(
                    task_ids == task,
                    targets_tmp,
                    paddle.to_tensor(-100, dtype=paddle.int32).cuda(),
                )

                task_loss_map: Dict[str, float] = {}
                task_loss_tot: Dict[str, float] = {}
                for i in range(task_num):
                    task_loss_map[task_names[i]] = loss_func(
                        logits.reshape([-1, logits.shape[-1]]), targets_tmp[i, :].reshape([-1])
                    ).item()
                    task_loss_tot[task_names[i]] = (targets_tmp[i, :].reshape([-1]) >= 0).sum().astype(paddle.float32).item()
                # gatherd_task_loss_map: List[Dict[str, float]] = allgather_objects(task_loss_map)
                # gatherd_task_loss_tot: List[Dict[str, float]] = allgather_objects(task_loss_tot)

                # global_task_loss_map: Dict[str, Union[List[float], float]] = {}
                # global_task_loss_tot: Dict[str, Union[List[float], float]] = {}

                # for idx, local_task_loss_map in enumerate(gatherd_task_loss_map):
                #     for task_name, task_loss in local_task_loss_map.items():
                #         if task_name not in global_task_loss_map:
                #             global_task_loss_map[task_name] = []
                #         global_task_loss_map[task_name].append(task_loss)
                #     for task_name, task_tot in gatherd_task_loss_tot[idx].items():
                #         if task_name not in global_task_loss_tot:
                #             global_task_loss_tot[task_name] = []
                #         global_task_loss_tot[task_name].append(task_tot)

                # task_loss_map = {}
                # for task_name in sorted(list(global_task_loss_map.keys())):
                #     avg_loss = 0.0
                #     sum_token = sum(global_task_loss_tot[task_name])
                #     for loss, token in zip(global_task_loss_map[task_name], global_task_loss_tot[task_name]):
                #         avg_loss += loss * token / sum_token
                #     task_loss_map[task_name] = avg_loss
            print("========================OKKKK6======================")
            local_total_rate = paddle.to_tensor([input_length.astype(paddle.float32).mean() / args.max_length]).cuda()
            local_total_rate = bmt.sum_loss(local_total_rate).item()
            global_token_pass += global_world_size * local_total_rate * args.max_length * args.batch_size
            avg_time = average_time.value
            #lsd.update_loss(iteration, task_loss_map)

            for task_id in data["task_ids"]:
                for task in task_id:
                    if task != -1:
                        if not data["task_names"][task] in hash:
                            hash[data["task_names"][task]] = 0
                        hash[data["task_names"][task]] += 1.0
                        total += 1.0

            # gathered_hash = allgather_objects(hash)
            # sum_total = sum(allgather_objects(total))

            # final_hash = defaultdict(int)
            # for local_hash in gathered_hash:
            #     for task, num in local_hash.items():
            #         final_hash[task] += num

            # for i in final_hash:
            #     bmt.print_rank(i, final_hash[i] / sum_total)
            # bmt.print_rank("=========================================")

            train_info = {
                "time": tim_usage["init"],
                "iteration": iteration,
                "loss": global_loss,
                "lr": lr_scheduler.current_lr,
                "lr_scale": int(optim_manager.loss_scale),
                "time_usage": tim_usage,
                "mem_usage": mem_usage,
                "avg_time": avg_time,
                "token_max": local_total_rate,
                "token_pass": global_token_pass,
                "throughout": args.max_length * args.batch_size * local_total_rate / avg_time,
                "grad_norm": grad_norm.item(),
                "mask_max": ((targets >= 0).sum(-1).astype(paddle.float32).mean() / args.max_length).item(),
                "num_gpus": global_world_size,
                "task_loss": task_loss_map,
            }
            # bmt.print_rank(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            # print(f"global_loss  { global_loss} lr_scheduler.current_lr {lr_scheduler.current_lr} \
            #       grad_norm {grad_norm}")
            # print(input_length.astype(paddle.float32).mean() / args.max_length / (args.batch_size if args.flash == "cuda" else 1))
            # print((targets >= 0).sum(-1).astype(paddle.float32).mean() / args.max_length / (args.batch_size if args.flash == "cuda" else 1))
            bmt.print_rank(
                (
                    "| Iter: {:6d} | loss: {:.4f} | lr: {:.4e}, scale: {:10.4f} | avg_time: {:.4f}; cur_time:{:.4f}={:.4f}+{:.4f} |"
                    + " token/max: {:.4f} | mask/max: {:.4f} | grad_norm: {:.4f} | mem: {:.2f} |"
                ).format(
                    iteration,
                    global_loss,
                    lr_scheduler.current_lr,
                    int(optim_manager.loss_scale),
                    #iter_time,
                    avg_time,
                    tim_usage["optim"] - tim_usage["init"],
                    tim_usage["backward"] - tim_usage["init"],
                    tim_usage["optim"] - tim_usage["backward"],
                    (input_length.astype(paddle.float32).mean() / args.max_length / (args.batch_size if args.flash == "cuda" else 1)).item(),
                    ((targets >= 0).sum(-1).astype(paddle.float32).mean()
                    / args.max_length
                    / (args.batch_size if args.flash == "cuda" else 1)).item(),
                    grad_norm.item(),
                    max(mem_usage["forward"][1], mem_usage["backward"][1]),
                )
            )

            bmt.print_rank(
                "| "
                + " | ".join(["{}: {:.4f}".format(task_name, loss) for task_name, loss in task_loss_map.items()])
                + " |"
            )

            # if iteration % args.inspect_iters == 0:
            #     model_inspect = bmt.inspect.inspect_model(model, "*")
            #     bmt.print_rank(bmt.inspect.format_summary(model_inspect))
            #     train_info["model_inspect"] = model_inspect

            if args.log_dir is not None and bmt.rank() == 0:
                log_mgr.write(**train_info)
            if args.tensorboard is not None and bmt.rank() == 0:
                writer.add_scalar("Loss/train", global_loss, iteration)
                writer.add_scalar("Optimizer/lr", lr_scheduler.current_lr, iteration)
                writer.add_scalar("Optimizer/scale", optim_manager.loss_scale, iteration)
                writer.add_scalar("Optimizer/grad_norm", grad_norm.item(), iteration)
                for task_name, loss in task_loss_map.items():
                    writer.add_scalar("Loss/train/{}".format(task_name), loss, iteration)

            # -------- save file. If need to backup by Klara platform, use export.xx_save --------
            # if args.save is not None and iteration % args.save_iters == 0:
            #     exporter.export(model, dataloader, optimizer, iteration, args, final_save=False)
            
            if iteration >= args.train_iters:
                break

    except Exception as e:
        print(f"train loop err: {e}")
        raise e
    #finally:
    # exporter.export(model, dataloader, optimizer, -1, args, final_save=False)
    dataloader.close()


def main():
    args = initialize()
    bmt.print_rank(json.dumps(vars(args), indent=2, sort_keys=True))
    tokenizer, model, optimizer, lr_scheduler = setup_model_and_optimizer(args)
    bmt.print_rank("finish loading")
    # register_activation_hooks(model)
    # bmt.print_rank("钩子注册成功")
    pretrain(args, tokenizer, model, optimizer, lr_scheduler)


if __name__ == "__main__":
    main()
