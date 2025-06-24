# -*- coding:UTF-8 -*-

import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def default_log_parser(file_name):
    loss_list = []
    for line in open(file_name):
        idx = line.find(target_line)
        if idx != -1:
            loss = float(line[idx+len(target_line):idx+len(target_line)+loss_len])
            loss_list.append(loss)
    return loss_list

def R_square(l1, l2):
    min_len = min(len(l1), len(l2))
    n1, n2 = np.array(l1[:min_len]), np.array(l2[:min_len])
    r = np.corrcoef(n1, n2)[1, 0]
    return r**2

def _smooth_loss(train_iter, loss):
    smooth_iter = []
    smooth_output = []
    tmp = 0
    for i in range(len(loss)):
        if (i+1) % 10 == 0:
            tmp += loss[i]
            smooth_iter.append(i+1)
            smooth_output.append(tmp/10.0)
            tmp = 0
        else:
            tmp += loss[i]
    return smooth_iter, smooth_output

def draw_loss(title, log_files, log_parser=default_log_parser):
    smooth = 0
    '''
    log_files:要绘制的日志文件名的list
    log_parser:日志解析器
    '''
    logs_loss = list(map(log_parser, log_files))

    #print(f"top 10 data of MLU: {mlu_loss_stride10[:11]}")
    #print(f"top 10 data of GPU: {logs_loss[1][:11]}")
    #print(f"last 10 data of MLU: {mlu_loss_stride10[-10:]}")
    #print(f"last 10 data of GPU: {logs_loss[1][-10:]}")

    length = min(map(len, logs_loss))
    # length = min(length, 500)
    width = length * 8.0 / 500.0;
    if width < 8:
        width = 32.0
    if width > 25:
        width = 25.0
    plt.rcParams['figure.figsize']=(width, 5.0)
    plt.ion()

    mlu_x = [i for i in range(len(logs_loss[0]))]
    mlu_y = logs_loss[0]
    if smooth:
        mlu_x, mlu_y = _smooth_loss(mlu_x, logs_loss[0])
    plt.plot(mlu_x, mlu_y, label="bi-7b-sft", linewidth=1)
    
    #gpu_x = [i for i in range(len(logs_loss[1]))]
    #gpu_y = logs_loss[1]
    #plt.plot(gpu_x, gpu_y, label="gpu-7b-incremental", linewidth=1)

    plt.legend()
    plt.show()
    plt.savefig(title)

    if len(log_files) == 2:
        print('R_square: %f' % R_square(logs_loss[0], logs_loss[1]))


def print_usage():
    print('Usage:')
    print('python draw_loss.py img_title mlu_log_file gpu_log_file')
    print('\n')

def configure_pattern():
    #title
    global title
    title = ""

    #plt params
    #plt.rcParams['figure.figsize']=(20.0, 5.0)
    plt.rcParams['savefig.dpi']=150
    plt.grid()
    #plt.xlim((0, 1000))
    #plt.ylim((0, 12))
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.title(title)

    # target line: *** lm loss 1.178288E+01 ***
    global target_line
    global loss_len
    target_line = "loss: "
    loss_len = len("1.4365")

    #default log files
    global mlu_file
    global gpu_file
    #mlu_file = "01-26-16-37-sft-llama2-7B-MLU.log"
    #gpu_file = "01-26-16-37-sft-llama2-7B-MLU.log"
    mlu_file = "sft_train.3_4096_FA_bf16.log"
    gpu_file = "sft_train.3_4096_FA_bf16.log"
    #gpu_file = "sft-llama2-7B-bf16-no-offload-no-fa-GPU.log"



if __name__ == "__main__":
    configure_pattern()
    if len(sys.argv) == 4:
        print_usage()
        title = sys.argv[1]
        log_files = sys.argv[2:]
    else:
        log_files = [mlu_file, gpu_file]
    print(f"log_files: {log_files}")
    draw_loss("7b-sft-loss", log_files)


