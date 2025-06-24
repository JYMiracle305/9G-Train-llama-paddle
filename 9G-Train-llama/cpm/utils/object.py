import pickle

import bmtrain_paddle as bmt
import paddle


def allgather_objects(obj):
    if bmt.world_size() == 1:
        return [obj]

    with paddle.no_grad():
        data_bytes: bytes = pickle.dumps(obj)
        data_length: int = len(data_bytes)

        gpu_data_length = paddle.to_tensor([data_length], dtype=paddle.int64).cuda()
        gathered_length = bmt.distributed.all_gather(gpu_data_length).reshape([-1]).cpu()
        max_data_length = gathered_length.max().item()

        gpu_data_bytes = paddle.zeros(max_data_length, dtype=paddle.uint8).cuda()
        # byte_storage = paddle.ByteStorage.from_buffer(data_bytes)
        # gpu_data_bytes[:data_length] = paddle.ByteTensor(byte_storage)

        gathered_data = bmt.distributed.all_gather(gpu_data_bytes).cpu()

        ret = []
        for i in range(gathered_data.shape[0]):
            data_bytes = gathered_data[i, : gathered_length[i].item()].numpy().tobytes()
            ret.append(pickle.loads(data_bytes))
        return ret
