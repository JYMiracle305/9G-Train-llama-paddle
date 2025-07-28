import os
import pickle
import struct
import multiprocessing
from tqdm import tqdm
from typing import Dict, Any
import base64
import math

# 常量定义
BLOCK_SIZE = 16 * 1024 * 1024  # 16MB块大小
MAGIC_BYTE = b'\x1F'           # 记录起始魔法字节
PICKLE_PROTOCOL = 3            # Python pickle协议版本
MAX_TEXT_LENGTH = 4096         # Base64文本最大长度限制

class BinaryToLlamaConverter:
    @staticmethod
    def split_binary_data(data: bytes) -> list[bytes]:
        """
        将二进制数据分割成适合Base64编码的小块
        计算规则：MAX_TEXT_LENGTH // 4 * 3 （Base64编码后长度膨胀约4/3倍）
        """
        max_binary_size = math.floor(MAX_TEXT_LENGTH / 4 * 3)
        chunks = []
        for i in range(0, len(data), max_binary_size):
            chunks.append(data[i:i + max_binary_size])
        return chunks

    @staticmethod
    def create_record(data: Dict[str, Any]) -> bytes:
        """创建LLAMA3兼容的记录格式"""
        pickle_data = pickle.dumps(data, protocol=PICKLE_PROTOCOL)
        assert pickle_data.startswith(b'\x80\x03'), "Invalid pickle header"
        return MAGIC_BYTE + struct.pack("<I", len(pickle_data)) + pickle_data

    @staticmethod
    def convert_file(
        input_path: str,
        output_path: str,
        chunk_size: int = BLOCK_SIZE // 2  # 默认8MB原始数据块
    ) -> None:
        """
        转换原始二进制文件为LLAMA3训练格式
        关键修改：确保每个Base64编码后的文本不超过MAX_TEXT_LENGTH
        """
        file_size = os.path.getsize(input_path)
        total_chunks = (file_size + chunk_size - 1) // chunk_size

        with open(input_path, "rb") as f_in, open(output_path, "wb") as f_out:
            current_block_remaining = BLOCK_SIZE
            chunk_id = 0

            with tqdm(total=file_size, unit='B', unit_scale=True, desc="Converting") as pbar:
                while True:
                    # 读取原始数据块
                    raw_chunk = f_in.read(chunk_size)
                    if not raw_chunk:
                        break

                    # 分割二进制数据确保Base64编码后不超过限制
                    binary_chunks = BinaryToLlamaConverter.split_binary_data(raw_chunk)
                    
                    for seq_id, binary_part in enumerate(binary_chunks):
                        # Base64编码并验证长度
                        mid = len(binary_part) // 2
                        input_part = binary_part[:mid]
                        output_part = binary_part[mid:]
                        b64_text = base64.b64encode(input_part).decode('utf-8')
                        output_b64 = base64.b64encode(output_part).decode('utf-8')
                        assert len(b64_text) <= MAX_TEXT_LENGTH, f"Base64长度超标: {len(b64_text)}"

                        # 构建记录
                        record = BinaryToLlamaConverter.create_record({
                            "input": b64_text,  # 关键修改：使用标准字段名
                            "output": output_b64,
                            "meta": {
                                "source": os.path.basename(input_path),
                                "chunk_id": f"{chunk_id}.{seq_id}",  # 添加子序列ID
                                "original_size": len(binary_part),
                                "total_parts": len(binary_chunks)
                            }
                        })

                        # 块对齐处理
                        if len(record) > current_block_remaining:
                            f_out.write(b'\x00' * current_block_remaining)
                            current_block_remaining = BLOCK_SIZE

                        f_out.write(record)
                        current_block_remaining -= len(record)

                    chunk_id += 1
                    pbar.update(len(raw_chunk))

            # 填充最后一个块
            if current_block_remaining < BLOCK_SIZE:
                f_out.write(b'\x00' * current_block_remaining)

        print(f"\nConversion successful: {output_path}")
        print(f"Input size: {file_size/1024/1024:.2f} MB")
        print(f"Output size: {os.path.getsize(output_path)/1024/1024:.2f} MB")
        print(f"Total chunks: {chunk_id} (split into {len(binary_chunks)} parts each)")

class LlamaDatasetValidator:
    @staticmethod
    def validate_file(file_path: str, max_records: int = 5) -> None:
        """验证文件格式完整性"""
        print(f"\nValidating {file_path}...")
        with open(file_path, "rb") as f:
            record_count = 0
            error_count = 0
            current_pos = 0

            while True:
                pos = f.tell()
                magic = f.read(1)

                # 文件结束检查
                if not magic:
                    break

                # 跳过填充字节
                if magic == b'\x00':
                    current_pos += 1
                    continue

                # 魔法字节验证
                if magic != MAGIC_BYTE:
                    print(f"ERROR: Invalid magic byte 0x{magic.hex()} at position {pos}")
                    error_count += 1
                    f.seek(pos + 1)  # 跳过当前字节继续检查
                    continue

                # 读取记录长度
                try:
                    length_bytes = f.read(4)
                    length = struct.unpack("<I", length_bytes)[0]
                except:
                    print(f"ERROR: Broken length field at {pos}")
                    error_count += 1
                    break

                # 读取记录数据
                try:
                    pickle_data = f.read(length)
                    record = pickle.loads(pickle_data)
                    record_count += 1

                    if record_count <= max_records:
                        print(f"\nRecord #{record_count}:")
                        print(f"Chunk ID: {record['meta']['chunk_id']}")
                        print(f"Data size: {len(record['binary_data'])} bytes")
                        print(f"Meta: {record['meta']}")

                except Exception as e:
                    print(f"ERROR: Failed to unpack record at {pos}: {str(e)}")
                    error_count += 1
                    break

            print(f"\nValidation complete: {record_count} records checked")
            if error_count == 0:
                print("SUCCESS: No errors found")
            else:
                print(f"WARNING: Found {error_count} errors")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert binary files to LLAMA3 training format")
    parser.add_argument("input", help="Input binary file path")
    parser.add_argument("output", help="Output file path")
    parser.add_argument("--validate", action="store_true", help="Run validation after conversion")
    args = parser.parse_args()

    # 执行转换
    BinaryToLlamaConverter.convert_file(args.input, args.output)

    # 可选验证
    if args.validate:
        LlamaDatasetValidator.validate_file(args.output)