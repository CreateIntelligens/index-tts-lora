#!/usr/bin/env python3
"""
GPU 管理器 - 自動偵測與分配 GPU 資源。

功能：
- 自動偵測可用 GPU 數量與狀態
- 智慧分配 GPU 給不同的 Worker
- 監控記憶體使用
- 負載平衡
"""

import os
import logging
from typing import List, Optional, Dict
from dataclasses import dataclass

try:
    import torch
except ImportError:
    print("[錯誤] 請安裝 PyTorch: pip install torch")
    raise


@dataclass
class GPUInfo:
    """GPU 資訊類別"""
    index: int
    name: str
    total_memory: float  # GB
    available_memory: float  # GB
    utilization: float  # 0-100%


class GPUManager:
    """GPU 資源管理器"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.gpu_count = 0
        self.gpu_info: List[GPUInfo] = []
        self.enabled_gpus: List[int] = []

        self._detect_gpus()

    def _detect_gpus(self):
        """偵測可用的 GPU"""
        if not torch.cuda.is_available():
            self.logger.warning("[警告] 未偵測到 CUDA GPU，將使用 CPU 模式")
            return

        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
        if cuda_visible:
            self.logger.info(f"[資訊] CUDA_VISIBLE_DEVICES 設定: {cuda_visible}")

        self.gpu_count = torch.cuda.device_count()
        self.logger.info(f"[資訊] 偵測到 {self.gpu_count} 個 GPU")

        for i in range(self.gpu_count):
            try:
                props = torch.cuda.get_device_properties(i)
                name = props.name
                total_memory = props.total_memory / (1024**3)

                torch.cuda.set_device(i)
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                reserved = torch.cuda.memory_reserved(i) / (1024**3)
                available = total_memory - reserved

                gpu_info = GPUInfo(
                    index=i,
                    name=name,
                    total_memory=total_memory,
                    available_memory=available,
                    utilization=0.0
                )

                self.gpu_info.append(gpu_info)
                self.enabled_gpus.append(i)

                self.logger.info(
                    f"  GPU {i}: {name} | "
                    f"記憶體: {total_memory:.1f}GB (可用: {available:.1f}GB)"
                )

            except Exception as e:
                self.logger.warning(f"[警告] 無法讀取 GPU {i} 資訊: {e}")

    def get_gpu_count(self) -> int:
        """獲取可用 GPU 數量"""
        return self.gpu_count

    def get_gpu_info(self, gpu_id: int) -> Optional[GPUInfo]:
        """獲取指定 GPU 的資訊"""
        if 0 <= gpu_id < len(self.gpu_info):
            return self.gpu_info[gpu_id]
        return None

    def get_all_gpu_info(self) -> List[GPUInfo]:
        """獲取所有 GPU 資訊"""
        return self.gpu_info.copy()

    def assign_gpu_to_worker(self, worker_id: int) -> int:
        """
        為 Worker 分配 GPU。

        策略：輪詢分配 (Round-Robin)。

        Args:
            worker_id: Worker 編號。

        Returns:
            分配的 GPU ID。
        """
        if self.gpu_count == 0:
            return -1

        gpu_id = self.enabled_gpus[worker_id % self.gpu_count]
        self.logger.debug(f"[分配] Worker {worker_id} -> GPU {gpu_id}")
        return gpu_id

    def get_optimal_worker_count(self,
                                  memory_per_worker: float = 4.0,
                                  max_workers: Optional[int] = None) -> int:
        """
        計算最佳 Worker 數量。

        Args:
            memory_per_worker: 每個 Worker 預計使用的記憶體 (GB)。
            max_workers: 最大 Worker 數量限制。

        Returns:
            建議的 Worker 數量。
        """
        if self.gpu_count == 0:
            import multiprocessing
            cpu_count = multiprocessing.cpu_count()
            workers = max(1, cpu_count // 2)
            self.logger.info(f"[建議] CPU 模式：建議 {workers} 個 Worker")
            return workers

        workers_per_gpu = []
        for gpu in self.gpu_info:
            available_workers = max(1, int(gpu.available_memory / memory_per_worker))
            workers_per_gpu.append(available_workers)

        total_workers = sum(workers_per_gpu)

        if max_workers is not None:
            total_workers = min(total_workers, max_workers)

        self.logger.info(f"[分析] GPU 資源分析：")
        for i, (gpu, workers) in enumerate(zip(self.gpu_info, workers_per_gpu)):
            self.logger.info(
                f"  GPU {i} ({gpu.name}): "
                f"{gpu.available_memory:.1f}GB 可用 -> {workers} Workers"
            )
        self.logger.info(f"[建議] 總 Worker 數: {total_workers}")

        return total_workers

    def get_device_string(self, gpu_id: int) -> str:
        """
        獲取 PyTorch Device 字串。

        Args:
            gpu_id: GPU ID (-1 表示 CPU)。

        Returns:
            Device 字串 (例如: "cuda:0", "cpu")。
        """
        if gpu_id < 0 or self.gpu_count == 0:
            return "cpu"
        return f"cuda:{gpu_id}"

    def set_cuda_visible_devices(self, gpu_ids: List[int]):
        """
        設定 CUDA_VISIBLE_DEVICES 環境變數。

        注意：必須在初始化 CUDA 前調用才有效。

        Args:
            gpu_ids: 要使用的 GPU ID 列表。
        """
        gpu_str = ",".join(map(str, gpu_ids))
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
        self.logger.info(f"[設定] CUDA_VISIBLE_DEVICES={gpu_str}")

    def print_summary(self):
        """列印 GPU 狀態摘要"""
        print("\n" + "="*60)
        print("🎮 GPU 資源摘要")
        print("="*60)

        if self.gpu_count == 0:
            print("[警告] 未偵測到 GPU，將使用 CPU 模式")
            return

        print(f"📊 可用 GPU 數量: {self.gpu_count}")
        print()

        for gpu in self.gpu_info:
            print(f"GPU {gpu.index}: {gpu.name}")
            print(f"  └─ 總記憶體: {gpu.total_memory:.1f} GB")
            print(f"  └─ 可用記憶體: {gpu.available_memory:.1f} GB")
            print()

        print("="*60 + "\n")


def get_global_gpu_manager() -> GPUManager:
    """獲取全域 GPU 管理器實例 (單例模式)"""
    if not hasattr(get_global_gpu_manager, '_instance'):
        get_global_gpu_manager._instance = GPUManager()
    return get_global_gpu_manager._instance


def get_available_gpu_count() -> int:
    """快速獲取可用 GPU 數量"""
    manager = get_global_gpu_manager()
    return manager.get_gpu_count()


def assign_worker_gpu(worker_id: int) -> str:
    """快速為 Worker 分配 GPU 並返回 Device 字串"""
    manager = get_global_gpu_manager()
    gpu_id = manager.assign_gpu_to_worker(worker_id)
    return manager.get_device_string(gpu_id)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    manager = GPUManager()
    manager.print_summary()

    if manager.get_gpu_count() > 0:
        print("🧪 測試 Worker 分配：")
        for i in range(10):
            device = assign_worker_gpu(i)
            print(f"  Worker {i} -> {device}")

        print()
        optimal = manager.get_optimal_worker_count(memory_per_worker=4.0)
        print(f"💡 建議 Worker 數量: {optimal}")
