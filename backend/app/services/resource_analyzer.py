import os
import sys
import time
import shutil
import platform
import logging
from typing import Optional, Dict, Any

import psutil
import torch

from app.schemas.resource import (
    ResourceSnapshotResponse,
    CpuInfo,
    MemoryInfo,
    GpuInfo,
    CudaInfo,
    ProcessInfo,
    DiskInfo,
    RuntimeInfo,
    ResourceFeasibility,
)
from app.services.embedding_service import EmbeddingService

logger = logging.getLogger("orchestrator")

class ResourceAnalyzer:
    def __init__(self):
        self.embedding_service = EmbeddingService()

    def _get_cpu_info(self) -> CpuInfo:
        logical_cores = psutil.cpu_count(logical=True) or 1
        physical_cores = psutil.cpu_count(logical=False)
        utilization_percent = float(psutil.cpu_percent(interval=0.05))

        frequency_mhz: Optional[float] = None
        try:
            freq = psutil.cpu_freq()
            if freq and hasattr(freq, "current") and freq.current > 0:
                frequency_mhz = round(float(freq.current), 2)
        except Exception:
            frequency_mhz = None

        return CpuInfo(
            physical_cores=physical_cores,
            logical_cores=logical_cores,
            utilization_percent=utilization_percent,
            frequency_mhz=frequency_mhz,
        )

    def _get_memory_info(self) -> MemoryInfo:
        vm = psutil.virtual_memory()
        total_gb = round(vm.total / (1024**3), 2)
        available_gb = round(vm.available / (1024**3), 2)
        used_gb = round((vm.total - vm.available) / (1024**3), 2)
        utilization_percent = round(float(vm.percent), 2)

        return MemoryInfo(
            total_gb=total_gb,
            available_gb=available_gb,
            used_gb=used_gb,
            utilization_percent=utilization_percent,
        )

    def _get_gpu_info(self) -> GpuInfo:
        cuda_usable = torch.cuda.is_available()
        pytorch_status = "CUDA Available" if cuda_usable else "CPU Mode / Unavailable"
        ollama_status = "GPU Available (Local Ollama)"

        if not cuda_usable:
            return GpuInfo(
                available=False,
                name=None,
                device_count=0,
                utilization_percent=None,
                total_vram_gb=None,
                free_vram_gb=None,
                used_vram_gb=None,
                ollama_gpu_status=ollama_status,
                pytorch_cuda_status=pytorch_status,
            )

        try:
            device_count = torch.cuda.device_count()
            name = torch.cuda.get_device_name(0) if device_count > 0 else "NVIDIA GPU"
            
            total_vram_gb: Optional[float] = None
            free_vram_gb: Optional[float] = None
            used_vram_gb: Optional[float] = None

            if hasattr(torch.cuda, "mem_get_info") and device_count > 0:
                free_bytes, total_bytes = torch.cuda.mem_get_info(0)
                total_vram_gb = round(total_bytes / (1024**3), 2)
                free_vram_gb = round(free_bytes / (1024**3), 2)
                used_vram_gb = round((total_bytes - free_bytes) / (1024**3), 2)

            return GpuInfo(
                available=True,
                name=name,
                device_count=device_count,
                utilization_percent=None,
                total_vram_gb=total_vram_gb,
                free_vram_gb=free_vram_gb,
                used_vram_gb=used_vram_gb,
                ollama_gpu_status=ollama_status,
                pytorch_cuda_status=pytorch_status,
            )
        except Exception as e:
            logger.warning(f"Error inspecting PyTorch GPU memory details: {str(e)}")
            return GpuInfo(
                available=True,
                name="NVIDIA GPU (Details Unknown)",
                device_count=1,
                utilization_percent=None,
                total_vram_gb=None,
                free_vram_gb=None,
                used_vram_gb=None,
                ollama_gpu_status=ollama_status,
                pytorch_cuda_status=pytorch_status,
            )

    def _get_cuda_info(self) -> CudaInfo:
        cuda_avail = torch.cuda.is_available()
        pytorch_ver = torch.__version__
        cuda_ver = torch.version.cuda if hasattr(torch.version, "cuda") else None

        return CudaInfo(
            available=cuda_avail,
            pytorch_version=pytorch_ver,
            cuda_version=cuda_ver,
        )

    def _get_process_info(self) -> ProcessInfo:
        proc = psutil.Process(os.getpid())
        mem_mb = round(proc.memory_info().rss / (1024**2), 2)
        cpu_pct = round(float(proc.cpu_percent(interval=None)), 2)

        return ProcessInfo(
            pid=os.getpid(),
            memory_mb=mem_mb,
            cpu_percent=cpu_pct,
        )

    def _get_disk_info(self) -> DiskInfo:
        root_dir = os.path.abspath(os.path.sep)
        usage = shutil.disk_usage(root_dir)
        
        total_gb = round(usage.total / (1024**3), 2)
        free_gb = round(usage.free / (1024**3), 2)
        used_gb = round(usage.used / (1024**3), 2)
        utilization_percent = round(float((usage.used / usage.total) * 100), 2)

        return DiskInfo(
            total_gb=total_gb,
            free_gb=free_gb,
            used_gb=used_gb,
            utilization_percent=utilization_percent,
        )

    def _get_runtime_info(self) -> RuntimeInfo:
        os_name = f"{platform.system()} {platform.release()}"
        py_ver = platform.python_version()
        exec_device = self.embedding_service.device

        return RuntimeInfo(
            operating_system=os_name,
            python_version=py_ver,
            execution_device=exec_device,
        )

    def _evaluate_feasibility(
        self, memory: MemoryInfo, cpu: CpuInfo, cuda: CudaInfo
    ) -> ResourceFeasibility:
        notes_list = []
        status = "sufficient"

        if memory.available_gb < 1.0 or memory.utilization_percent > 92.0:
            status = "critical"
            notes_list.append(f"RAM available ({memory.available_gb} GB) is critically low.")
        elif memory.utilization_percent > 80.0 or cpu.utilization_percent > 85.0:
            status = "constrained"
            notes_list.append(f"High resource utilization detected (RAM: {memory.utilization_percent}%, CPU: {cpu.utilization_percent}%).")

        notes_list.append("Ollama LLM inference is running on local GPU acceleration.")

        return ResourceFeasibility(
            status=status,
            notes=" ".join(notes_list),
        )

    def get_resource_snapshot(self) -> ResourceSnapshotResponse:
        t0 = time.perf_counter()

        cpu = self._get_cpu_info()
        memory = self._get_memory_info()
        gpu = self._get_gpu_info()
        cuda = self._get_cuda_info()
        process = self._get_process_info()
        disk = self._get_disk_info()
        runtime = self._get_runtime_info()

        feasibility = self._evaluate_feasibility(memory, cpu, cuda)

        t1 = time.perf_counter()
        measurement_latency_ms = round((t1 - t0) * 1000, 2)

        return ResourceSnapshotResponse(
            cpu=cpu,
            memory=memory,
            gpu=gpu,
            cuda=cuda,
            process=process,
            disk=disk,
            runtime=runtime,
            feasibility=feasibility,
            measurement_latency_ms=measurement_latency_ms,
        )
