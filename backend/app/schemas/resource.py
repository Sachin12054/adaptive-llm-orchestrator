from typing import Optional, Dict
from pydantic import BaseModel, Field

class CpuInfo(BaseModel):
    physical_cores: Optional[int] = Field(None, description="Physical CPU core count")
    logical_cores: int = Field(..., description="Logical CPU core count")
    utilization_percent: float = Field(..., description="Current system-wide CPU utilization percentage [0-100]")
    frequency_mhz: Optional[float] = Field(None, description="Current CPU frequency in MHz")

class MemoryInfo(BaseModel):
    total_gb: float = Field(..., description="Total system RAM in Gigabytes")
    available_gb: float = Field(..., description="Available system RAM in Gigabytes")
    used_gb: float = Field(..., description="Used system RAM in Gigabytes")
    utilization_percent: float = Field(..., description="RAM utilization percentage [0-100]")

class GpuInfo(BaseModel):
    available: bool = Field(..., description="True if GPU acceleration or hardware is present")
    name: Optional[str] = Field(None, description="GPU device name if available")
    device_count: int = Field(0, description="Number of detected GPU devices")
    utilization_percent: Optional[float] = Field(None, description="GPU compute utilization percentage if measurable")
    total_vram_gb: Optional[float] = Field(None, description="Total VRAM in Gigabytes if measurable")
    free_vram_gb: Optional[float] = Field(None, description="Free VRAM in Gigabytes if measurable")
    used_vram_gb: Optional[float] = Field(None, description="Used VRAM in Gigabytes if measurable")
    ollama_gpu_status: str = Field("GPU Available (Local Ollama)", description="Ollama runtime GPU execution status")
    pytorch_cuda_status: str = Field("CPU Mode / Unavailable", description="PyTorch runtime CUDA status")

class CudaInfo(BaseModel):
    available: bool = Field(..., description="True if PyTorch CUDA support is available")
    pytorch_version: str = Field(..., description="Installed PyTorch package version string")
    cuda_version: Optional[str] = Field(None, description="CUDA runtime version reported by PyTorch")

class ProcessInfo(BaseModel):
    pid: int = Field(..., description="Current Python process identifier")
    memory_mb: float = Field(..., description="Resident Set Size (RSS) memory used by current process in MB")
    cpu_percent: float = Field(..., description="Process CPU utilization percentage")

class DiskInfo(BaseModel):
    total_gb: float = Field(..., description="Total disk storage space in Gigabytes")
    free_gb: float = Field(..., description="Free disk storage space in Gigabytes")
    used_gb: float = Field(..., description="Used disk storage space in Gigabytes")
    utilization_percent: float = Field(..., description="Disk storage utilization percentage [0-100]")

class RuntimeInfo(BaseModel):
    operating_system: str = Field(..., description="Operating System platform name and release version")
    python_version: str = Field(..., description="Python interpreter version string")
    execution_device: str = Field(..., description="Actual execution device used by embedding service (cpu/cuda)")

class ResourceFeasibility(BaseModel):
    status: str = Field(..., description="Factual system status: sufficient, constrained, critical, unknown")
    notes: str = Field(..., description="Technical rationale based on measured hardware metrics")

class ResourceSnapshotResponse(BaseModel):
    cpu: CpuInfo = Field(..., description="Real-time CPU telemetry")
    memory: MemoryInfo = Field(..., description="Real-time system RAM telemetry")
    gpu: GpuInfo = Field(..., description="Real-time GPU acceleration telemetry")
    cuda: CudaInfo = Field(..., description="PyTorch CUDA environment status")
    process: ProcessInfo = Field(..., description="Current application process resource usage")
    disk: DiskInfo = Field(..., description="System disk storage telemetry")
    runtime: RuntimeInfo = Field(..., description="Python & OS runtime environment details")
    feasibility: ResourceFeasibility = Field(..., description="Real-time resource feasibility evaluation")
    measurement_latency_ms: float = Field(..., description="Time taken to gather resource snapshot in milliseconds")
