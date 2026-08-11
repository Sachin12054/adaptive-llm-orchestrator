import React from 'react';
import { Activity } from 'lucide-react';
import MetricRing from './MetricRing';
import ProgressBar from './ProgressBar';

const ResourceMonitorCard = ({ resourceData }) => {
  const cpuPercent = resourceData?.cpu?.utilization_percent;
  const ramPercent = resourceData?.memory?.utilization_percent;
  const memoryUsedGb = resourceData?.memory?.used_gb;
  const memoryTotalGb = resourceData?.memory?.total_gb;

  const gpuObj = resourceData?.gpu;
  const freeVram = gpuObj?.free_vram_gb;
  const usedVram = gpuObj?.used_vram_gb;
  const totalVram = gpuObj?.total_vram_gb;
  const vramUsedPercent = (usedVram != null && totalVram != null && totalVram > 0)
    ? (usedVram / totalVram) * 100
    : null;

  const ollamaGpu = gpuObj?.ollama_gpu_status || 'GPU Available (Local Ollama)';
  const pytorchCuda = gpuObj?.pytorch_cuda_status || 'CPU Mode / Unavailable';

  return (
    <div className="glass-panel p-3.5 space-y-3">
      <div className="flex items-center justify-between border-b border-slate-800/80 pb-2">
        <div className="flex items-center gap-1.5">
          <Activity className="w-4 h-4 text-emerald-400" />
          <h4 className="font-bold text-[11px] uppercase tracking-wider text-slate-300">Resource Monitor</h4>
        </div>
        <span className="text-[10px] text-slate-400 font-mono">Telemetry</span>
      </div>

      <div className="grid grid-cols-2 gap-2 text-xs font-mono">
        {/* GPU VRAM Ring Gauge */}
        {vramUsedPercent != null && (
          <div className="bg-slate-950/80 p-2 rounded-lg border border-slate-800/80 flex flex-col items-center justify-center">
            <MetricRing
              value={vramUsedPercent}
              label="GPU VRAM"
              size={60}
              strokeWidth={5}
              color="emerald"
              formattedValue={`${usedVram.toFixed(1)} GB`}
            />
            {freeVram != null && totalVram != null && (
              <span className="text-[9px] text-slate-400 mt-1 font-sans">
                {freeVram.toFixed(1)} GB FREE / {totalVram.toFixed(1)} GB TOTAL
              </span>
            )}
          </div>
        )}

        {/* CPU & System RAM Progress Bars */}
        <div className={`bg-slate-950/80 p-2 rounded-lg border border-slate-800/80 space-y-2 flex flex-col justify-center ${vramUsedPercent == null ? 'col-span-2' : ''}`}>
          {cpuPercent != null && (
            <ProgressBar
              value={cpuPercent}
              maxValue={100}
              label="CPU Utilization"
              color="cyan"
              formattedValue={`${cpuPercent.toFixed(1)}%`}
            />
          )}
          {ramPercent != null && (
            <ProgressBar
              value={ramPercent}
              maxValue={100}
              label="System RAM"
              color="indigo"
              formattedValue={memoryUsedGb != null && memoryTotalGb != null ? `${memoryUsedGb.toFixed(1)} / ${memoryTotalGb.toFixed(1)} GB` : `${ramPercent.toFixed(1)}%`}
            />
          )}
        </div>

        {/* Compact Runtime Status */}
        <div className="col-span-2 bg-slate-950/80 p-2 rounded-lg border border-slate-800/80 text-[10px] space-y-1">
          <div className="flex items-center justify-between">
            <span className="text-slate-400 flex items-center gap-1 font-sans">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400"></span> Ollama GPU:
            </span>
            <span className="text-emerald-400 font-bold">{ollamaGpu}</span>
          </div>
          <div className="flex items-center justify-between border-t border-slate-800/60 pt-1">
            <span className="text-slate-400 flex items-center gap-1 font-sans">
              <span className="w-1.5 h-1.5 rounded-full bg-amber-400"></span> PyTorch CUDA:
            </span>
            <span className="text-amber-300 font-bold">{pytorchCuda}</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ResourceMonitorCard;
