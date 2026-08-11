import React from 'react';
import { DollarSign, Clock, Zap } from 'lucide-react';
import { formatLatency } from '../utils/formatLatency';

const CostLatencyPanel = ({ orchestrationRes, executionMode = 'local' }) => {
  const isOnline = executionMode === 'online' || executionMode === 'mistral' || executionMode === 'gemini';
  const pipelineMs = orchestrationRes?.pipeline_latency_ms || orchestrationRes?.total_pipeline_latency_ms;
  const genMs = orchestrationRes?.generation?.latency_ms;
  const loadMs = orchestrationRes?.generation?.ollama_load_ms;
  const evalMs = orchestrationRes?.generation?.ollama_eval_ms;
  const verifyMs = orchestrationRes?.verification?.verification_latency_ms;
  const rewardMs = orchestrationRes?.reward?.latency_ms;
  const provider = orchestrationRes?.generation?.provider || (isOnline ? 'Online API' : 'Local Ollama');

  const headerLabel = isOnline ? 'Online API Pool' : '100% Local Ollama';
  const costDisplay = isOnline ? 'Not Tracked' : '$0.00';
  const inferenceLabel = isOnline ? `${provider} Inference:` : 'Ollama Generation:';

  return (
    <div className="glass-panel p-3.5 space-y-2.5 border-emerald-500/20 shadow-emerald-500/5">
      <div className="flex items-center justify-between border-b border-slate-800/80 pb-2">
        <div className="flex items-center gap-1.5">
          <DollarSign className="w-4 h-4 text-emerald-400" />
          <h4 className="font-bold text-[11px] uppercase tracking-wider text-slate-300">Cost & Latency Telemetry</h4>
        </div>
        <span className={`text-[10px] px-2 py-0.5 rounded font-mono font-semibold border ${
          isOnline
            ? 'bg-purple-500/15 text-purple-300 border-purple-500/30'
            : 'bg-emerald-500/15 text-emerald-300 border-emerald-500/30'
        }`}>
          {headerLabel}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-2 text-xs font-mono">
        <div className="bg-slate-950/80 p-2 rounded-lg border border-slate-800/80">
          <span className="text-slate-500 block text-[10px] uppercase font-sans">API Cost</span>
          <span className={`font-bold text-sm ${isOnline ? 'text-purple-300' : 'text-emerald-400'}`}>{costDisplay}</span>
        </div>

        {pipelineMs != null && (
          <div className="bg-slate-950/80 p-2 rounded-lg border border-slate-800/80">
            <span className="text-slate-500 block text-[10px] uppercase font-sans">Pipeline Latency</span>
            <span className="text-cyan-300 font-bold text-base font-mono">
              {formatLatency(pipelineMs)}
            </span>
          </div>
        )}
      </div>

      {/* Latency Stage Breakdown */}
      {(genMs != null || verifyMs != null || rewardMs != null) && (
        <div className="bg-slate-950/80 border border-slate-800/80 rounded-lg p-2 font-mono text-[10px] space-y-1">
          <div className="flex items-center justify-between text-slate-400 font-sans border-b border-slate-800/60 pb-1 mb-1 font-semibold">
            <span>Stage Latency Breakdown</span>
            <Clock className="w-3 h-3 text-cyan-400" />
          </div>

          {genMs != null && (
            <div className="flex justify-between text-slate-300">
              <span className="text-slate-400">{inferenceLabel}</span>
              <span className="text-cyan-300 font-bold">{formatLatency(genMs)}</span>
            </div>
          )}
          {!isOnline && loadMs != null && loadMs > 0 && (
            <div className="flex justify-between text-slate-400 pl-2">
              <span>└ Model Load:</span>
              <span className="text-slate-300">{formatLatency(loadMs)}</span>
            </div>
          )}
          {!isOnline && evalMs != null && evalMs > 0 && (
            <div className="flex justify-between text-slate-400 pl-2">
              <span>└ Token Eval:</span>
              <span className="text-slate-300">{formatLatency(evalMs)}</span>
            </div>
          )}
          {verifyMs != null && (
            <div className="flex justify-between text-slate-300">
              <span className="text-slate-400">Response Verification:</span>
              <span className="text-emerald-400 font-bold">{formatLatency(verifyMs)}</span>
            </div>
          )}
          {rewardMs != null && (
            <div className="flex justify-between text-slate-300">
              <span className="text-slate-400">Reward Computation:</span>
              <span className="text-purple-300 font-bold">{formatLatency(rewardMs)}</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default CostLatencyPanel;
