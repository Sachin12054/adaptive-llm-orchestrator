import React from 'react';
import { Layers, Cpu, GitBranch, ArrowDown, Code, Brain, Zap, CheckCircle2, AlertCircle, Clock } from 'lucide-react';
import { formatLatency } from '../utils/formatLatency';

const ExecutionLevelsPanel = ({ complexPlan }) => {
  if (!complexPlan || !complexPlan.subtasks) {
    return null;
  }

  const getModelBadge = (modelId) => {
    const idLower = (modelId || '').toLowerCase();
    if (idLower.includes('gemini')) {
      return { name: 'Gemini 3.5 Flash', bg: 'bg-purple-500/15 text-purple-300 border-purple-500/30', icon: Cpu };
    }
    if (idLower.includes('mistral')) {
      return { name: 'Mistral Small', bg: 'bg-indigo-500/15 text-indigo-300 border-indigo-500/30', icon: Cpu };
    }
    if (idLower.includes('groq') || idLower.includes('versatile')) {
      return { name: 'Groq LLaMA 3.3', bg: 'bg-cyan-500/15 text-cyan-300 border-cyan-500/30', icon: Code };
    }
    if (idLower.includes('openrouter') || idLower.includes('meta-llama')) {
      return { name: 'OpenRouter LLaMA', bg: 'bg-purple-500/15 text-purple-300 border-purple-500/30', icon: Brain };
    }
    if (idLower.includes('qwen') || idLower.includes('coder')) {
      return { name: 'Qwen Coder 3B', bg: 'bg-cyan-500/15 text-cyan-400 border-cyan-500/30', icon: Code };
    }
    if (idLower.includes('deepseek') || idLower.includes('r1')) {
      return { name: 'DeepSeek R1 7B', bg: 'bg-purple-500/15 text-purple-400 border-purple-500/30', icon: Brain };
    }
    return { name: modelId || 'Gemma 3 4B', bg: 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30', icon: Zap };
  };

  const getStatusBadge = (status, success) => {
    const sUpper = (status || '').toUpperCase();
    if (sUpper === 'COMPLETED' || success) {
      return { label: 'COMPLETED', bg: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40', icon: CheckCircle2 };
    }
    if (sUpper === 'RUNNING') {
      return { label: 'RUNNING', bg: 'bg-purple-500/20 text-purple-300 border-purple-500/40 glow-active', icon: Clock };
    }
    if (sUpper === 'FAILED') {
      return { label: 'FAILED', bg: 'bg-red-500/20 text-red-300 border-red-500/40', icon: AlertCircle };
    }
    return { label: sUpper || 'READY', bg: 'bg-slate-800 text-slate-400 border-slate-700', icon: Clock };
  };

  const levels = complexPlan.execution_levels || [];
  const subtaskMap = new Map((complexPlan.subtasks || []).map((t) => [t.task_id, t]));
  const isExecuted = Boolean(complexPlan.execution_success || complexPlan.total_execution_latency_ms > 0);

  return (
    <div className="glass-panel p-5 space-y-4 border-purple-500/20 shadow-purple-500/5">
      <div className="flex items-center justify-between border-b border-slate-800 pb-3">
        <div className="flex items-center gap-2">
          <Layers className="w-5 h-5 text-purple-400" />
          <div>
            <h4 className="font-semibold text-sm text-slate-100">Complex Task Decomposition & Parallel Multi-LLM Execution</h4>
            <p className="text-[11px] text-slate-400">Dependency DAG breakdown and subtask model routing</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <span className="px-2.5 py-0.5 rounded-full text-xs font-mono bg-purple-500/20 text-purple-300 border border-purple-500/30 font-semibold">
            {complexPlan.total_subtasks} Subtasks
          </span>
          <span className={`px-2.5 py-0.5 rounded-full text-xs font-mono font-semibold border ${isExecuted ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30' : 'bg-amber-500/20 text-amber-300 border-amber-500/30'}`}>
            {isExecuted ? `Parallel Executed (${formatLatency(complexPlan.total_execution_latency_ms)})` : 'Plan Generated'}
          </span>
        </div>
      </div>

      {/* Execution Levels Visualization */}
      <div className="space-y-4">
        {levels.map((levelTasks, levelIdx) => (
          <div key={levelIdx} className="relative">
            <div className="flex items-center gap-2 mb-2">
              <span className="px-2 py-0.5 rounded text-[11px] font-mono font-bold bg-slate-800 text-purple-300 border border-slate-700">
                Execution Level {levelIdx + 1}
              </span>
              <span className="text-[11px] text-slate-500">
                ({levelTasks.length} {levelTasks.length === 1 ? 'task (sequential)' : 'tasks (parallel executable)'})
              </span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
              {levelTasks.map((taskId) => {
                const task = subtaskMap.get(taskId);
                if (!task) return null;
                const modelInfo = getModelBadge(task.assigned_model);
                const statusInfo = getStatusBadge(task.status, task.execution_success);
                const ModelIcon = modelInfo.icon;
                const StatusIcon = statusInfo.icon;

                return (
                  <div
                    key={taskId}
                    className="bg-slate-950/80 border border-slate-800 hover:border-purple-500/40 rounded-xl p-3.5 space-y-2 transition-all shadow-md"
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-1.5">
                        <span className="font-mono text-xs font-bold text-purple-400">{task.task_id}</span>
                        <span className={`px-2 py-0.5 rounded text-[10px] font-mono border flex items-center gap-1 ${statusInfo.bg}`}>
                          <StatusIcon className="w-3 h-3" />
                          {statusInfo.label}
                        </span>
                      </div>
                      <span className={`px-2 py-0.5 rounded text-[10px] font-mono border flex items-center gap-1 ${modelInfo.bg}`}>
                        <ModelIcon className="w-3 h-3" />
                        {modelInfo.name}
                      </span>
                    </div>

                    <p className="text-xs text-slate-300 leading-snug">{task.description}</p>

                    {task.failover_used && (
                      <div className="text-[10px] font-mono flex items-center justify-between bg-amber-500/10 rounded px-2 py-1 border border-amber-500/30 text-amber-300">
                        <span>Failover: <strong className="line-through opacity-70">{task.initial_model}</strong> → <strong className="text-emerald-400 font-bold">{task.assigned_model}</strong></span>
                        <span className="font-bold">Attempt {task.attempts}</span>
                      </div>
                    )}

                    {task.rl_shadow_model && (
                      <div className="text-[10px] font-mono flex items-center justify-between bg-slate-900/60 rounded px-2 py-1 border border-slate-850">
                        <span className="text-slate-400">RL Shadow: <strong className="text-purple-300">{task.rl_shadow_model}</strong></span>
                        <span className={task.rl_agreement ? "text-emerald-400 font-bold" : "text-amber-400 font-bold"}>
                          {task.rl_agreement ? "Agreed ✓" : "Disagreed ✗"}
                        </span>
                      </div>
                    )}

                    <div className="flex items-center justify-between text-[10px] text-slate-500 border-t border-slate-900 pt-2">
                      <span>Category: <strong className="text-slate-400 capitalize">{task.category}</strong></span>
                      {task.latency_ms > 0 && <span>Latency: <strong className="text-purple-300">{formatLatency(task.latency_ms)}</strong></span>}
                      <span>Deps: <strong className="text-cyan-400 font-mono">{task.dependencies?.length ? task.dependencies.join(', ') : 'None'}</strong></span>
                    </div>
                  </div>
                );
              })}
            </div>

            {levelIdx < levels.length - 1 && (
              <div className="flex justify-center my-2 text-purple-500/50">
                <ArrowDown className="w-4 h-4" />
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

export default ExecutionLevelsPanel;
