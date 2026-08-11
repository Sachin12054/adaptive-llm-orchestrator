import React from 'react';
import { BrainCircuit, Database, Award, Eye, ShieldCheck, Cpu } from 'lucide-react';

const RlResearchPage = ({ experienceStatus, rlStatus }) => {
  const currentSize = experienceStatus?.current_size ?? 8;
  const capacity = experienceStatus?.capacity ?? 1000;
  const stateDim = experienceStatus?.state_dim ?? 12;

  const actions = [
    { index: 0, model: 'gemma-3-4b', role: 'General QA, Factual, Explanation, Summarization', badge: 'bg-emerald-500/20 text-emerald-300' },
    { index: 1, model: 'qwen-coder-3b', role: 'Coding, Programming, Scripting', badge: 'bg-cyan-500/20 text-cyan-300' },
    { index: 2, model: 'deepseek-r1-7b', role: 'Complex Reasoning, Mathematics, Analytical Tasks', badge: 'bg-purple-500/20 text-purple-300' },
  ];

  return (
    <div className="p-6 space-y-6 max-w-6xl mx-auto">
      <div>
        <h3 className="text-lg font-bold text-slate-100 flex items-center gap-2">
          <BrainCircuit className="w-5 h-5 text-purple-400" />
          <span>Reinforcement Learning & Shadow Research Framework</span>
        </h3>
        <p className="text-xs text-slate-400">Step 20 Experience Replay Buffer & Step 21 Contextual Bandit Policy</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Experience Replay Buffer Card */}
        <div className="glass-panel p-5 space-y-3">
          <div className="flex items-center gap-2 text-slate-400 border-b border-slate-800 pb-2">
            <Database className="w-4 h-4 text-cyan-400" />
            <h4 className="font-semibold text-xs text-slate-200">Step 20 Experience Buffer</h4>
          </div>
          <div className="space-y-2 text-xs">
            <div className="flex justify-between">
              <span className="text-slate-500">Current Samples:</span>
              <span className="font-mono text-cyan-400 font-bold">{currentSize} / {capacity}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">State Vector (D):</span>
              <span className="font-mono text-emerald-400 font-bold">{stateDim} Dimensions</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Training Threshold:</span>
              <span className="font-mono text-purple-400 font-bold">100 Samples Minimum</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Persistence Path:</span>
              <span className="font-mono text-slate-400 text-[10px]">data/rl/experience_buffer.jsonl</span>
            </div>
          </div>
        </div>

        {/* Step 21 Policy Status */}
        <div className="glass-panel p-5 space-y-3 border-purple-500/20">
          <div className="flex items-center gap-2 text-slate-400 border-b border-slate-800 pb-2">
            <Eye className="w-4 h-4 text-purple-400" />
            <h4 className="font-semibold text-xs text-slate-200">Step 21 Shadow RL Policy</h4>
          </div>
          <div className="space-y-2 text-xs">
            <div className="flex justify-between">
              <span className="text-slate-500">Policy Class:</span>
              <span className="font-mono text-purple-300 font-bold">rl_contextual_bandit_policy</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Execution Mode:</span>
              <span className="font-mono text-purple-400 font-bold">Offline Shadow Evaluation</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Production Override:</span>
              <span className="font-mono text-emerald-400 font-bold">Disabled (Shadow Only)</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Production Policy:</span>
              <span className="font-mono text-cyan-300 font-bold">BaselineAdaptivePolicy</span>
            </div>
          </div>
        </div>

        {/* Step 18 Reward Formula */}
        <div className="glass-panel p-5 space-y-3">
          <div className="flex items-center gap-2 text-slate-400 border-b border-slate-800 pb-2">
            <Award className="w-4 h-4 text-amber-400" />
            <h4 className="font-semibold text-xs text-slate-200">Step 18 Reward Signal Formula</h4>
          </div>
          <div className="space-y-1.5 text-[11px] font-mono text-slate-300">
            <div className="p-2 bg-slate-950/80 rounded border border-slate-800 text-purple-300">
              R = 0.30 S_qual + 0.20 S_cmpl + 0.25 S_rel + 0.15 S_ver + 0.10 S_exec
            </div>
            <p className="text-[10px] text-slate-500 font-sans">
              Exact scalar rewards are recorded in Step 20 without transformation or alteration.
            </p>
          </div>
        </div>
      </div>

      {/* 3-Action Mapping Table */}
      <div className="glass-panel p-5 space-y-4">
        <h4 className="font-semibold text-sm text-slate-200">Contextual Bandit 3-Action Mapping</h4>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {actions.map((act) => (
            <div key={act.index} className="p-3.5 bg-slate-950/60 border border-slate-800 rounded-xl space-y-2">
              <div className="flex items-center justify-between">
                <span className="font-mono font-bold text-xs text-cyan-400">Action Index {act.index}</span>
                <span className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold ${act.badge}`}>
                  {act.model}
                </span>
              </div>
              <p className="text-xs text-slate-400">{act.role}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default RlResearchPage;
