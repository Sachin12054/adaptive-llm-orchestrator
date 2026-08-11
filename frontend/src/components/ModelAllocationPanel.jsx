import React from 'react';
import { Cpu, Zap, Code, Brain, Cloud, Server, Globe, Shield } from 'lucide-react';
import ProgressBar from './ProgressBar';

const DEFAULT_LOCAL_MODELS = [
  { model_id: 'gemma-3-4b', display_name: 'Gemma 3 4B', provider: 'Local Ollama', icon: Zap },
  { model_id: 'qwen-coder-3b', display_name: 'Qwen Coder 3B', provider: 'Local Ollama', icon: Code },
  { model_id: 'deepseek-r1-7b', display_name: 'DeepSeek R1 7B', provider: 'Local Ollama', icon: Brain },
];

const DEFAULT_ONLINE_MODELS = [
  { model_id: 'gemini-2.0-flash', display_name: 'Gemini 2.0 Flash', provider: 'Google Gemini API', icon: Globe },
  { model_id: 'mistral-small-latest', display_name: 'Mistral Small Latest', provider: 'Mistral API', icon: Cloud },
  { model_id: 'llama-3.3-70b-versatile', display_name: 'Groq LLaMA 3.3 70B', provider: 'Groq API', icon: Server },
  { model_id: 'meta-llama/llama-3.3-70b-instruct', display_name: 'OpenRouter LLaMA 3.3 70B', provider: 'OpenRouter API', icon: Shield },
];

const ModelAllocationPanel = ({ executionMode = 'local', selectedModel, candidates }) => {
  const isOnline = executionMode === 'online' || executionMode === 'mistral' || executionMode === 'gemini';

  // Determine active candidate model list based strictly on current executionMode
  let displayCandidates = [];

  if (candidates && Array.isArray(candidates) && candidates.length > 0) {
    const modeFiltered = candidates.filter((c) => {
      const p = (c.provider || '').toLowerCase();
      const m = (c.model_id || '').toLowerCase();
      const isOnlineCand = p.includes('api') || p.includes('gemini') || p.includes('mistral') || p.includes('groq') || p.includes('openrouter') || m.includes('/') || m.includes('flash') || m.includes('versatile');
      return isOnline ? isOnlineCand : !isOnlineCand;
    });

    displayCandidates = modeFiltered.length > 0 ? modeFiltered : (isOnline ? DEFAULT_ONLINE_MODELS : DEFAULT_LOCAL_MODELS);
  } else {
    displayCandidates = isOnline ? DEFAULT_ONLINE_MODELS : DEFAULT_LOCAL_MODELS;
  }

  const headerLabel = isOnline ? 'Online API Pool' : '100% Local Ollama';

  const getIcon = (providerOrId) => {
    const key = (providerOrId || '').toLowerCase();
    if (key.includes('gemini') || key.includes('google')) return Globe;
    if (key.includes('mistral')) return Cloud;
    if (key.includes('groq')) return Server;
    if (key.includes('openrouter')) return Shield;
    if (key.includes('code') || key.includes('qwen')) return Code;
    if (key.includes('deepseek') || key.includes('reason')) return Brain;
    return Zap;
  };

  return (
    <div className="glass-panel p-3.5 space-y-3">
      <div className="flex items-center justify-between border-b border-slate-800/80 pb-2">
        <div className="flex items-center gap-1.5">
          <Cpu className="w-4 h-4 text-cyan-400" />
          <h4 className="font-bold text-[11px] uppercase tracking-wider text-slate-300">Model Allocation & Candidates</h4>
        </div>
        <span className={`text-[10px] font-mono px-2 py-0.5 rounded border font-semibold ${
          isOnline
            ? 'bg-purple-500/15 text-purple-300 border-purple-500/30'
            : 'bg-cyan-500/15 text-cyan-300 border-cyan-500/30'
        }`}>
          {headerLabel}
        </span>
      </div>

      <div className="space-y-2.5">
        {displayCandidates.map((cand) => {
          const modelId = cand.model_id;
          const provider = cand.provider || (isOnline ? 'Online API' : 'Local Ollama');
          const displayName = cand.display_name || cand.name || modelId;
          const IconComponent = getIcon(provider + ' ' + modelId);

          const isSelected = selectedModel === modelId;
          const isEligible = cand.eligible != null ? cand.eligible : true;
          const candidateScore = cand.candidate_score != null
            ? (cand.candidate_score * 100).toFixed(1)
            : (isSelected ? '100.0' : null);

          // Status Badge Logic
          let badgeText = '[ ELIGIBLE ]';
          let badgeStyle = 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30';

          if (isSelected) {
            badgeText = '[ SELECTED ]';
            badgeStyle = 'bg-cyan-500/20 text-cyan-300 border-cyan-500/40 font-bold';
          } else if (!isEligible) {
            const reason = (cand.ineligible_reason || '').toLowerCase();
            if (reason.includes('not_configured') || reason.includes('unconfigured')) {
              badgeText = '[ NOT CONFIGURED ]';
              badgeStyle = 'bg-amber-500/15 text-amber-400 border-amber-500/30';
            } else if (reason.includes('excluded')) {
              badgeText = '[ EXCLUDED ]';
              badgeStyle = 'bg-rose-500/20 text-rose-400 border-rose-500/40 font-bold';
            } else {
              badgeText = '[ UNAVAILABLE ]';
              badgeStyle = 'bg-slate-800/80 text-slate-400 border-slate-700';
            }
          }

          return (
            <div
              key={modelId}
              className={`p-2.5 rounded-lg border transition-all ${
                isSelected
                  ? 'bg-slate-900 border-cyan-500/50 shadow-md ring-1 ring-cyan-500/30'
                  : (isEligible ? 'bg-slate-950/60 border-slate-800/80' : 'bg-slate-950/40 border-slate-900 opacity-70')
              }`}
            >
              <div className="flex items-center justify-between mb-1.5">
                <div className="flex items-center gap-2 min-w-0 flex-1">
                  <div className={`p-1 rounded flex-shrink-0 ${isSelected ? 'bg-cyan-500/20 text-cyan-300' : 'bg-slate-800 text-slate-400'}`}>
                    <IconComponent className="w-3.5 h-3.5" />
                  </div>
                  <div className="min-w-0 flex-1">
                    <h5 className="font-bold text-xs text-slate-100 font-sans truncate">{displayName}</h5>
                    <div className="flex items-center gap-1.5 text-[10px] text-slate-400 font-mono truncate">
                      <span className="truncate">{modelId}</span>
                      <span className="text-slate-600">•</span>
                      <span className="text-cyan-400/80 font-medium truncate">{provider}</span>
                    </div>
                  </div>
                </div>

                <div className="flex items-center gap-2 flex-shrink-0">
                  <span className={`px-2 py-0.5 rounded text-[10px] font-mono ${badgeStyle}`}>
                    {badgeText}
                  </span>
                  {candidateScore && (
                    <span className="font-mono font-bold text-xs text-emerald-400">
                      {candidateScore}%
                    </span>
                  )}
                </div>
              </div>

              {/* Factor Score Progress Bars */}
              {isEligible && cand.capability_score != null ? (
                <div className="grid grid-cols-2 gap-2 pt-1 border-t border-slate-800/40">
                  <ProgressBar value={cand.capability_score} maxValue={1.0} label="Capability" color="cyan" />
                  <ProgressBar value={cand.complexity_fit_score} maxValue={1.0} label="Complexity" color="purple" />
                  <ProgressBar value={cand.resource_fit_score} maxValue={1.0} label="Resource" color="emerald" />
                  <ProgressBar value={cand.context_fit_score} maxValue={1.0} label="Context" color="indigo" />
                </div>
              ) : !isEligible && cand.ineligible_reason ? (
                <div className="bg-rose-950/20 border border-rose-800/30 rounded p-2 mt-1.5 font-mono text-[10px] space-y-1">
                  <div className="text-rose-400 font-bold flex justify-between uppercase">
                    <span>CANDIDATE STATUS</span>
                  </div>
                  <div className="text-rose-300/80 font-sans text-[10px] leading-tight">
                    {cand.ineligible_reason}
                  </div>
                </div>
              ) : null}
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default ModelAllocationPanel;
