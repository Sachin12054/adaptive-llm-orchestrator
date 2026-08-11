import React from 'react';
import { Cpu, Eye } from 'lucide-react';
import MetricRing from './MetricRing';

const DecisionEngineCard = ({ decisionData, complexPlanData, executionMode = 'local', generationData }) => {
  const isComplex = Boolean(
    complexPlanData?.is_complex ||
    decisionData?.decision_trace?.complexity_level === 'very_high' ||
    decisionData?.complexity_info?.complexity_level === 'very_high'
  );

  if (!decisionData && !complexPlanData) {
    return (
      <div className="glass-panel p-3.5 space-y-2">
        <div className="flex items-center gap-1.5 border-b border-slate-800/80 pb-2">
          <Cpu className="w-4 h-4 text-indigo-400" />
          <h4 className="font-bold text-[11px] uppercase tracking-wider text-slate-300">Adaptive Decision Engine</h4>
        </div>
        <p className="text-[11px] text-slate-500 italic py-4 text-center">
          Execute prompt to evaluate candidate routing
        </p>
      </div>
    );
  }

  const selectedModel = decisionData?.selected_model || (isComplex ? 'Multi-LLM Task Plan' : null);

  const decisionScoreRaw = decisionData?.decision_score;
  const decisionScorePercent = decisionScoreRaw != null ? decisionScoreRaw * 100 : null;

  const policyName = decisionData?.policy || 'BaselineAdaptivePolicy';
  const shadowRl = decisionData?.shadow_rl_decision || null;
  const isOnline = executionMode === 'online' || executionMode === 'mistral' || executionMode === 'gemini';
  const providerName = generationData?.provider || (isOnline ? 'Online Cloud API' : 'Local Ollama');

  return (
    <div className="glass-panel p-3.5 space-y-3 border-indigo-500/20 shadow-indigo-500/5">
      <div className="flex items-center justify-between border-b border-slate-800/80 pb-2">
        <div className="flex items-center gap-1.5">
          <Cpu className="w-4 h-4 text-indigo-400" />
          <h4 className="font-bold text-[11px] uppercase tracking-wider text-slate-300">Adaptive Decision Engine</h4>
        </div>
        <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-indigo-500/15 text-indigo-300 border border-indigo-500/30 font-semibold">
          {policyName}
        </span>
      </div>

      <div className="flex items-center justify-between gap-2">
        {/* Left Side: Winning Production Model Summary */}
        <div className="space-y-2 text-xs flex-1">
          {selectedModel && (
            <div>
              <span className="text-slate-500 block text-[10px] uppercase font-mono mb-0.5">Selected Model</span>
              <span className="font-mono text-cyan-300 font-bold text-xs bg-slate-950/80 px-2 py-1 rounded border border-slate-800 block truncate">
                {selectedModel}
              </span>
            </div>
          )}

          <div className="bg-slate-950/80 p-2 rounded-lg border border-slate-800/80 text-[10px] font-mono space-y-1">
            <div className="flex justify-between">
              <span className="text-slate-500">Execution Mode:</span>
              <span className={`font-bold ${isOnline ? 'text-purple-300' : 'text-cyan-300'}`}>
                {isOnline ? 'ONLINE' : 'LOCAL'}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Provider:</span>
              <span className="text-slate-200 font-bold truncate max-w-[110px]">{providerName}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Routing Mode:</span>
              <span className="text-cyan-400 font-bold">AUTOMATIC</span>
            </div>
          </div>
        </div>

        {/* Right Side: Decision Score Ring */}
        {decisionScorePercent != null && (
          <div className="flex-shrink-0">
            <MetricRing
              value={decisionScorePercent}
              label="DECISION SCORE"
              size={68}
              strokeWidth={5}
              color="purple"
              formattedValue={`${decisionScorePercent.toFixed(1)}%`}
            />
          </div>
        )}
      </div>

      {/* Shadow RL Policy Summary */}
      <div className="bg-purple-950/20 border border-purple-800/40 rounded-lg p-2.5 space-y-1.5 text-[10px] font-mono">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1 text-purple-300 font-bold">
            <Eye className="w-3 h-3 text-purple-400" />
            <span>SHADOW RL: RLContextualBanditPolicy</span>
          </div>
          <span className="px-1.5 py-0.2 rounded bg-purple-500/20 text-purple-300 font-semibold">
            SHADOW MODE
          </span>
        </div>

        <div className="grid grid-cols-2 gap-2 text-slate-300 pt-1 border-t border-purple-800/30">
          <div>Proposed: <span className="text-purple-200 font-bold">{shadowRl?.proposed_model || (isComplex ? 'Multi-LLM Plan' : 'N/A')}</span></div>
          {shadowRl?.predicted_reward != null && (
            <div>Predicted Reward: <span className="text-purple-300 font-bold">{shadowRl.predicted_reward.toFixed(4)}</span></div>
          )}
          <div>Production Override: <span className="text-emerald-400 font-bold">NO</span></div>
          <div>State Vector: <span className="text-cyan-400 font-bold">12D</span></div>
        </div>
      </div>
    </div>
  );
};

export default DecisionEngineCard;
