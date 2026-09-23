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

  const rlSelectedModel = decisionData?.selected_model || (isComplex ? 'Multi-LLM Task Plan' : null);
  const actualExecutedModel = generationData?.model_id || complexPlanData?.final_model || complexPlanData?.actual_models?.[0] || null;
  const actualProvider = generationData?.provider || complexPlanData?.final_provider || complexPlanData?.actual_providers?.[0] || null;
  const providerFailover = Boolean(generationData?.failover_used || complexPlanData?.failover_used);

  const decisionScoreRaw = decisionData?.decision_score;
  const decisionScorePercent = decisionScoreRaw != null ? decisionScoreRaw * 100 : null;

  const policyName = decisionData?.policy || 'BaselineAdaptivePolicy';
  const shadowRl = decisionData?.shadow_rl_decision || null;
  const isOnline = executionMode === 'online' || executionMode === 'mistral' || executionMode === 'gemini';
  const providerName = actualProvider || 'Not recorded';

  const isRLProduction = policyName === 'rl_contextual_bandit_policy' || shadowRl?.production_policy === 'rl_contextual_bandit_policy';

  return (
    <div className="glass-panel p-3.5 space-y-3 border-indigo-500/20 shadow-indigo-500/5">
      <div className="flex items-center justify-between border-b border-slate-800/80 pb-2">
        <div className="flex items-center gap-1.5">
          <Cpu className="w-4 h-4 text-indigo-400" />
          <h4 className="font-bold text-[11px] uppercase tracking-wider text-slate-300">Adaptive Decision Engine</h4>
        </div>
        <span className={`px-2 py-0.5 rounded text-[10px] font-mono border font-semibold ${
          isRLProduction
            ? 'bg-purple-500/15 text-purple-300 border-purple-500/30'
            : 'bg-indigo-500/15 text-indigo-300 border-indigo-500/30'
        }`}>
          {isRLProduction ? 'RL Contextual Bandit (Production)' : policyName}
        </span>
      </div>

      <div className="flex items-center justify-between gap-2">
        {/* Left Side: Winning Production Model Summary */}
        <div className="space-y-2 text-xs flex-1">
          {actualExecutedModel && (
            <div>
              <span className="text-slate-500 block text-[10px] uppercase font-mono mb-0.5">Actual Executed Model</span>
              <span className="font-mono text-cyan-300 font-bold text-xs bg-slate-950/80 px-2 py-1 rounded border border-slate-800 block truncate">
                {actualExecutedModel}
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
              <span className="text-slate-500">Production Policy:</span>
              <span className="text-purple-300 font-bold">{isRLProduction ? 'RL BANDIT' : 'BASELINE'}</span>
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
              color={isRLProduction ? "purple" : "indigo"}
              formattedValue={`${decisionScorePercent.toFixed(1)}%`}
            />
          </div>
        )}
      </div>

      {/* Active Production RL Policy Panel */}
      <div className="bg-purple-950/20 border border-purple-800/40 rounded-lg p-2.5 space-y-1.5 text-[10px] font-mono">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1 text-purple-300 font-bold">
            <Eye className="w-3 h-3 text-purple-400" />
            <span>PRODUCTION RL: RLContextualBanditPolicy (v2.0.0)</span>
          </div>
          <span className={`px-1.5 py-0.2 rounded font-semibold ${
            shadowRl?.fallback_used
              ? 'bg-amber-500/20 text-amber-300 border border-amber-500/30'
              : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
          }`}>
            {shadowRl?.fallback_used ? 'BASELINE FALLBACK' : 'PRODUCTION ACTIVE'}
          </span>
        </div>

        <div className="grid grid-cols-2 gap-2 text-slate-300 pt-1 border-t border-purple-800/30">
          <div>RL Selected: <span className="text-purple-200 font-bold">{shadowRl?.rl_selected_model || shadowRl?.proposed_model || (isComplex ? 'Multi-LLM Plan' : rlSelectedModel)}</span></div>
          <div>Actual Executed: <span className="text-cyan-300 font-bold">{actualExecutedModel}</span></div>
              <div>Provider Failover: <span className={providerFailover ? "text-amber-400 font-bold" : "text-emerald-400 font-bold"}>{providerFailover ? 'YES' : 'NO'}</span></div>
              <div>Baseline Fallback: <span className={shadowRl?.fallback_used ? "text-amber-400 font-bold" : "text-emerald-400 font-bold"}>{shadowRl?.fallback_used ? 'YES' : 'NO'}</span></div>
              <div>Provider: <span className="text-cyan-400 font-bold">{providerName}</span></div>
        </div>
      </div>
    </div>
  );
};

export default DecisionEngineCard;
