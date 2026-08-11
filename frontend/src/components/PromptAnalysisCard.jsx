import React from 'react';
import { Target } from 'lucide-react';
import MetricRing from './MetricRing';

const PromptAnalysisCard = ({ decisionData, complexPlanData }) => {
  if (!decisionData && !complexPlanData) {
    return (
      <div className="glass-panel p-3.5 space-y-2">
        <div className="flex items-center gap-1.5 border-b border-slate-800/80 pb-2">
          <Target className="w-4 h-4 text-cyan-400" />
          <h4 className="font-bold text-[11px] uppercase tracking-wider text-slate-300">Prompt Analysis</h4>
        </div>
        <p className="text-[11px] text-slate-500 italic py-4 text-center">
          Execute prompt to run semantic analysis
        </p>
      </div>
    );
  }

  const trace = decisionData?.decision_trace;
  const intentInfo = decisionData?.intent_info;
  const complexityInfo = decisionData?.complexity_info;

  const primaryIntent = trace?.intent || intentInfo?.intent || (complexPlanData ? 'complex_workflow' : null);
  const topSimilarity = intentInfo?.top_similarity;
  const confidencePercent = topSimilarity != null ? topSimilarity * 100 : null;

  const isComplex = complexPlanData?.is_complex || trace?.complexity_level === 'very_high' || complexityInfo?.complexity_level === 'very_high';
  const rawComplexityLevel = isComplex
    ? 'COMPLEX'
    : (trace?.complexity_level || complexityInfo?.complexity_level || 'SIMPLE');
  const complexityLevel = rawComplexityLevel.toUpperCase();

  const rawScore = trace?.complexity_score ?? complexityInfo?.complexity_score;
  const complexityScorePercent = rawScore != null ? rawScore * 100 : null;

  const getComplexityBadgeClass = (level) => {
    switch (level) {
      case 'COMPLEX':
      case 'VERY_HIGH':
        return 'bg-purple-500/15 text-purple-300 border-purple-500/30';
      case 'MEDIUM':
      case 'HIGH':
        return 'bg-cyan-500/15 text-cyan-300 border-cyan-500/30';
      default:
        return 'bg-emerald-500/15 text-emerald-300 border-emerald-500/30';
    }
  };

  return (
    <div className="glass-panel p-3.5 space-y-3">
      <div className="flex items-center justify-between border-b border-slate-800/80 pb-2">
        <div className="flex items-center gap-1.5">
          <Target className="w-4 h-4 text-cyan-400" />
          <h4 className="font-bold text-[11px] uppercase tracking-wider text-slate-300">Prompt Analysis</h4>
        </div>
        <div className="flex items-center gap-1.5">
          {intentInfo?.is_ambiguous === true && (
            <span className="px-1.5 py-0.5 rounded bg-amber-500/15 text-amber-300 border border-amber-500/30 font-mono text-[9px] font-bold">
              ● AMBIGUOUS
            </span>
          )}
          <span className={`px-2 py-0.5 rounded text-[10px] font-mono font-semibold border ${getComplexityBadgeClass(complexityLevel)}`}>
            {complexityLevel}
          </span>
        </div>
      </div>

      <div className="flex items-center justify-between gap-2">
        {/* Left Side: Intent & Embedding Info */}
        <div className="space-y-2 text-xs flex-1">
          {primaryIntent && (
            <div>
              <span className="text-slate-500 block text-[10px] uppercase font-mono mb-0.5">Intent</span>
              <span className="px-2 py-0.5 rounded bg-cyan-500/15 text-cyan-300 border border-cyan-500/30 font-mono text-xs font-bold capitalize inline-block">
                {primaryIntent}
              </span>
            </div>
          )}

          <div className="bg-slate-950/80 p-2 rounded-lg border border-slate-800/80 font-mono text-[10px] space-y-1">
            <div className="flex justify-between">
              <span className="text-slate-500">Embedding:</span>
              <span className="text-emerald-400 font-bold">BGE-M3 (1024D)</span>
            </div>
            {complexityScorePercent != null && (
              <div className="flex justify-between">
                <span className="text-slate-500">Complexity:</span>
                <span className="text-purple-300 font-bold">{complexityScorePercent.toFixed(1)}%</span>
              </div>
            )}
          </div>
        </div>

        {/* Right Side: Confidence Ring */}
        {confidencePercent != null && (
          <div className="flex-shrink-0">
            <MetricRing
              value={confidencePercent}
              label="CONFIDENCE"
              size={64}
              strokeWidth={5}
              color="cyan"
              formattedValue={`${confidencePercent.toFixed(1)}%`}
            />
          </div>
        )}
      </div>
    </div>
  );
};

export default PromptAnalysisCard;