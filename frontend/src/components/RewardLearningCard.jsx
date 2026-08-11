import React from 'react';
import { Award, BrainCircuit } from 'lucide-react';
import MetricRing from './MetricRing';
import ProgressBar from './ProgressBar';

const RewardLearningCard = ({ orchestrationRes, experienceStatus }) => {
  const rewardData = orchestrationRes?.reward;
  const rewardVal = rewardData?.reward;
  const rewardPercent = rewardVal != null ? rewardVal * 100 : null;

  const breakdown = rewardData?.reward_breakdown;

  const transitions = experienceStatus?.total_transitions;
  const maxCapacity = experienceStatus?.max_capacity || 1000;
  const isBufferReady = transitions != null ? transitions >= (experienceStatus?.training_start_threshold || 10) : false;

  return (
    <div className="glass-panel p-3.5 space-y-3 border-purple-500/20 shadow-purple-500/5">
      <div className="flex items-center justify-between border-b border-slate-800/80 pb-2">
        <div className="flex items-center gap-1.5">
          <Award className="w-4 h-4 text-purple-400" />
          <h4 className="font-bold text-[11px] uppercase tracking-wider text-slate-300">Reward & Learning Engine</h4>
        </div>
        <span className="text-[10px] px-2 py-0.5 rounded bg-purple-500/15 text-purple-300 border border-purple-500/30 font-mono font-semibold">
          Step 18 & 20
        </span>
      </div>

      <div className="flex items-center justify-between gap-2">
        {/* Left Side: Replay Buffer & Readiness */}
        <div className="space-y-2 flex-1">
          {transitions != null && (
            <div className="bg-slate-950/80 p-2 rounded-lg border border-slate-800/80 space-y-1 font-mono text-[10px]">
              <div className="flex justify-between text-slate-400">
                <span>Replay Buffer:</span>
                <span className="text-purple-300 font-bold">{transitions} / {maxCapacity}</span>
              </div>
              <ProgressBar
                value={transitions}
                maxValue={maxCapacity}
                label="Buffer Utilization"
                color="purple"
                formattedValue={`${((transitions / maxCapacity) * 100).toFixed(1)}%`}
              />
              <div className="flex justify-between items-center pt-1 border-t border-slate-800/60">
                <span className="text-slate-500">Training Status:</span>
                <span className={`font-bold ${isBufferReady ? 'text-emerald-400' : 'text-amber-400'}`}>
                  {isBufferReady ? 'READY' : 'COLLECTING'}
                </span>
              </div>
            </div>
          )}
        </div>

        {/* Right Side: Step 18 Reward Ring */}
        {rewardPercent != null && (
          <div className="flex-shrink-0">
            <MetricRing
              value={rewardPercent}
              label="STEP 18 REWARD"
              size={68}
              strokeWidth={5}
              color="purple"
              formattedValue={`${rewardPercent.toFixed(1)}%`}
            />
          </div>
        )}
      </div>

      {/* Step 18 Detailed Component Breakdown (Quality, Completeness, Relevance, Verification, Execution) */}
      {breakdown && (
        <div className="bg-slate-950/80 border border-slate-800/80 rounded-lg p-2.5 space-y-1.5 font-mono text-[10px]">
          <span className="text-slate-400 font-sans block text-[10px] font-semibold mb-1">
            Reward Formula Breakdown (Step 18)
          </span>
          <div className="grid grid-cols-2 gap-2">
            {breakdown.quality && (
              <ProgressBar value={breakdown.quality.score} maxValue={1.0} label="Quality (0.30)" color="cyan" />
            )}
            {breakdown.completeness && (
              <ProgressBar value={breakdown.completeness.score} maxValue={1.0} label="Completeness (0.20)" color="purple" />
            )}
            {breakdown.relevance && (
              <ProgressBar value={breakdown.relevance.score} maxValue={1.0} label="Relevance (0.25)" color="emerald" />
            )}
            {breakdown.verification && (
              <ProgressBar value={breakdown.verification.score} maxValue={1.0} label="Verification (0.15)" color="indigo" />
            )}
            {breakdown.execution && (
              <ProgressBar value={breakdown.execution.score} maxValue={1.0} label="Execution (0.10)" color="amber" />
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default RewardLearningCard;
