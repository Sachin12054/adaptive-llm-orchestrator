import React from 'react';

const ProgressBar = ({
  value = 0,
  maxValue = 1.0,
  label,
  color = 'cyan',
  formattedValue,
  height = 'h-1.5'
}) => {
  const numericVal = typeof value === 'number' ? value : parseFloat(value) || 0;
  const percentage = Math.min(Math.max((numericVal / maxValue) * 100, 0), 100);

  const bgMap = {
    cyan: 'bg-cyan-400',
    purple: 'bg-purple-400',
    emerald: 'bg-emerald-400',
    indigo: 'bg-indigo-400',
    rose: 'bg-rose-400',
    amber: 'bg-amber-400'
  };

  const selectedBg = bgMap[color] || bgMap.cyan;

  const displayVal = formattedValue !== undefined
    ? formattedValue
    : (maxValue === 1.0 ? `${(numericVal * 100).toFixed(0)}%` : `${numericVal.toFixed(0)}%`);

  return (
    <div className="space-y-0.5 text-[10px] font-mono select-none">
      <div className="flex items-center justify-between">
        <span className="text-slate-400 font-sans">{label}</span>
        <span className="text-slate-200 font-bold">{displayVal}</span>
      </div>
      <div className={`w-full bg-slate-950 rounded-full ${height} overflow-hidden border border-slate-800/80`}>
        <div
          className={`${selectedBg} ${height} rounded-full transition-all duration-500 ease-out`}
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
};

export default ProgressBar;
