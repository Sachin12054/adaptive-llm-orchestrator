import React from 'react';

const MetricRing = ({
  value = 0,
  maxValue = 100,
  label,
  subtitle,
  size = 72,
  strokeWidth = 6,
  color = 'cyan',
  formattedValue
}) => {
  const numericVal = typeof value === 'number' ? value : parseFloat(value) || 0;
  const percentage = Math.min(Math.max((numericVal / maxValue) * 100, 0), 100);

  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (percentage / 100) * circumference;

  const colorMap = {
    cyan: 'stroke-cyan-400 text-cyan-400',
    purple: 'stroke-purple-400 text-purple-400',
    emerald: 'stroke-emerald-400 text-emerald-400',
    indigo: 'stroke-indigo-400 text-indigo-400',
    rose: 'stroke-rose-400 text-rose-400',
    amber: 'stroke-amber-400 text-amber-400'
  };

  const selectedColor = colorMap[color] || colorMap.cyan;

  const displayText = formattedValue !== undefined
    ? formattedValue
    : (percentage % 1 === 0 ? `${percentage.toFixed(0)}%` : `${percentage.toFixed(1)}%`);

  return (
    <div className="flex flex-col items-center justify-center select-none">
      <div className="relative flex items-center justify-center" style={{ width: size, height: size }}>
        <svg width={size} height={size} className="transform -rotate-90">
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            stroke="currentColor"
            strokeWidth={strokeWidth}
            fill="transparent"
            className="text-slate-800/80"
          />
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            stroke="currentColor"
            strokeWidth={strokeWidth}
            fill="transparent"
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            strokeLinecap="round"
            className={`${selectedColor} transition-all duration-500 ease-out`}
          />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center text-center">
          <span className={`font-mono font-bold text-xs ${selectedColor}`}>
            {displayText}
          </span>
        </div>
      </div>
      {label && (
        <span className="text-[10px] font-mono uppercase tracking-wider text-slate-400 mt-1 text-center font-medium">
          {label}
        </span>
      )}
      {subtitle && (
        <span className="text-[9px] text-slate-500 text-center">
          {subtitle}
        </span>
      )}
    </div>
  );
};

export default MetricRing;
