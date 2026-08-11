import React from 'react';
import { Terminal } from 'lucide-react';
import { formatLatency } from '../utils/formatLatency';

const formatLogMessage = (msg) => {
  if (!msg) return '';
  return msg.replace(/(\d+(?:\.\d+)?)\s*ms/g, (match, val) => {
    const num = parseFloat(val);
    return Number.isFinite(num) ? formatLatency(num) : match;
  });
};

const ExecutionLogPanel = ({ logs }) => {
  return (
    <div className="glass-panel p-3 flex flex-col h-full overflow-hidden">
      <div className="flex items-center justify-between border-b border-slate-800/80 pb-1.5 flex-shrink-0">
        <div className="flex items-center gap-1.5 text-slate-300">
          <Terminal className="w-3.5 h-3.5 text-cyan-400" />
          <h4 className="font-bold text-[11px] uppercase tracking-wider text-slate-300">Execution Log Trace</h4>
        </div>
        {logs?.length > 0 && (
          <span className="text-[10px] text-slate-400 font-mono">
            {logs.length} events
          </span>
        )}
      </div>

      <div className="flex-1 overflow-y-auto font-mono text-[11px] space-y-1 mt-2 pr-1 min-h-0">
        {(!logs || logs.length === 0) ? (
          <p className="text-slate-500 italic py-4 text-center text-xs">
            Execute a prompt to view real-time trace events.
          </p>
        ) : (
          logs.map((log, idx) => (
            <div key={idx} className="flex items-start gap-2 text-slate-300 leading-tight">
              <span className="text-slate-500 flex-shrink-0 text-[10px]">[{log.time}]</span>
              <span className="text-emerald-400 flex-shrink-0 font-bold">✓</span>
              <span className="break-all">{formatLogMessage(log.message)}</span>
            </div>
          ))
        )}
      </div>
    </div>
  );
};

export default ExecutionLogPanel;
