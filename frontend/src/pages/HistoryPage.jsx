import React from 'react';
import { History, Clock, CheckCircle2, Award, Cpu } from 'lucide-react';

const HistoryPage = ({ historyItems, onSelectHistoryItem }) => {
  return (
    <div className="p-6 space-y-6 max-w-6xl mx-auto">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-bold text-slate-100 flex items-center gap-2">
            <History className="w-5 h-5 text-cyan-400" />
            <span>Orchestration Request History</span>
          </h3>
          <p className="text-xs text-slate-400">Past executed requests, routing decisions, and rewards</p>
        </div>
        <span className="px-3 py-1 bg-slate-900 border border-slate-800 rounded-lg text-xs font-mono text-cyan-400">
          {historyItems.length} Requests Saved
        </span>
      </div>

      <div className="glass-panel overflow-hidden">
        {historyItems.length === 0 ? (
          <div className="p-12 text-center text-slate-500 italic text-sm">
            No request history recorded yet. Execute requests on the Dashboard to view history.
          </div>
        ) : (
          <div className="divide-y divide-slate-800/80">
            {historyItems.map((item, idx) => (
              <div
                key={idx}
                onClick={() => onSelectHistoryItem(item)}
                className="p-4 hover:bg-slate-800/40 cursor-pointer transition-colors flex items-center justify-between gap-4"
              >
                <div className="space-y-1 flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-[10px] text-slate-500">{item.timestamp}</span>
                    <span className={`px-2 py-0.5 rounded text-[10px] font-mono font-semibold border ${
                      item.complexity === 'COMPLEX'
                        ? 'bg-purple-500/15 text-purple-400 border-purple-500/30'
                        : item.complexity === 'MEDIUM'
                        ? 'bg-cyan-500/15 text-cyan-400 border-cyan-500/30'
                        : 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30'
                    }`}>
                      {item.complexity}
                    </span>
                  </div>
                  <p className="text-xs font-medium text-slate-200 truncate">"{item.prompt}"</p>
                </div>

                <div className="flex items-center gap-4 text-xs font-mono">
                  <div className="text-right">
                    <span className="text-[10px] text-slate-500 block">Selected Model</span>
                    <span className="text-cyan-300 font-semibold">{item.model}</span>
                  </div>
                  <div className="text-right">
                    <span className="text-[10px] text-slate-500 block">Reward</span>
                    <span className="text-purple-300 font-semibold">{item.reward}</span>
                  </div>
                  <div className="text-right">
                    <span className="text-[10px] text-slate-500 block">Latency</span>
                    <span className="text-slate-300">{item.latency}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default HistoryPage;
