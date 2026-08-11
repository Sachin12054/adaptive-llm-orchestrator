import React from 'react';
import { Play, Sparkles, Code, Compass, Brain } from 'lucide-react';

const PromptInputBar = ({ prompt, setPrompt, onExecute, isExecuting }) => {
  const presets = [
    {
      label: 'Simple QA',
      type: 'SIMPLE',
      icon: Compass,
      color: 'border-emerald-500/30 text-emerald-400 bg-emerald-500/10 hover:bg-emerald-500/20',
      text: 'What is Python?',
    },
    {
      label: 'CSV Coding',
      type: 'MEDIUM',
      icon: Code,
      color: 'border-cyan-500/30 text-cyan-400 bg-cyan-500/10 hover:bg-cyan-500/20',
      text: 'Write Python code to read a CSV using pandas.',
    },
    {
      label: 'ML Pipeline',
      type: 'COMPLEX',
      icon: Brain,
      color: 'border-purple-500/30 text-purple-400 bg-purple-500/10 hover:bg-purple-500/20',
      text: 'Design a distributed Python ML pipeline...',
    },
  ];

  return (
    <div className="glass-panel p-3 border-purple-500/20 shadow-purple-500/5">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-1.5 text-xs font-semibold text-slate-200">
          <Sparkles className="w-4 h-4 text-purple-400" />
          <span>User Prompt & Request Execution</span>
        </div>
        <div className="flex items-center gap-1.5 text-[11px]">
          <span className="text-slate-500">Presets:</span>
          {presets.map((p, idx) => (
            <button
              key={idx}
              onClick={() => setPrompt(p.text)}
              className={`px-2 py-0.5 rounded border text-[11px] font-medium transition-all ${p.color}`}
            >
              {p.label}
            </button>
          ))}
        </div>
      </div>

      <div className="flex items-center gap-3">
        <input
          type="text"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !isExecuting && prompt.trim()) {
              onExecute();
            }
          }}
          placeholder="Type user prompt (e.g. 'Write Python code to read a CSV using pandas...')"
          className="flex-1 bg-slate-950/90 border border-slate-800 rounded-lg px-3.5 py-2 text-xs text-slate-100 placeholder-slate-500 focus:outline-none focus:border-purple-500/60 focus:ring-1 focus:ring-purple-500/60 font-sans"
        />
        <button
          onClick={onExecute}
          disabled={isExecuting || !prompt.trim()}
          className={`px-5 py-2 rounded-lg font-semibold text-xs flex items-center gap-2 transition-all shadow-lg flex-shrink-0 ${
            isExecuting || !prompt.trim()
              ? 'bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-700'
              : 'bg-gradient-to-r from-purple-600 via-indigo-600 to-cyan-500 text-white hover:from-purple-500 hover:to-cyan-400 shadow-purple-500/25 active:scale-95 border border-purple-400/30'
          }`}
        >
          {isExecuting ? (
            <>
              <div className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
              <span>Orchestrating...</span>
            </>
          ) : (
            <>
              <Play className="w-3.5 h-3.5 fill-current" />
              <span>Execute Pipeline</span>
            </>
          )}
        </button>
      </div>
    </div>
  );
};

export default PromptInputBar;
