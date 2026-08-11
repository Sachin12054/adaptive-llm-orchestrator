import React from 'react';
import { Cpu, Activity, RefreshCw, Zap } from 'lucide-react';

const Header = ({ backendStatus, isExecuting, onRefreshStatus, executionMode = 'local', setExecutionMode }) => {
  const isOnline = executionMode === 'online' || executionMode === 'mistral' || executionMode === 'gemini';

  const handleModeChange = (mode) => {
    if (setExecutionMode && mode !== executionMode) {
      setExecutionMode(mode);
    }
  };

  return (
    <header className="h-14 bg-slate-900/90 backdrop-blur-md border-b border-slate-800/90 px-5 flex items-center justify-between flex-shrink-0 z-20">
      <div className="flex items-center gap-3">
        <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-cyan-500 via-indigo-500 to-purple-600 flex items-center justify-center shadow-md shadow-cyan-500/20">
          <Zap className="w-4 h-4 text-white" />
        </div>
        <div>
          <div className="flex items-center gap-2">
            <h2 className="text-sm font-bold text-slate-100 tracking-tight">
              AI Orchestration Platform
            </h2>
            <span className={`text-[10px] px-2 py-0.2 rounded-full border font-mono font-medium ${
              isOnline
                ? 'bg-purple-500/15 text-purple-300 border-purple-500/30'
                : 'bg-cyan-500/10 text-cyan-400 border-cyan-500/20'
            }`}>
              {isOnline ? 'Online API Pool' : '100% Local Ollama'}
            </span>
          </div>
          <p className="text-[11px] text-slate-400 leading-none mt-0.5">
            Intelligent Multi-LLM Orchestration & Workflow Automation
          </p>
        </div>
      </div>

      <div className="flex items-center gap-3">
        {/* Provider Mode Switch Selector */}
        <div className="flex items-center bg-slate-950 p-1 rounded-lg border border-slate-800 text-xs font-mono">
          <button
            onClick={() => handleModeChange('local')}
            className={`px-3 py-1 rounded-md text-[11px] font-bold transition-all ${
              !isOnline
                ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            LOCAL
          </button>
          <button
            onClick={() => handleModeChange('online')}
            className={`px-3 py-1 rounded-md text-[11px] font-bold transition-all ${
              isOnline
                ? 'bg-purple-500/20 text-purple-300 border border-purple-500/40 shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            ONLINE
          </button>
        </div>

        {/* Active Models Quick Pill */}
        <div className="hidden lg:flex items-center gap-2 px-3 py-1 bg-slate-950/70 border border-slate-800 rounded-lg text-xs font-mono">
          <Cpu className="w-3.5 h-3.5 text-cyan-400" />
          <span className="text-slate-400 text-[11px]">Models:</span>
          {isOnline ? (
            <>
              <span className="text-purple-300 font-medium text-[11px]">gemini-2.0-flash</span>
              <span className="text-slate-600">|</span>
              <span className="text-indigo-300 font-medium text-[11px]">mistral-small</span>
              <span className="text-slate-600">|</span>
              <span className="text-cyan-300 font-medium text-[11px]">llama-3.3-70b</span>
            </>
          ) : (
            <>
              <span className="text-emerald-400 font-medium text-[11px]">gemma-3-4b</span>
              <span className="text-slate-600">|</span>
              <span className="text-cyan-400 font-medium text-[11px]">qwen-coder-3b</span>
              <span className="text-slate-600">|</span>
              <span className="text-purple-400 font-medium text-[11px]">deepseek-r1-7b</span>
            </>
          )}
        </div>

        {/* Backend Status Indicator */}
        <div className="flex items-center gap-2 px-3 py-1 bg-slate-950/70 border border-slate-800 rounded-lg text-xs">
          <Activity className={`w-3.5 h-3.5 ${backendStatus ? 'text-emerald-400' : 'text-amber-400'}`} />
          <span className="text-slate-400 text-[11px]">Backend:</span>
          <span className={`font-mono text-[11px] ${backendStatus ? 'text-emerald-400' : 'text-amber-400'}`}>
            {backendStatus ? '127.0.0.1:8000 (Connected)' : 'Connecting...'}
          </span>
          <button
            onClick={onRefreshStatus}
            title="Refresh Backend Status"
            className="p-0.5 hover:bg-slate-800 rounded text-slate-400 hover:text-white transition-colors ml-1"
          >
            <RefreshCw className={`w-3 h-3 ${isExecuting ? 'animate-spin text-cyan-400' : ''}`} />
          </button>
        </div>
      </div>
    </header>
  );
};

export default Header;
