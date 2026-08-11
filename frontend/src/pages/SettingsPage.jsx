import React from 'react';
import { Settings, Server, HardDrive, ShieldCheck, Activity } from 'lucide-react';

const SettingsPage = ({ backendStatus }) => {
  return (
    <div className="p-6 space-y-6 max-w-4xl mx-auto">
      <div>
        <h3 className="text-lg font-bold text-slate-100 flex items-center gap-2">
          <Settings className="w-5 h-5 text-cyan-400" />
          <span>Platform Settings & Environment Status</span>
        </h3>
        <p className="text-xs text-slate-400">Local server endpoints and infrastructure parameters</p>
      </div>

      <div className="glass-panel p-6 space-y-4">
        <h4 className="font-semibold text-sm text-slate-200 border-b border-slate-800 pb-2">
          Backend API Connection
        </h4>
        <div className="space-y-3 text-xs">
          <div className="flex items-center justify-between">
            <span className="text-slate-400">FastAPI Server URL:</span>
            <span className="font-mono text-cyan-300">http://127.0.0.1:8000</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-slate-400">API Readiness Status:</span>
            <span className={`font-mono font-semibold ${backendStatus ? 'text-emerald-400' : 'text-amber-400'}`}>
              {backendStatus ? 'Connected & Healthy (200 OK)' : 'Connecting / Offline'}
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-slate-400">Active Routing Policy:</span>
            <span className="font-mono text-indigo-400">BaselineAdaptivePolicy</span>
          </div>
        </div>
      </div>

      <div className="glass-panel p-6 space-y-4">
        <h4 className="font-semibold text-sm text-slate-200 border-b border-slate-800 pb-2">
          Local Ollama Service Configuration
        </h4>
        <div className="space-y-3 text-xs">
          <div className="flex items-center justify-between">
            <span className="text-slate-400">Ollama API Endpoint:</span>
            <span className="font-mono text-cyan-300">http://localhost:11434/api/generate</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-slate-400">Execution Mode:</span>
            <span className="font-mono text-emerald-400">100% Local Inference (Zero Cloud API Keys)</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-slate-400">Active Model Map:</span>
            <span className="font-mono text-slate-300">Action 0: gemma-3-4b, Action 1: qwen-coder-3b, Action 2: deepseek-r1-7b</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SettingsPage;
