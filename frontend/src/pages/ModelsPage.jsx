import React from 'react';
import { Cpu, Zap, Code, Brain, CheckCircle2, ShieldCheck } from 'lucide-react';

const ModelsPage = ({ modelsList }) => {
  const models = [
    {
      id: 'gemma-3-4b',
      display_name: 'Gemma 3 4B',
      provider: 'ollama',
      type: 'llm',
      icon: Zap,
      accent: 'border-emerald-500/40 bg-emerald-500/5 text-emerald-400',
      badge: 'bg-emerald-500/20 text-emerald-300',
      capabilities: ['general_qa', 'factual', 'explanation', 'conversation', 'summarization'],
      context: '8,192 tokens',
      execution_mode: 'local',
      status: 'Configured & Ready',
      endpoint: 'http://localhost:11434/api/generate',
      actionIndex: 0
    },
    {
      id: 'qwen-coder-3b',
      display_name: 'Qwen Coder 3B',
      provider: 'ollama',
      type: 'llm',
      icon: Code,
      accent: 'border-cyan-500/40 bg-cyan-500/5 text-cyan-400',
      badge: 'bg-cyan-500/20 text-cyan-300',
      capabilities: ['coding', 'programming', 'code_explanation', 'scripting'],
      context: '8,192 tokens',
      execution_mode: 'local',
      status: 'Configured & Ready',
      endpoint: 'http://localhost:11434/api/generate',
      actionIndex: 1
    },
    {
      id: 'deepseek-r1-7b',
      display_name: 'DeepSeek R1 7B',
      provider: 'ollama',
      type: 'llm',
      icon: Brain,
      accent: 'border-purple-500/40 bg-purple-500/5 text-purple-400',
      badge: 'bg-purple-500/20 text-purple-300',
      capabilities: ['reasoning', 'mathematics', 'data_analysis', 'analytical'],
      context: '8,192 tokens',
      execution_mode: 'local',
      status: 'Configured & Ready',
      endpoint: 'http://localhost:11434/api/generate',
      actionIndex: 2
    },
    {
      id: 'BAAI/bge-m3',
      display_name: 'BGE-M3 Dense Embedding',
      provider: 'huggingface',
      type: 'embedding',
      icon: Cpu,
      accent: 'border-slate-700 bg-slate-900/60 text-slate-300',
      badge: 'bg-slate-800 text-slate-400',
      capabilities: ['semantic_understanding', 'dense_embedding', '1024_dim_vectors'],
      context: '1,024 dim',
      execution_mode: 'local',
      status: 'Active Embedding Model',
      endpoint: 'Local PyTorch Sentence-Transformers',
      actionIndex: 'N/A'
    }
  ];

  return (
    <div className="p-6 space-y-6 max-w-6xl mx-auto">
      <div>
        <h3 className="text-lg font-bold text-slate-100 flex items-center gap-2">
          <Cpu className="w-5 h-5 text-cyan-400" />
          <span>Active Model Registry & Local Providers</span>
        </h3>
        <p className="text-xs text-slate-400">100% Local Ollama Model Allocation and Capabilities</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {models.map((m) => {
          const Icon = m.icon;
          return (
            <div key={m.id} className={`glass-panel p-5 space-y-4 border ${m.accent}`}>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className={`p-2 rounded-xl ${m.badge}`}>
                    <Icon className="w-5 h-5" />
                  </div>
                  <div>
                    <h4 className="font-bold text-sm text-slate-100">{m.display_name}</h4>
                    <span className="text-xs font-mono text-slate-400">{m.id}</span>
                  </div>
                </div>
                <span className="px-2.5 py-1 rounded-full text-xs font-mono bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 flex items-center gap-1 font-medium">
                  <CheckCircle2 className="w-3 h-3 text-emerald-400" /> {m.status}
                </span>
              </div>

              <div className="space-y-2 text-xs">
                <div className="flex items-center justify-between">
                  <span className="text-slate-500">Provider:</span>
                  <span className="font-mono text-cyan-300 uppercase font-semibold">{m.provider}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-slate-500">Execution Mode:</span>
                  <span className="font-mono text-emerald-400 capitalize">{m.execution_mode}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-slate-500">Context Limit:</span>
                  <span className="font-mono text-slate-300">{m.context}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-slate-500">Step 20 Action Index:</span>
                  <span className="font-mono text-purple-400 font-bold">{m.actionIndex}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-slate-500">Local Endpoint:</span>
                  <span className="font-mono text-slate-400 text-[10px] truncate max-w-[200px]">{m.endpoint}</span>
                </div>
              </div>

              <div className="border-t border-slate-800/80 pt-3">
                <span className="text-[11px] text-slate-500 font-medium block mb-1.5">Capabilities:</span>
                <div className="flex flex-wrap gap-1.5">
                  {m.capabilities.map((cap, idx) => (
                    <span key={idx} className="px-2 py-0.5 rounded text-[10px] font-mono bg-slate-950/80 text-slate-300 border border-slate-800">
                      {cap}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default ModelsPage;
