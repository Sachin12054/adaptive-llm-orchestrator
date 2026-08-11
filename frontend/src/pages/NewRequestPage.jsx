import React, { useState } from 'react';
import { PlusCircle, Sparkles, Play, Compass, Code, Brain } from 'lucide-react';

const NewRequestPage = ({ onExecutePrompt }) => {
  const [promptText, setPromptText] = useState('');
  const [systemInstruction, setSystemInstruction] = useState('');

  const templates = [
    {
      title: 'Simple Factual Query',
      type: 'SIMPLE',
      icon: Compass,
      color: 'border-emerald-500/30 text-emerald-400 bg-emerald-500/10',
      description: 'Quick factual QA answered by Gemma 3 4B',
      prompt: 'What is the capital of Japan and what is its population?'
    },
    {
      title: 'Medium Coding Task',
      type: 'MEDIUM',
      icon: Code,
      color: 'border-cyan-500/30 text-cyan-400 bg-cyan-500/10',
      description: 'Code synthesis and explanation routed to Qwen Coder 3B',
      prompt: 'Write a Python function to reverse a singly linked list and explain its time complexity.'
    },
    {
      title: 'Complex Multi-Agent Workflow',
      type: 'COMPLEX',
      icon: Brain,
      color: 'border-purple-500/30 text-purple-400 bg-purple-500/10',
      description: 'Decomposed into DAG subtasks across Gemma, DeepSeek, and Qwen',
      prompt: 'Build a Python web scraper that collects product prices, cleans the data, calculates statistics, and explains the results.'
    }
  ];

  const handleSubmit = (e) => {
    e.preventDefault();
    if (promptText.trim()) {
      onExecutePrompt(promptText, systemInstruction);
    }
  };

  return (
    <div className="p-6 space-y-6 max-w-5xl mx-auto">
      <div>
        <h3 className="text-lg font-bold text-slate-100 flex items-center gap-2">
          <PlusCircle className="w-5 h-5 text-cyan-400" />
          <span>New AI Orchestration Request</span>
        </h3>
        <p className="text-xs text-slate-400">
          Execute a prompt through the Adaptive Decision Engine and Multi-LLM Orchestration Pipeline
        </p>
      </div>

      {/* Preset Templates */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {templates.map((tpl, idx) => {
          const Icon = tpl.icon;
          return (
            <button
              key={idx}
              onClick={() => setPromptText(tpl.prompt)}
              className="glass-panel p-4 text-left hover:border-cyan-500/40 transition-all space-y-2 group"
            >
              <div className="flex items-center justify-between">
                <span className={`px-2 py-0.5 rounded text-[10px] font-mono font-semibold border ${tpl.color}`}>
                  {tpl.type}
                </span>
                <Icon className="w-4 h-4 text-slate-500 group-hover:text-cyan-400 transition-colors" />
              </div>
              <h4 className="font-semibold text-xs text-slate-200">{tpl.title}</h4>
              <p className="text-[11px] text-slate-400">{tpl.description}</p>
            </button>
          );
        })}
      </div>

      {/* Prompt Form */}
      <form onSubmit={handleSubmit} className="glass-panel p-6 space-y-4">
        <div className="space-y-1.5">
          <label className="block text-xs font-medium text-slate-300">User Prompt Text *</label>
          <textarea
            value={promptText}
            onChange={(e) => setPromptText(e.target.value)}
            placeholder="Type your prompt here..."
            rows={5}
            className="w-full bg-slate-950/80 border border-slate-800 rounded-xl p-4 text-slate-100 placeholder-slate-500 focus:outline-none focus:border-cyan-500/60 focus:ring-1 focus:ring-cyan-500/60 font-sans text-sm resize-none"
            required
          />
        </div>

        <div className="space-y-1.5">
          <label className="block text-xs font-medium text-slate-300">Optional System Instruction / Persona</label>
          <input
            type="text"
            value={systemInstruction}
            onChange={(e) => setSystemInstruction(e.target.value)}
            placeholder="e.g. You are an expert software engineer and computer scientist."
            className="w-full bg-slate-950/80 border border-slate-800 rounded-lg px-4 py-2.5 text-xs text-slate-100 placeholder-slate-500 focus:outline-none focus:border-cyan-500/60"
          />
        </div>

        <div className="flex justify-end pt-2">
          <button
            type="submit"
            disabled={!promptText.trim()}
            className="px-6 py-2.5 rounded-lg bg-gradient-to-r from-cyan-500 to-indigo-600 hover:from-cyan-400 hover:to-indigo-500 text-white font-medium text-sm flex items-center gap-2 shadow-lg shadow-cyan-500/20 active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
          >
            <Play className="w-4 h-4 fill-current" />
            <span>Execute Prompt</span>
          </button>
        </div>
      </form>
    </div>
  );
};

export default NewRequestPage;
