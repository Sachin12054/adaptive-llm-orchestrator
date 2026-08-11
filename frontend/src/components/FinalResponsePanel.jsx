import React, { useState } from 'react';
import { FileText, Copy, Check } from 'lucide-react';

const FinalResponsePanel = ({ orchestrationRes, complexPlan }) => {
  const [activeTab, setActiveTab] = useState('response');
  const [copied, setCopied] = useState(false);

  const isComplex = Boolean(complexPlan?.is_complex || orchestrationRes?.decision?.is_complex);
  const rawText = orchestrationRes?.generation?.generated_text;
  const genText = typeof rawText === 'string' && rawText.trim().length > 0 ? rawText.trim() : null;
  const errMessage = orchestrationRes?.generation?.error_message || orchestrationRes?.error || null;

  let textToDisplay = genText;
  if (!textToDisplay && errMessage) {
    textToDisplay = `[Generation Failure]\n${errMessage}`;
  } else if (!textToDisplay && isComplex) {
    textToDisplay = `[Task Decomposer Plan Generated]\nSubtasks: ${complexPlan?.total_subtasks || 0}\nExecution Levels: ${complexPlan?.execution_levels?.length || 0}\n(Subtasks compiled into DAG schedule, awaiting execution)`;
  }

  const handleCopy = () => {
    if (!textToDisplay) return;
    navigator.clipboard.writeText(textToDisplay);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="glass-panel p-3 flex flex-col h-full overflow-hidden border-cyan-500/20">
      {/* Header with Tabs & Copy Button */}
      <div className="flex items-center justify-between border-b border-slate-800/80 pb-1.5 flex-shrink-0">
        <div className="flex items-center gap-2">
          <FileText className="w-3.5 h-3.5 text-emerald-400" />
          <h4 className="font-bold text-[11px] uppercase tracking-wider text-slate-300">Response Output</h4>
          <div className="flex items-center gap-1 bg-slate-950/80 border border-slate-800 rounded p-0.5 ml-2 text-[10px]">
            <button
              onClick={() => setActiveTab('response')}
              className={`px-2 py-0.5 rounded font-medium transition-colors ${
                activeTab === 'response' ? 'bg-cyan-500/20 text-cyan-300 font-semibold' : 'text-slate-400 hover:text-white'
              }`}
            >
              Response Text
            </button>
            <button
              onClick={() => setActiveTab('json')}
              className={`px-2 py-0.5 rounded font-medium transition-colors ${
                activeTab === 'json' ? 'bg-purple-500/20 text-purple-300 font-semibold' : 'text-slate-400 hover:text-white'
              }`}
            >
              Structured JSON
            </button>
          </div>
        </div>

        {textToDisplay && (
          <button
            onClick={handleCopy}
            className="flex items-center gap-1 text-[10px] px-2 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white transition-colors border border-slate-700"
          >
            {copied ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
            <span>{copied ? 'Copied' : 'Copy'}</span>
          </button>
        )}
      </div>

      {/* Response Area (Internal Scrollable) */}
      <div className="flex-1 overflow-y-auto mt-2 text-xs font-sans leading-relaxed text-slate-200 pr-1 min-h-0">
        {!textToDisplay ? (
          <p className="text-slate-500 italic py-6 text-center text-xs">
            Execute prompt to view output response.
          </p>
        ) : activeTab === 'response' ? (
          <div className="bg-slate-950/80 border border-slate-800/80 rounded-lg p-3 whitespace-pre-wrap font-mono text-[11px] text-slate-200">
            {textToDisplay}
          </div>
        ) : (
          <pre className="bg-slate-950/90 border border-slate-800/80 rounded-lg p-3 font-mono text-[10px] text-cyan-300 overflow-x-auto">
            {JSON.stringify(orchestrationRes || complexPlan, null, 2)}
          </pre>
        )}
      </div>
    </div>
  );
};

export default FinalResponsePanel;
