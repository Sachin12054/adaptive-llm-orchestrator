import React, { useEffect, useState } from 'react';
import { BarChart3, PieChart, TrendingUp, Clock, Award, Layers, DollarSign, Cloud, Server } from 'lucide-react';
import { getCostMetrics } from '../services/api';

const AnalyticsPage = ({ historyItems }) => {
  const [costMetrics, setCostMetrics] = useState(null);
  const [loadingCost, setLoadingCost] = useState(false);

  useEffect(() => {
    let isMounted = true;
    const fetchMetrics = async () => {
      setLoadingCost(true);
      try {
        const data = await getCostMetrics();
        if (isMounted) setCostMetrics(data);
      } catch (err) {
        console.warn('Failed to load cost metrics:', err);
      } finally {
        if (isMounted) setLoadingCost(false);
      }
    };
    fetchMetrics();
    return () => { isMounted = false; };
  }, [historyItems]);

  const totalRequests = historyItems.length || 1;
  const simpleCount = historyItems.filter((i) => i.complexity === 'SIMPLE').length;
  const mediumCount = historyItems.filter((i) => i.complexity === 'MEDIUM').length;
  const complexCount = historyItems.filter((i) => i.complexity === 'COMPLEX').length;

  const gemmaCount = historyItems.filter((i) => i.model === 'gemma-3-4b').length;
  const qwenCount = historyItems.filter((i) => i.model === 'qwen-coder-3b').length;
  const deepseekCount = historyItems.filter((i) => i.model === 'deepseek-r1-7b').length;

  return (
    <div className="p-6 space-y-6 max-w-6xl mx-auto">
      <div>
        <h3 className="text-lg font-bold text-slate-100 flex items-center gap-2">
          <BarChart3 className="w-5 h-5 text-cyan-400" />
          <span>Platform Performance & Routing Analytics</span>
        </h3>
        <p className="text-xs text-slate-400">Distribution analysis of prompt complexity, API costs, and model allocations</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="glass-panel p-4 space-y-1">
          <span className="text-slate-500 text-xs">Total Requests Executed</span>
          <span className="font-mono text-cyan-400 font-bold text-xl block">{historyItems.length}</span>
        </div>
        <div className="glass-panel p-4 space-y-1">
          <span className="text-slate-500 text-xs">Simple Workflow Share</span>
          <span className="font-mono text-emerald-400 font-bold text-xl block">
            {((simpleCount / totalRequests) * 100).toFixed(0)}%
          </span>
        </div>
        <div className="glass-panel p-4 space-y-1">
          <span className="text-slate-500 text-xs">Medium Workflow Share</span>
          <span className="font-mono text-cyan-300 font-bold text-xl block">
            {((mediumCount / totalRequests) * 100).toFixed(0)}%
          </span>
        </div>
        <div className="glass-panel p-4 space-y-1">
          <span className="text-slate-500 text-xs">Complex Workflow Share</span>
          <span className="font-mono text-purple-400 font-bold text-xl block">
            {((complexCount / totalRequests) * 100).toFixed(0)}%
          </span>
        </div>
      </div>

      {/* API Cost & Telemetry Section */}
      <div className="glass-panel p-5 space-y-4 border-emerald-500/20">
        <div className="flex items-center justify-between border-b border-slate-800 pb-3">
          <div className="flex items-center gap-2">
            <DollarSign className="w-5 h-5 text-emerald-400" />
            <h4 className="font-bold text-sm text-slate-200">Authoritative API Cost Telemetry</h4>
          </div>
          <span className="text-xs font-mono text-slate-400">
            {loadingCost ? 'Loading backend metrics...' : `Buffer Records: ${costMetrics?.cloud_requests_count + costMetrics?.local_requests_count || 0}`}
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 text-xs font-mono">
          <div className="bg-slate-950/80 p-3 rounded-lg border border-slate-800">
            <span className="text-slate-400 block text-[11px] font-sans">Total API Cost (USD)</span>
            <span className="text-emerald-400 font-bold text-lg font-mono">
              ${costMetrics?.total_api_cost?.toFixed(6) ?? '0.000000'}
            </span>
          </div>

          <div className="bg-slate-950/80 p-3 rounded-lg border border-slate-800">
            <span className="text-slate-400 block text-[11px] font-sans flex items-center gap-1">
              <Server className="w-3.5 h-3.5 text-emerald-400" /> Local Ollama ($0.00)
            </span>
            <span className="text-slate-200 font-bold text-lg font-mono">
              {costMetrics?.local_requests_count ?? 0} <span className="text-[10px] text-slate-400 font-normal">reqs</span>
            </span>
          </div>

          <div className="bg-slate-950/80 p-3 rounded-lg border border-slate-800">
            <span className="text-slate-400 block text-[11px] font-sans flex items-center gap-1">
              <Cloud className="w-3.5 h-3.5 text-purple-400" /> Cloud API Reqs
            </span>
            <span className="text-purple-300 font-bold text-lg font-mono">
              {costMetrics?.cloud_requests_count ?? 0} <span className="text-[10px] text-slate-400 font-normal">reqs</span>
            </span>
          </div>

          <div className="bg-slate-950/80 p-3 rounded-lg border border-slate-800">
            <span className="text-slate-400 block text-[11px] font-sans">Total Tokens Processed</span>
            <span className="text-cyan-300 font-bold text-lg font-mono">
              {costMetrics?.total_tokens?.toLocaleString() ?? 0}
            </span>
          </div>
        </div>

        {/* Cost Breakdown by Provider */}
        {costMetrics?.cost_by_provider && Object.keys(costMetrics.cost_by_provider).length > 0 && (
          <div className="space-y-2 pt-2">
            <span className="text-xs text-slate-400 font-semibold block">Cost Breakdown by Provider</span>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              {Object.entries(costMetrics.cost_by_provider).map(([prov, data]) => (
                <div key={prov} className="bg-slate-950/60 p-2.5 rounded border border-slate-800/80 text-xs font-mono flex justify-between items-center">
                  <div>
                    <span className="text-slate-200 font-semibold block uppercase">{prov}</span>
                    <span className="text-[10px] text-slate-400">{data.requests} reqs ({data.total_tokens || (data.input_tokens + data.output_tokens)} tok)</span>
                  </div>
                  <span className="text-emerald-400 font-bold">${data.total_cost.toFixed(6)}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Model Frequency Bar Chart Simulation */}
        <div className="glass-panel p-5 space-y-4">
          <h4 className="font-semibold text-xs text-slate-200 uppercase tracking-wider font-mono">
            Model Selection Frequency
          </h4>
          <div className="space-y-3">
            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-emerald-400 font-mono">Gemma 3 4B</span>
                <span className="font-mono text-slate-300">{gemmaCount} requests</span>
              </div>
              <div className="w-full bg-slate-800 rounded-full h-2 overflow-hidden">
                <div className="bg-emerald-400 h-full rounded-full" style={{ width: `${Math.max((gemmaCount / totalRequests) * 100, 15)}%` }}></div>
              </div>
            </div>

            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-cyan-400 font-mono">Qwen Coder 3B</span>
                <span className="font-mono text-slate-300">{qwenCount} requests</span>
              </div>
              <div className="w-full bg-slate-800 rounded-full h-2 overflow-hidden">
                <div className="bg-cyan-400 h-full rounded-full" style={{ width: `${Math.max((qwenCount / totalRequests) * 100, 15)}%` }}></div>
              </div>
            </div>

            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-purple-400 font-mono">DeepSeek R1 7B</span>
                <span className="font-mono text-slate-300">{deepseekCount} requests</span>
              </div>
              <div className="w-full bg-slate-800 rounded-full h-2 overflow-hidden">
                <div className="bg-purple-400 h-full rounded-full" style={{ width: `${Math.max((deepseekCount / totalRequests) * 100, 15)}%` }}></div>
              </div>
            </div>
          </div>
        </div>

        {/* Complexity Breakdown */}
        <div className="glass-panel p-5 space-y-4">
          <h4 className="font-semibold text-xs text-slate-200 uppercase tracking-wider font-mono">
            Complexity Breakdown
          </h4>
          <div className="space-y-3">
            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-emerald-400 font-mono">SIMPLE Requests</span>
                <span className="font-mono text-slate-300">{simpleCount}</span>
              </div>
              <div className="w-full bg-slate-800 rounded-full h-2 overflow-hidden">
                <div className="bg-emerald-400 h-full rounded-full" style={{ width: `${Math.max((simpleCount / totalRequests) * 100, 10)}%` }}></div>
              </div>
            </div>

            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-cyan-400 font-mono">MEDIUM Requests</span>
                <span className="font-mono text-slate-300">{mediumCount}</span>
              </div>
              <div className="w-full bg-slate-800 rounded-full h-2 overflow-hidden">
                <div className="bg-cyan-400 h-full rounded-full" style={{ width: `${Math.max((mediumCount / totalRequests) * 100, 10)}%` }}></div>
              </div>
            </div>

            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-purple-400 font-mono">COMPLEX Tasks</span>
                <span className="font-mono text-slate-300">{complexCount}</span>
              </div>
              <div className="w-full bg-slate-800 rounded-full h-2 overflow-hidden">
                <div className="bg-purple-400 h-full rounded-full" style={{ width: `${Math.max((complexCount / totalRequests) * 100, 10)}%` }}></div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AnalyticsPage;

