import React from 'react';
import {
  LayoutDashboard,
  PlusCircle,
  History,
  BrainCircuit,
  Cpu,
  BarChart3,
  Settings
} from 'lucide-react';

const Sidebar = ({ activeTab, setActiveTab, resourceData }) => {
  const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { id: 'new_request', label: 'New Request', icon: PlusCircle },
    { id: 'history', label: 'History', icon: History },
    { id: 'models', label: 'Models & Providers', icon: Cpu },
    { id: 'analytics', label: 'Analytics', icon: BarChart3 },
    { id: 'rl_research', label: 'RL & Research', icon: BrainCircuit },
    { id: 'settings', label: 'Settings', icon: Settings },
  ];

  const cpuPercent = resourceData?.cpu?.utilization_percent != null
    ? `${resourceData.cpu.utilization_percent.toFixed(0)}%`
    : null;

  const ramUsedGb = resourceData?.memory?.used_gb != null
    ? `${resourceData.memory.used_gb.toFixed(1)} GB`
    : null;

  const ramTotalGb = resourceData?.memory?.total_gb != null
    ? `${resourceData.memory.total_gb.toFixed(1)} GB`
    : null;

  const freeVram = resourceData?.gpu?.free_vram_gb != null
    ? `${resourceData.gpu.free_vram_gb.toFixed(1)} GB`
    : null;

  const totalVram = resourceData?.gpu?.total_vram_gb != null
    ? `${resourceData.gpu.total_vram_gb.toFixed(1)} GB`
    : null;

  return (
    <aside className="w-52 bg-slate-900/90 border-r border-slate-800/80 flex flex-col h-full flex-shrink-0 select-none">
      {/* Navigation */}
      <nav className="flex-1 p-2 space-y-1 overflow-y-auto">
        <div className="px-3 py-1.5 text-[10px] font-semibold tracking-wider text-slate-500 uppercase">
          Control Center
        </div>
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeTab === item.id;
          return (
            <button
              key={item.id}
              onClick={() => setActiveTab(item.id)}
              className={`w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-xs font-medium transition-all duration-150 ${
                isActive
                  ? 'bg-gradient-to-r from-purple-600/30 via-indigo-600/20 to-transparent text-purple-300 border-l-2 border-purple-400 shadow-sm font-semibold'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
              }`}
            >
              <Icon className={`w-4 h-4 ${isActive ? 'text-purple-400' : 'text-slate-500'}`} />
              <span>{item.label}</span>
            </button>
          );
        })}
      </nav>

      {/* System Status Panel */}
      <div className="p-3 border-t border-slate-800/80 bg-slate-950/60">
        <div className="flex items-center justify-between text-[11px] font-semibold text-slate-300 mb-2">
          <span className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
            SYSTEM STATUS
          </span>
          <span className="text-[10px] text-emerald-400 font-mono">Operational</span>
        </div>

        <div className="bg-slate-900/90 border border-slate-800/80 rounded-lg p-2.5 space-y-1.5 text-[11px] font-mono">
          <div className="flex items-center justify-between text-slate-400">
            <span>Models Online:</span>
            <span className="text-emerald-400 font-bold">3 / 3</span>
          </div>
          <div className="flex items-center justify-between text-slate-400">
            <span>Active Workflows:</span>
            <span className="text-cyan-400 font-bold">1 Active</span>
          </div>
          {freeVram && totalVram && (
            <div className="flex items-center justify-between text-slate-400">
              <span>GPU Free VRAM:</span>
              <span className="text-emerald-300 font-bold">{freeVram} / {totalVram}</span>
            </div>
          )}
          {cpuPercent && (
            <div className="flex items-center justify-between text-slate-400">
              <span>CPU Utilization:</span>
              <span className="text-cyan-300 font-bold">{cpuPercent}</span>
            </div>
          )}
          {ramUsedGb && ramTotalGb && (
            <div className="flex items-center justify-between text-slate-400">
              <span>System RAM:</span>
              <span className="text-indigo-300 font-bold">{ramUsedGb} / {ramTotalGb}</span>
            </div>
          )}
        </div>
      </div>
    </aside>
  );
};

export default Sidebar;
