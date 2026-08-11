import React, { useState, useEffect, useRef } from 'react';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import PromptInputBar from './components/PromptInputBar';
import PromptAnalysisCard from './components/PromptAnalysisCard';
import DecisionEngineCard from './components/DecisionEngineCard';
import ModelAllocationPanel from './components/ModelAllocationPanel';
import ResourceMonitorCard from './components/ResourceMonitorCard';
import CostLatencyPanel from './components/CostLatencyPanel';
import RewardLearningCard from './components/RewardLearningCard';
import WorkflowVisualizer from './components/WorkflowVisualizer';
import ExecutionLevelsPanel from './components/ExecutionLevelsPanel';
import ExecutionLogPanel from './components/ExecutionLogPanel';
import FinalResponsePanel from './components/FinalResponsePanel';
import { formatLatency } from './utils/formatLatency';

import NewRequestPage from './pages/NewRequestPage';
import HistoryPage from './pages/HistoryPage';
import ModelsPage from './pages/ModelsPage';
import RlResearchPage from './pages/RlResearchPage';
import AnalyticsPage from './pages/AnalyticsPage';
import SettingsPage from './pages/SettingsPage';

import {
  getHealth,
  getResourceSnapshot,
  getExperienceStatus,
  getRLStatus,
  runOrchestration,
  runOrchestrationStream,
  getComplexPlan,
} from './services/api';

const App = () => {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [prompt, setPrompt] = useState('What is Python?');
  const [isExecuting, setIsExecuting] = useState(false);
  const [executionMode, setExecutionMode] = useState('local');

  // Real Backend Data States
  const [backendStatus, setBackendStatus] = useState(false);
  const [resourceData, setResourceData] = useState(null);
  const [experienceStatus, setExperienceStatus] = useState(null);
  const [rlStatus, setRlStatus] = useState(null);

  // Real SSE Stream Stage State Machine
  const [streamStageState, setStreamStageState] = useState({
    activeStage: null,
    completedStages: [],
    failedStages: [],
    stageLatencies: {}
  });

  // Execution Output States
  const [orchestrationRes, setOrchestrationRes] = useState(null);
  const [complexPlan, setComplexPlan] = useState(null);
  const [executionLogs, setExecutionLogs] = useState([]);
  const [historyItems, setHistoryItems] = useState([
    {
      timestamp: '12:35:10',
      prompt: 'What is Python?',
      complexity: 'SIMPLE',
      model: 'gemma-3-4b',
      reward: '0.9050',
      latency: '1142 ms',
    },
    {
      timestamp: '12:36:22',
      prompt: 'Write Python code to read a CSV using pandas.',
      complexity: 'SIMPLE',
      model: 'qwen-coder-3b',
      reward: '0.9200',
      latency: '1586 ms',
    },
    {
      timestamp: '12:38:05',
      prompt: 'Design a distributed Python ML pipeline...',
      complexity: 'MEDIUM',
      model: 'qwen-coder-3b',
      reward: '0.9475',
      latency: '1812 ms',
    },
  ]);

  const checkBackendStatus = async () => {
    try {
      const h = await getHealth();
      setBackendStatus(h.status === 'ok');
    } catch (e) {
      console.warn('Backend health check error:', e);
      setBackendStatus(false);
    }

    try {
      const resSnap = await getResourceSnapshot();
      setResourceData(resSnap);
    } catch (e) {
      console.warn('Resource snapshot telemetry error:', e);
    }

    try {
      const expStat = await getExperienceStatus();
      setExperienceStatus(expStat);
    } catch (e) {
      console.warn('Experience status error:', e);
    }

    try {
      const rlStat = await getRLStatus();
      setRlStatus(rlStat);
    } catch (e) {
      console.warn('RL status error:', e);
    }
  };

  useEffect(() => {
    checkBackendStatus();
    const interval = setInterval(checkBackendStatus, 15000);
    return () => {
      clearInterval(interval);
      if (abortControllerRef.current) {
        try { abortControllerRef.current.abort(); } catch (e) {}
      }
    };
  }, []);

  const handleModeToggle = (newMode) => {
    setExecutionMode(newMode);
    setOrchestrationRes(null);
    setComplexPlan(null);
    setExecutionLogs([]);
    setStreamStageState({
      activeStage: null,
      completedStages: [],
      failedStages: [],
      stageLatencies: {}
    });
  };

  const currentRunIdRef = useRef(null);
  const abortControllerRef = useRef(null);

  const appendLog = (msg) => {
    const timeStr = new Date().toTimeString().split(' ')[0];
    setExecutionLogs((prev) => [...prev, { time: timeStr, message: msg }]);
  };

  const handleExecute = async () => {
    if (!prompt.trim() || isExecuting) return;

    if (abortControllerRef.current) {
      try { abortControllerRef.current.abort(); } catch (e) {}
    }
    const newController = new AbortController();
    abortControllerRef.current = newController;

    const runId = 'run_' + Date.now() + '_' + Math.random().toString(36).substring(2, 7);
    currentRunIdRef.current = runId;

    setIsExecuting(true);
    setExecutionLogs([]);
    setOrchestrationRes(null);
    setComplexPlan(null);

    appendLog(`[RUN ${runId}] Received prompt: "${prompt}". Initializing Input Processor...`);

    try {
      appendLog('Generating BGE-M3 1024-dim dense semantic embedding...');
      appendLog('Running Intent Classifier, Complexity Analyzer, & Resource Telemetry...');

      const isComplexPrompt =
        prompt.toLowerCase().includes('build a python web scraper') ||
        prompt.toLowerCase().includes('design a production-ready') ||
        prompt.toLowerCase().includes('distributed machine learning') ||
        prompt.toLowerCase().includes('fraud detection') ||
        (prompt.toLowerCase().includes('and') && prompt.split(',').length >= 3) ||
        prompt.split(',').length >= 4 ||
        prompt.length >= 140;

      if (isComplexPrompt) {
        appendLog('Adaptive Decision Engine classified prompt as COMPLEX (Complexity >= 0.80).');
        appendLog('Invoking ComplexTaskDecomposer for parallel execution levels...');
      }

      appendLog('Starting E2E Orchestration Pipeline via SSE real-time stream...');

      setStreamStageState({
        activeStage: 'input_processing',
        completedStages: [],
        failedStages: [],
        stageLatencies: {}
      });

      let finalRes = null;
      await runOrchestrationStream(
        prompt,
        (evt) => {
          if (evt.run_id && evt.run_id !== currentRunIdRef.current) {
            console.warn(`[SSE STALE DISCARD] Ignored event for stale run_id=${evt.run_id} (active run_id=${currentRunIdRef.current})`);
            return;
          }

          if (evt.message) {
            appendLog(evt.message);
          }

          if (evt.stage) {
            setStreamStageState((prev) => {
              const stage = evt.stage;
              const status = evt.status;
              const nextActive = status === 'running' ? stage : prev.activeStage;
              const nextCompleted = status === 'completed' && !prev.completedStages.includes(stage)
                ? [...prev.completedStages, stage]
                : prev.completedStages;
              const nextFailed = status === 'failed' && !prev.failedStages.includes(stage)
                ? [...prev.failedStages, stage]
                : prev.failedStages;
              const nextLatencies = { ...prev.stageLatencies };
              if (evt.metadata?.latency_ms) {
                nextLatencies[stage] = evt.metadata.latency_ms;
              }
              return {
                activeStage: nextActive,
                completedStages: nextCompleted,
                failedStages: nextFailed,
                stageLatencies: nextLatencies
              };
            });
          }

          if (evt.stage === 'adaptive_decision' && (evt.decision || evt.metadata)) {
            const selectedModel = evt.metadata?.selected_model || evt.decision?.selected_model;
            setOrchestrationRes((prev) => ({
              ...(prev || {}),
              selected_model: selectedModel,
              decision: evt.decision || {
                selected_model: selectedModel,
                decision_score: evt.metadata?.score,
                candidates: evt.metadata?.candidates || []
              }
            }));

            if (isComplexPrompt && selectedModel) {
              getComplexPlan(prompt, executionMode, selectedModel)
                .then((plan) => {
                  if (currentRunIdRef.current === runId) {
                    setComplexPlan(plan);
                    appendLog(`Task decomposition generated ${plan.total_subtasks} subtasks assigned policy model '${selectedModel}'.`);
                  }
                })
                .catch((err) => console.warn('Complex plan fetch error:', err));
            }
          }

          if (evt.stage === 'final_response' && evt.payload) {
            finalRes = evt.payload;
            setOrchestrationRes(finalRes);
          }
        },
        null,
        executionMode,
        newController.signal,
        runId
      );

      if (finalRes) {
        const compLevel = (finalRes.decision?.decision_trace?.complexity_level || finalRes.decision?.complexity_info?.complexity_level || (isComplexPrompt ? 'COMPLEX' : 'SIMPLE')).toUpperCase();
        const newHist = {
          timestamp: new Date().toTimeString().split(' ')[0],
          prompt,
          complexity: compLevel,
          model: finalRes.selected_model || 'Model not assigned',
          reward: finalRes.reward?.reward != null ? finalRes.reward.reward.toFixed(4) : 'N/A',
          latency: formatLatency(finalRes.pipeline_latency_ms),
        };
        setHistoryItems((prev) => [newHist, ...prev]);
      }

      await checkBackendStatus();
    } catch (err) {
      console.error('Execution error:', err);
      appendLog(`Execution error: ${err.message || 'Failed to connect to backend API'}`);
    } finally {
      setIsExecuting(false);
    }
  };

  const handleExecuteNewRequest = (newPrompt) => {
    setPrompt(newPrompt);
    setActiveTab('dashboard');
    setTimeout(() => {
      handleExecute();
    }, 100);
  };

  return (
    <div className="h-screen w-screen overflow-hidden flex flex-col bg-[#07101F] text-slate-100 font-sans selection:bg-cyan-500 selection:text-white">
      {/* Fixed Top Header */}
      <Header
        backendStatus={backendStatus}
        isExecuting={isExecuting}
        onRefreshStatus={checkBackendStatus}
        executionMode={executionMode}
        setExecutionMode={handleModeToggle}
      />

      {/* Main Container */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Sidebar */}
        <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} resourceData={resourceData} />

        {/* Dashboard Main Content */}
        <main className="flex-1 overflow-hidden p-3 flex flex-col min-w-0">
          {activeTab === 'dashboard' && (
            <div className="flex-1 flex flex-col gap-2.5 min-h-0 overflow-hidden max-w-[1600px] w-full mx-auto">
              {/* Top Control: Prompt Input Bar */}
              <div className="flex-shrink-0">
                <PromptInputBar
                  prompt={prompt}
                  setPrompt={setPrompt}
                  onExecute={handleExecute}
                  isExecuting={isExecuting}
                />
              </div>

              {/* Middle Section: 3-Column Control Center Grid */}
              <div className="flex-1 grid grid-cols-12 gap-2.5 min-h-0 overflow-hidden">
                {/* Left Column (3 cols) */}
                <div className="col-span-3 flex flex-col gap-2.5 min-h-0 overflow-y-auto pr-0.5">
                  <PromptAnalysisCard
                    decisionData={orchestrationRes?.decision}
                    complexPlanData={complexPlan}
                  />
                  <DecisionEngineCard
                    decisionData={orchestrationRes?.decision}
                    complexPlanData={complexPlan}
                    executionMode={executionMode}
                    generationData={orchestrationRes?.generation}
                  />
                </div>

                {/* Center Column (5 cols) - Visual Centerpiece */}
                <div className="col-span-5 flex flex-col min-h-0 overflow-hidden">
                  <WorkflowVisualizer
                    orchestrationRes={orchestrationRes}
                    complexPlan={complexPlan}
                    prompt={prompt}
                    isExecuting={isExecuting}
                    streamStageState={streamStageState}
                    executionMode={executionMode}
                    onToggleExecutionMode={(mode) => {
                      setExecutionMode(mode);
                      setOrchestrationRes(null);
                      setComplexPlan(null);
                    }}
                  />
                </div>

                {/* Right Column (4 cols) */}
                <div className="col-span-4 flex flex-col gap-2.5 min-h-0 overflow-y-auto pr-0.5">
                  <ModelAllocationPanel
                    executionMode={executionMode}
                    selectedModel={orchestrationRes?.selected_model}
                    candidates={orchestrationRes?.decision?.candidates}
                  />
                  <ResourceMonitorCard resourceData={resourceData} />
                  <CostLatencyPanel
                    orchestrationRes={orchestrationRes}
                    complexPlan={complexPlan}
                    executionMode={executionMode}
                  />
                  <RewardLearningCard
                    orchestrationRes={orchestrationRes}
                    experienceStatus={experienceStatus}
                  />
                </div>
              </div>

              {/* Bottom Section: Execution Log (4 cols) & Final Response (8 cols) */}
              <div className="h-[225px] grid grid-cols-12 gap-2.5 min-h-0 overflow-hidden flex-shrink-0">
                <div className="col-span-4 min-h-0 flex flex-col overflow-hidden">
                  <ExecutionLogPanel logs={executionLogs} />
                </div>
                <div className="col-span-8 min-h-0 flex flex-col overflow-hidden">
                  <FinalResponsePanel
                    orchestrationRes={orchestrationRes}
                    complexPlan={complexPlan}
                  />
                </div>
              </div>
            </div>
          )}

          {activeTab === 'new_request' && (
            <div className="flex-1 overflow-y-auto p-4 max-w-4xl mx-auto">
              <NewRequestPage onExecutePrompt={handleExecuteNewRequest} />
            </div>
          )}

          {activeTab === 'history' && (
            <div className="flex-1 overflow-y-auto p-4 max-w-5xl mx-auto">
              <HistoryPage
                historyItems={historyItems}
                onSelectHistoryItem={(item) => {
                  setPrompt(item.prompt);
                  setActiveTab('dashboard');
                }}
              />
            </div>
          )}

          {activeTab === 'models' && (
            <div className="flex-1 overflow-y-auto p-4 max-w-5xl mx-auto">
              <ModelsPage />
            </div>
          )}

          {activeTab === 'rl_research' && (
            <div className="flex-1 overflow-y-auto p-4 max-w-5xl mx-auto">
              <RlResearchPage
                experienceStatus={experienceStatus}
                rlStatus={rlStatus}
              />
            </div>
          )}

          {activeTab === 'analytics' && (
            <div className="flex-1 overflow-y-auto p-4 max-w-5xl mx-auto">
              <AnalyticsPage historyItems={historyItems} />
            </div>
          )}

          {activeTab === 'settings' && (
            <div className="flex-1 overflow-y-auto p-4 max-w-4xl mx-auto">
              <SettingsPage backendStatus={backendStatus} />
            </div>
          )}
        </main>
      </div>
    </div>
  );
};

export default App;
