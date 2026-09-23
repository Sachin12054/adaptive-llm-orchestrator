import React, { useState, useEffect, useRef, useMemo } from 'react';
import {
  User,
  Brain,
  GitBranch,
  Layers,
  CheckSquare,
  BarChart3,
  CheckCircle2,
  Maximize2,
  Minimize2,
  Minus,
  Plus,
  RotateCcw,
  Sparkles,
  Check,
  Code2,
  Sliders,
  Cpu
} from 'lucide-react';
import { formatLatency } from '../utils/formatLatency';

const STAGE_CONFIG = [
  { key: 'input_processing', title: 'User Prompt', getSub: (p) => p || 'What is Python?', icon: User, colorClass: 'border-slate-700 bg-slate-900/90 text-slate-200 shadow-slate-950/40' },
  { key: 'embedding', title: 'Semantic Understanding', sub: 'BGE-M3 1024D Vector', icon: Brain, colorClass: 'border-emerald-500/40 bg-emerald-950/30 text-emerald-300 shadow-emerald-950/20' },
  { key: 'intent_analysis', title: 'Intent & Complexity Analysis', sub: 'Semantic Classification', icon: Sliders, colorClass: 'border-cyan-500/40 bg-cyan-950/30 text-cyan-300 shadow-cyan-950/20' },
  { key: 'adaptive_decision', title: 'Adaptive Decision Engine', sub: 'BaselineAdaptivePolicy', icon: GitBranch, colorClass: 'border-blue-500/40 bg-blue-950/30 text-blue-300 shadow-blue-950/20' },
  { key: 'local_inference', title: 'Local LLM Inference', getSub: (m) => m || 'gemma-3-4b', icon: Cpu, colorClass: 'border-purple-500/40 bg-purple-950/30 text-purple-300 shadow-purple-950/20' },
  { key: 'response_verifier', title: 'Response Verification', sub: 'Structural & Relevance Checks', icon: CheckSquare, colorClass: 'border-indigo-500/40 bg-indigo-950/30 text-indigo-300 shadow-indigo-950/20' },
  { key: 'reward_signal', title: 'Step 18 Reward Signal', sub: 'Deterministic Signal', icon: BarChart3, colorClass: 'border-purple-500/40 bg-purple-950/30 text-purple-300 shadow-purple-950/20' },
  { key: 'experience_replay', title: 'Step 20 Experience Replay', sub: '12D Transition Recorded', icon: Sparkles, colorClass: 'border-emerald-500/40 bg-emerald-950/30 text-emerald-300 shadow-emerald-950/20' }
];

const TASK_COLORS = [
  { border: 'border-purple-500/50', bg: 'bg-purple-950/40', text: 'text-purple-300', icon: Code2 },
  { border: 'border-amber-500/50', bg: 'bg-amber-950/40', text: 'text-amber-300', icon: Layers },
  { border: 'border-blue-500/50', bg: 'bg-blue-950/40', text: 'text-blue-300', icon: Brain },
  { border: 'border-emerald-500/50', bg: 'bg-emerald-950/40', text: 'text-emerald-300', icon: CheckSquare }
];

const WorkflowVisualizer = ({
  orchestrationRes,
  complexPlan,
  prompt,
  isExecuting,
  streamStageState,
  executionMode = 'local',
  onToggleExecutionMode
}) => {
  const workflowContainerRef = useRef(null);
  const viewportRef = useRef(null);
  const graphRef = useRef(null);

  const [zoom, setZoom] = useState(0.85);
  const [userHasZoomed, setUserHasZoomed] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);

  const isComplexPlan = Boolean(
    complexPlan?.is_complex ||
    (complexPlan?.subtasks && complexPlan.subtasks.length > 0) ||
    orchestrationRes?.decision?.is_complex ||
    orchestrationRes?.decision?.decision_trace?.complexity_level?.toLowerCase() === 'very_high' ||
    orchestrationRes?.decision?.decision_trace?.complexity_level?.toLowerCase() === 'complex'
  );

  const selectedModel = orchestrationRes?.selected_model || null;
  const hasResult = Boolean(orchestrationRes || complexPlan);

  const activeStage = streamStageState?.activeStage;
  const completedStages = streamStageState?.completedStages || [];
  const failedStages = streamStageState?.failedStages || [];
  const stageLatencies = streamStageState?.stageLatencies || {};

  const traceLat = orchestrationRes?.decision?.decision_trace;
  const breakLat = orchestrationRes?.decision?.intent_info?.breakdown_latency_ms;
  const genMs = orchestrationRes?.generation?.latency_ms;
  const verMs = orchestrationRes?.verification?.verification_latency_ms;
  const rwdMs = orchestrationRes?.reward?.latency_ms;

  const isOnlineMode = executionMode === 'online' || executionMode === 'mistral' || executionMode === 'gemini';
  const currentProvider = orchestrationRes?.generation?.provider || (isOnlineMode ? 'Online API' : 'Local Ollama');
  const inferenceTitle = isOnlineMode ? 'Online LLM Inference' : 'Local LLM Inference';
  const inferenceKey = isOnlineMode ? 'online_inference' : 'local_inference';

  const dynamicStageConfig = useMemo(() => {
    const displayModel = selectedModel || 'Model not assigned';
    return [
      { key: 'input_processing', title: 'User Prompt', getSub: (p) => p || 'What is Python?', icon: User, colorClass: 'border-slate-700 bg-slate-900/90 text-slate-200 shadow-slate-950/40' },
      { key: 'embedding', title: 'Semantic Understanding', sub: 'BGE-M3 1024D Vector', icon: Brain, colorClass: 'border-emerald-500/40 bg-emerald-950/30 text-emerald-300 shadow-emerald-950/20' },
      { key: 'intent_analysis', title: 'Intent & Complexity Analysis', sub: 'Semantic Classification', icon: Sliders, colorClass: 'border-cyan-500/40 bg-cyan-950/30 text-cyan-300 shadow-cyan-950/20' },
      { key: 'adaptive_decision', title: 'Adaptive Decision Engine', sub: 'BaselineAdaptivePolicy', icon: GitBranch, colorClass: 'border-blue-500/40 bg-blue-950/30 text-blue-300 shadow-blue-950/20' },
      { key: inferenceKey, title: inferenceTitle, getSub: (m) => `${m || displayModel} (${currentProvider})`, icon: Cpu, colorClass: 'border-purple-500/40 bg-purple-950/30 text-purple-300 shadow-purple-950/20' },
      { key: 'response_verifier', title: 'Response Verification', sub: 'Structural & Relevance Checks', icon: CheckSquare, colorClass: 'border-indigo-500/40 bg-indigo-950/30 text-indigo-300 shadow-indigo-950/20' },
      { key: 'reward_signal', title: 'Step 18 Reward Signal', sub: 'Deterministic Signal', icon: BarChart3, colorClass: 'border-purple-500/40 bg-purple-950/30 text-purple-300 shadow-purple-950/20' },
      { key: 'experience_replay', title: 'Step 20 Experience Replay', sub: '12D Transition Recorded', icon: Sparkles, colorClass: 'border-emerald-500/40 bg-emerald-950/30 text-emerald-300 shadow-emerald-950/20' }
    ];
  }, [inferenceKey, inferenceTitle, selectedModel, currentProvider]);

  const getStageLatency = (stageKey) => {
    if (stageLatencies[stageKey]) return formatLatency(stageLatencies[stageKey]);
    if ((stageKey === 'local_inference' || stageKey === 'online_inference') && (stageLatencies['local_inference'] || stageLatencies['online_inference'] || genMs)) {
      return formatLatency(stageLatencies['local_inference'] || stageLatencies['online_inference'] || genMs);
    }
    if (stageKey === 'embedding' && breakLat?.embedding_ms) return formatLatency(breakLat.embedding_ms);
    if (stageKey === 'intent_analysis' && breakLat?.classification_ms) return formatLatency(breakLat.classification_ms);
    if (stageKey === 'adaptive_decision' && traceLat?.decision_latency_ms) return formatLatency(traceLat.decision_latency_ms);
    if (stageKey === 'response_verifier' && verMs) return formatLatency(verMs);
    if (stageKey === 'reward_signal' && rwdMs) return formatLatency(rwdMs);
    if (stageKey === 'experience_replay' && hasResult) return formatLatency(0.4);
    if (stageKey === 'input_processing' && hasResult) return formatLatency(0.1);
    return null;
  };

  const getNodeState = (stageKey, stageIdx) => {
    const isStage5 = stageKey === 'local_inference' || stageKey === 'online_inference';
    const isFailed = failedStages.includes(stageKey) || (isStage5 && (failedStages.includes('local_inference') || failedStages.includes('online_inference')));
    const isDone = completedStages.includes(stageKey) || (isStage5 && (completedStages.includes('local_inference') || completedStages.includes('online_inference')));
    const isRunning = isExecuting && (activeStage === stageKey || (isStage5 && (activeStage === 'local_inference' || activeStage === 'online_inference')));

    if (isFailed) {
      return { status: 'FAILED', label: 'FAILED', badgeClass: 'bg-red-500/20 text-red-300 border-red-500/40', cardStyle: 'bg-red-950/40 border-red-500/60 text-red-200' };
    }

    if (isDone) {
      return { status: 'COMPLETED', label: 'COMPLETED', badgeClass: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40', isDone: true };
    }

    if (isRunning) {
      return { status: 'RUNNING', label: 'RUNNING', badgeClass: 'bg-purple-500/20 text-purple-300 border-purple-500/40 glow-active', cardStyle: 'bg-purple-950/60 border-purple-500/80 text-purple-200 ring-1 ring-purple-500/50', isRunning: true };
    }

    if (isExecuting) {
      const firstUncompletedIdx = dynamicStageConfig.findIndex(s => {
        const k = s.key;
        const isDoneK = completedStages.includes(k) || (k.includes('inference') && (completedStages.includes('local_inference') || completedStages.includes('online_inference')));
        return !isDoneK;
      });
      if (stageIdx <= firstUncompletedIdx + 1) {
        return { status: 'WAITING', label: 'WAITING', badgeClass: 'bg-slate-800 text-slate-400 border-slate-700', cardStyle: 'bg-slate-900/50 border-slate-800/80 text-slate-400' };
      }
      return { status: 'HIDDEN', label: 'HIDDEN', cardStyle: 'opacity-0 pointer-events-none' };
    }

    if (hasResult) {
      return { status: 'COMPLETED', label: 'COMPLETED', badgeClass: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40', isDone: true };
    }

    return { status: 'WAITING', label: 'WAITING', badgeClass: 'bg-slate-800 text-slate-400 border-slate-700', cardStyle: 'bg-slate-900/50 border-slate-800/80 text-slate-400' };
  };

  // Robust Subtask Extractor grounded in backend ComplexTaskPlan payload
  const subtasks = useMemo(() => {
    if (!complexPlan) return [];

    if (Array.isArray(complexPlan.subtasks) && complexPlan.subtasks.length > 0) {
      return complexPlan.subtasks.map((st, idx) => {
        const theme = TASK_COLORS[idx % TASK_COLORS.length];
        return {
          id: st.task_id || `TASK-${idx + 1}`,
          title: st.task_id || `Task ${idx + 1}`,
          sub: st.description || st.category || 'Subtask Objective',
          model: st.assigned_model || 'Model not recorded',
          actualModel: st.execution_success ? (st.actual_model || st.assigned_model || null) : null,
          actualProvider: st.execution_success ? (st.actual_provider || st.provider || null) : null,
          category: st.category || 'general',
          theme
        };
      });
    }

    if (Array.isArray(complexPlan.execution_levels) && complexPlan.execution_levels.length > 0) {
      const allTasks = [];
      complexPlan.execution_levels.forEach((lvl) => {
        if (Array.isArray(lvl.tasks)) {
          lvl.tasks.forEach((t) => allTasks.push(t));
        }
      });
      if (allTasks.length > 0) {
        return allTasks.map((st, idx) => {
          const theme = TASK_COLORS[idx % TASK_COLORS.length];
          return {
            id: st.task_id || `TASK-${idx + 1}`,
            title: st.task_id || `Task ${idx + 1}`,
            sub: st.description || st.category || 'Subtask Objective',
            model: st.assigned_model || 'Model not recorded',
            actualModel: st.execution_success ? (st.actual_model || st.assigned_model || null) : null,
            actualProvider: st.execution_success ? (st.actual_provider || st.provider || null) : null,
            category: st.category || 'general',
            theme
          };
        });
      }
    }

    return [];
  }, [complexPlan]);

  // Fullscreen Event Listener
  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(document.fullscreenElement === workflowContainerRef.current);
    };
    document.addEventListener('fullscreenchange', handleFullscreenChange);
    return () => {
      document.removeEventListener('fullscreenchange', handleFullscreenChange);
    };
  }, []);

  const toggleFullscreen = async () => {
    try {
      if (!document.fullscreenElement) {
        if (workflowContainerRef.current?.requestFullscreen) {
          await workflowContainerRef.current.requestFullscreen();
        }
      } else {
        if (document.exitFullscreen) {
          await document.exitFullscreen();
        }
      }
    } catch (err) {
      console.error('Fullscreen error:', err);
    }
  };

  // Dynamic Layout & Edge Generator (ONE -> MANY -> ONE DAG)
  const layout = useMemo(() => {
    const NODE_W = 340;
    const NODE_H = 68;
    const V_GAP = 24;

    if (!isComplexPlan || subtasks.length === 0) {
      // Standard Vertical Flow (Simple / Factual / Coding Pipeline)
      const visibleStages = dynamicStageConfig.filter((stg, idx) => {
        const nState = getNodeState(stg.key, idx);
        return nState.status !== 'HIDDEN';
      });

      const canvasW = 560;
      const nodes = visibleStages.map((stg, idx) => {
        const x = (canvasW - NODE_W) / 2;
        const y = idx * (NODE_H + V_GAP);
        return { ...stg, x, y, width: NODE_W, height: NODE_H, idx };
      });

      const canvasH = Math.max(nodes.length * (NODE_H + V_GAP) - V_GAP, 120);

      const connections = [];
      for (let i = 0; i < nodes.length - 1; i++) {
        connections.push({
          fromX: nodes[i].x + NODE_W / 2,
          fromY: nodes[i].y + NODE_H,
          toX: nodes[i + 1].x + NODE_W / 2,
          toY: nodes[i + 1].y,
          key: `${nodes[i].key}->${nodes[i + 1].key}`,
          isDashed: false
        });
      }

      return { canvasW, canvasH, nodes, connections };
    } else {
      // Complex ONE -> MANY -> ONE DAG Flow (Subtasks Active & Rendered)
      const BRANCH_W = 210;
      const H_GAP = 20;
      const numBranches = Math.max(subtasks.length, 1);
      const branchTotalW = numBranches * BRANCH_W + (numBranches - 1) * H_GAP;
      const canvasW = Math.max(880, branchTotalW + 80);

      const topStages = [
        { key: 'input_processing', title: 'User Prompt', getSub: (p) => p || 'What is Python?', icon: User, colorClass: 'border-slate-700 bg-slate-900/90 text-slate-200' },
        { key: 'embedding', title: 'Semantic Understanding', sub: 'BGE-M3 1024D Vector', icon: Brain, colorClass: 'border-emerald-500/40 bg-emerald-950/30 text-emerald-300' },
        { key: 'task_decomposer', title: 'Task Decomposition', sub: `Decomposed into ${subtasks.length} Subtasks`, icon: GitBranch, colorClass: 'border-blue-500/40 bg-blue-950/30 text-blue-300' }
      ];

      const bottomStages = [
        { key: 'response_verifier', title: 'Response Verification', sub: 'Structural & Relevance Checks', icon: CheckSquare, colorClass: 'border-blue-500/40 bg-blue-950/30 text-blue-300' },
        { key: 'reward_signal', title: 'Response Aggregation', sub: 'Step 18 Reward Signal', icon: BarChart3, colorClass: 'border-purple-500/40 bg-purple-950/30 text-purple-300' },
        { key: 'experience_replay', title: 'Final Response', sub: 'Step 20 Experience Recorded', icon: Sparkles, colorClass: 'border-emerald-500/40 bg-emerald-950/30 text-emerald-300' }
      ];

      const nodes = [];
      let currentY = 0;

      // Top Nodes (ONE)
      topStages.forEach((stg, idx) => {
        nodes.push({
          ...stg,
          x: (canvasW - NODE_W) / 2,
          y: currentY,
          width: NODE_W,
          height: NODE_H,
          idx,
          type: 'top'
        });
        currentY += NODE_H + V_GAP;
      });

      // Branch Subtask Nodes (MANY)
      const branchY = currentY;
      const branchStartX = (canvasW - branchTotalW) / 2;
      const branchNodes = subtasks.map((st, bIdx) => {
        const x = branchStartX + bIdx * (BRANCH_W + H_GAP);
        return {
          id: st.id,
          key: st.id,
          title: st.title,
          sub: st.sub,
          model: st.model,
          actualModel: st.actualModel,
          actualProvider: st.actualProvider,
          isSubtask: true,
          theme: st.theme,
          x,
          y: branchY,
          width: BRANCH_W,
          height: NODE_H + 10,
          type: 'branch'
        };
      });

      nodes.push(...branchNodes);
      currentY = branchY + NODE_H + 10 + V_GAP;

      // Bottom Nodes (ONE)
      bottomStages.forEach((stg, bIdx) => {
        const globalIdx = topStages.length + bIdx;
        nodes.push({
          ...stg,
          x: (canvasW - NODE_W) / 2,
          y: currentY,
          width: NODE_W,
          height: NODE_H,
          idx: globalIdx,
          type: 'bottom'
        });
        currentY += NODE_H + V_GAP;
      });

      const canvasH = currentY - V_GAP;

      // DAG Connectors
      const connections = [];

      // Top Straight Connections
      for (let i = 0; i < topStages.length - 1; i++) {
        connections.push({
          fromX: nodes[i].x + NODE_W / 2,
          fromY: nodes[i].y + NODE_H,
          toX: nodes[i + 1].x + NODE_W / 2,
          toY: nodes[i + 1].y,
          key: `top_${i}`,
          isDashed: false
        });
      }

      // Decomposer to Subtask Branches (Curved Dashed Branching Lines)
      const decomposer = nodes[topStages.length - 1];
      if (decomposer && typeof decomposer.x === 'number') {
        branchNodes.forEach((bNode) => {
          if (bNode && typeof bNode.x === 'number') {
            connections.push({
              fromX: decomposer.x + NODE_W / 2,
              fromY: decomposer.y + NODE_H,
              toX: bNode.x + BRANCH_W / 2,
              toY: bNode.y,
              key: `decomp->${bNode.id}`,
              isDashed: true
            });
          }
        });
      }

      // Subtask Branches to Response Verification (Curved Dashed Convergence Lines)
      const verifier = nodes.find(n => n.key === 'response_verifier');
      if (verifier && typeof verifier.x === 'number') {
        branchNodes.forEach((bNode) => {
          if (bNode && typeof bNode.x === 'number') {
            connections.push({
              fromX: bNode.x + BRANCH_W / 2,
              fromY: bNode.y + (bNode.height || 80),
              toX: verifier.x + NODE_W / 2,
              toY: verifier.y,
              key: `${bNode.id}->verifier`,
              isDashed: true
            });
          }
        });
      }

      // Bottom Straight Connections (NO Trailing Edge on Final Node)
      const bottomNodeList = nodes.filter(n => n.type === 'bottom');
      for (let i = 0; i < bottomNodeList.length - 1; i++) {
        if (bottomNodeList[i] && bottomNodeList[i + 1] && typeof bottomNodeList[i].x === 'number' && typeof bottomNodeList[i + 1].x === 'number') {
          connections.push({
            fromX: bottomNodeList[i].x + NODE_W / 2,
            fromY: bottomNodeList[i].y + NODE_H,
            toX: bottomNodeList[i + 1].x + NODE_W / 2,
            toY: bottomNodeList[i + 1].y,
            key: `bottom_${i}`,
            isDashed: false
          });
        }
      }

      return { canvasW, canvasH, nodes, connections };
    }
  }, [isComplexPlan, complexPlan, subtasks, streamStageState, isExecuting]);

  const calculateDynamicFit = () => {
    if (!viewportRef.current) return 0.85;

    const vpRect = viewportRef.current.getBoundingClientRect();
    const availableWidth = vpRect.width - 48;
    const availableHeight = vpRect.height - 48;

    const unscaledWidth = layout.canvasW;
    const unscaledHeight = layout.canvasH;

    if (availableWidth <= 0 || availableHeight <= 0) return 0.85;

    const scaleX = availableWidth / unscaledWidth;
    const scaleY = availableHeight / unscaledHeight;
    const fitScale = Math.min(scaleX, scaleY);

    const boundedScale = Math.min(Math.max(fitScale, 0.55), 1.0);
    return parseFloat(boundedScale.toFixed(2));
  };

  const handleFitView = () => {
    const fitScale = calculateDynamicFit();
    setZoom(fitScale);
    setUserHasZoomed(false);
    if (viewportRef.current) {
      viewportRef.current.scrollTop = 0;
      viewportRef.current.scrollLeft = 0;
    }
  };

  const handleZoomIn = () => {
    setZoom((prev) => Math.min(parseFloat((prev + 0.1).toFixed(2)), 1.25));
    setUserHasZoomed(true);
  };

  const handleZoomOut = () => {
    setZoom((prev) => Math.max(parseFloat((prev - 0.1).toFixed(2)), 0.45));
    setUserHasZoomed(true);
  };

  const handleResetZoom = () => {
    setZoom(1.0);
    setUserHasZoomed(true);
    if (viewportRef.current) {
      viewportRef.current.scrollTop = 0;
      viewportRef.current.scrollLeft = 0;
    }
  };

  useEffect(() => {
    if (viewportRef.current) {
      viewportRef.current.scrollTop = 0;
      viewportRef.current.scrollLeft = 0;
    }
    if (!userHasZoomed) {
      const fitScale = calculateDynamicFit();
      setZoom(fitScale);
    }
  }, [prompt, isExecuting, hasResult, layout, isFullscreen]);

  const isPipelineComplete = hasResult && (completedStages.length >= 7 || Boolean(complexPlan)) && !isExecuting;

  return (
    <div
      ref={workflowContainerRef}
      className={`glass-panel flex flex-col overflow-hidden border-slate-800/90 bg-slate-950/95 shadow-2xl select-none transition-all ${
        isFullscreen ? 'w-screen h-screen fixed inset-0 z-50 rounded-none' : 'h-full min-h-0'
      }`}
    >
      {/* Header Toolbar (56px height) */}
      <div className="flex items-center justify-between border-b border-slate-800/80 px-4 py-3 flex-shrink-0 z-20 bg-slate-950/95">
        <h4 className="font-bold text-xs uppercase tracking-wider text-slate-200 flex items-center gap-2">
          WORKFLOW VISUALIZATION
          {isPipelineComplete && (
            <span className="ml-2 px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 text-[9px] font-mono font-bold uppercase flex items-center gap-1">
              <Check className="w-3 h-3 text-emerald-400" /> Complete
            </span>
          )}
        </h4>

        {/* Toolbar Controls */}
        <div className="flex items-center gap-1.5 bg-slate-900 border border-slate-800 rounded-lg p-1 text-[11px]">
          {/* Execution Provider Switch: LOCAL | ONLINE */}
          <div className="flex items-center bg-slate-950 rounded border border-slate-800 p-0.5 mr-1 font-mono text-[10px]">
            <button
              onClick={() => onToggleExecutionMode && onToggleExecutionMode('local')}
              className={`px-3 py-0.5 rounded transition-all font-bold ${
                executionMode === 'local'
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
              title="Automatic Local Ollama Inference (Gemma 3 4B / Qwen Coder 3B / DeepSeek R1 7B)"
            >
              LOCAL
            </button>
            <button
              onClick={() => onToggleExecutionMode && onToggleExecutionMode('online')}
              className={`px-3 py-0.5 rounded transition-all font-bold ${
                executionMode === 'online' || executionMode === 'mistral' || executionMode === 'gemini'
                  ? 'bg-purple-500/20 text-purple-300 border border-purple-500/40 shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
              title="Automatic Online Cloud API Inference (Gemini / Mistral / Groq / OpenRouter)"
            >
              ONLINE
            </button>
          </div>

          <div className="w-[1px] h-3.5 bg-slate-800 mx-0.5"></div>

          <button
            onClick={toggleFullscreen}
            className="p-1 hover:bg-slate-800 rounded text-slate-400 hover:text-white transition-colors"
            title={isFullscreen ? 'Exit Fullscreen (ESC)' : 'Fullscreen Mode'}
          >
            {isFullscreen ? <Minimize2 className="w-3.5 h-3.5 text-cyan-400" /> : <Maximize2 className="w-3.5 h-3.5 text-cyan-400" />}
          </button>
          <div className="w-[1px] h-3.5 bg-slate-800 mx-0.5"></div>
          <button onClick={handleZoomOut} className="p-1 hover:bg-slate-800 rounded text-slate-400 hover:text-white transition-colors" title="Zoom out">
            <Minus className="w-3.5 h-3.5 text-slate-300" />
          </button>
          <span className="font-mono text-cyan-300 font-bold px-1.5">{Math.round(zoom * 100)}%</span>
          <button onClick={handleZoomIn} className="p-1 hover:bg-slate-800 rounded text-slate-400 hover:text-white transition-colors" title="Zoom in">
            <Plus className="w-3.5 h-3.5 text-slate-300" />
          </button>
          <button onClick={handleResetZoom} className="p-1 hover:bg-slate-800 rounded text-slate-400 hover:text-white transition-colors" title="Reset zoom">
            <RotateCcw className="w-3.5 h-3.5 text-slate-400" />
          </button>
        </div>
      </div>

      {/* Viewport Container */}
      <div
        ref={viewportRef}
        className="flex-1 bg-[#0b0f19] border-t border-slate-800/40 overflow-auto relative min-h-0"
      >
        {/* Centering Canvas Wrapper */}
        <div className="min-w-full min-h-full flex items-center justify-center p-6 box-border">
          {/* Scaled Workflow Canvas */}
          <div
            ref={graphRef}
            className="transition-transform duration-150 relative box-border"
            style={{
              width: `${layout.canvasW}px`,
              height: `${layout.canvasH}px`,
              transform: `scale(${zoom})`,
              transformOrigin: 'top center'
            }}
          >
            {/* SINGLE SVG CONNECTOR OVERLAY LAYER */}
            <svg
              className="absolute inset-0 w-full h-full pointer-events-none z-0"
              style={{ overflow: 'visible' }}
            >
              <defs>
                <marker
                  id="workflow-arrow"
                  viewBox="0 0 10 10"
                  refX="6"
                  refY="5"
                  markerWidth="6"
                  markerHeight="6"
                  orient="auto-start-reverse"
                >
                  <path d="M 0 0 L 10 5 L 0 10 z" fill="#38bdf8" />
                </marker>
              </defs>

              {layout.connections.map((conn) => {
                const { fromX, fromY, toX, toY, key, isDashed } = conn;
                const midY = (fromY + toY) / 2;
                const pathD = isDashed
                  ? `M ${fromX} ${fromY} C ${fromX} ${midY}, ${toX} ${midY}, ${toX} ${toY}`
                  : `M ${fromX} ${fromY} L ${toX} ${toY}`;

                return (
                  <path
                    key={key}
                    d={pathD}
                    fill="none"
                    stroke="#38bdf8"
                    strokeWidth="1.75"
                    strokeDasharray={isDashed ? '5 4' : 'none'}
                    opacity="0.75"
                    markerEnd="url(#workflow-arrow)"
                  />
                );
              })}
            </svg>

            {/* Nodes Layer */}
            {layout.nodes.map((node) => {
              if (node.isSubtask) {
                // Subtask Branch Pill (Complex DAG)
                const IconComponent = node.theme?.icon || Layers;
                return (
                  <div
                    key={node.id}
                    className={`absolute box-border p-2.5 rounded-xl border ${node.theme?.border} ${node.theme?.bg} ${node.theme?.text} shadow-lg flex flex-col justify-between z-10 font-mono`}
                    style={{
                      left: `${node.x}px`,
                      top: `${node.y}px`,
                      width: `${node.width}px`,
                      height: `${node.height}px`
                    }}
                  >
                    <div className="flex items-center justify-between gap-1.5 min-w-0">
                      <span className="font-bold text-[11px] text-slate-200 truncate flex-1 min-w-0 flex items-center gap-1.5">
                        <IconComponent className="w-3.5 h-3.5 text-cyan-400 flex-shrink-0" />
                        <span className="truncate">{node.title}</span>
                      </span>
                      <span className="px-1.5 py-0.5 rounded bg-slate-950/80 text-slate-400 border border-slate-800 text-[8px] flex-shrink-0">
                        SKIPPED (PLAN ONLY)
                      </span>
                    </div>
                    <div className="text-[9px] text-slate-300 truncate" title={node.sub}>
                      {node.sub}
                    </div>
                    <div className="flex justify-between items-center text-[8px] text-slate-400 border-t border-slate-800/80 pt-1">
                      <span>Executed: <span className="text-cyan-400 font-bold">{node.actualModel || node.model}</span></span>
                      {node.actualProvider && <span className="text-emerald-300 truncate max-w-[90px]">{node.actualProvider}</span>}
                      {node.level && <span>Level {node.level}</span>}
                    </div>
                  </div>
                );
              }

              // Pipeline Node Pill
              const nState = getNodeState(node.key, node.idx || 0);
              const IconComp = node.icon || Brain;
              const latStr = getStageLatency(node.key);
              const customStyle = nState.cardStyle || node.colorClass || 'border-slate-800 bg-slate-900/80 text-slate-300';

              return (
                <div
                  key={node.key}
                  className={`absolute box-border px-3.5 py-2.5 rounded-xl border ${customStyle} shadow-lg flex items-center justify-between gap-3 transition-all z-10`}
                  style={{
                    left: `${node.x}px`,
                    top: `${node.y}px`,
                    width: `${node.width}px`,
                    height: `${node.height}px`
                  }}
                >
                  <div className="flex items-center gap-3 min-w-0 flex-1">
                    <div className="w-8 h-8 rounded-lg bg-slate-950/60 border border-slate-800/80 flex items-center justify-center flex-shrink-0">
                      <IconComp className="w-4 h-4 text-cyan-400" />
                    </div>
                    <div className="flex flex-col min-w-0 flex-1 font-mono">
                      <div className="text-[11px] font-bold tracking-wide truncate text-slate-200">
                        {node.key === 'local_inference'
                          ? ((executionMode === 'online' || (orchestrationRes?.generation?.provider && orchestrationRes.generation.provider !== 'Local Ollama')) ? 'Online LLM Inference' : 'Local LLM Inference')
                          : node.title}
                      </div>
                      <div className="text-[9px] text-slate-400 truncate">
                        {node.key === 'local_inference'
                          ? (orchestrationRes?.generation?.model_id ? `${orchestrationRes.generation.model_id} (${orchestrationRes.generation.provider || 'Automatic'})` : (executionMode === 'online' ? 'Automatic Online Selection' : (node.getSub ? node.getSub(selectedModel) : node.sub)))
                          : (node.getSub ? node.getSub(selectedModel) : node.sub)}
                      </div>
                    </div>
                  </div>

                  {nState.isDone && <CheckCircle2 className="w-4 h-4 text-emerald-400 flex-shrink-0" />}
                  {nState.isRunning && <span className="w-2 h-2 rounded-full bg-purple-400 animate-ping flex-shrink-0" />}
                  {latStr && hasResult && (
                    <span className="text-[9px] text-cyan-300 font-mono bg-slate-950/80 px-2 py-0.5 rounded border border-slate-800 flex-shrink-0">
                      {latStr}
                    </span>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
};

export default WorkflowVisualizer;
