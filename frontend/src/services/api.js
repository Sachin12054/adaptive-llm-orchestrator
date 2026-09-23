import axios from 'axios';

const API_BASE_URL = '/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const getHealth = async () => {
  const res = await api.get('/health');
  return res.data;
};

export const getModels = async () => {
  const res = await api.get('/models');
  return res.data;
};

export const getResourceSnapshot = async () => {
  const res = await api.get('/resource/snapshot');
  return res.data;
};

export const getDecisionStatus = async () => {
  const res = await api.get('/decision/status');
  return res.data;
};

export const runDecision = async (prompt) => {
  const res = await api.post('/decision/decide', { text: prompt });
  return res.data;
};

export const runOrchestration = async (prompt, systemInstruction = null, executionMode = 'local') => {
  const res = await api.post('/orchestrate', {
    prompt,
    system_instruction: systemInstruction,
    execution_mode: executionMode
  });
  return res.data;
};

export const runOrchestrationStream = async (prompt, onEvent, systemInstruction = null, executionMode = 'local', signal = null, runId = null) => {
  const response = await fetch('/api/orchestrate/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      prompt,
      system_instruction: systemInstruction,
      execution_mode: executionMode,
      run_id: runId
    }),
    signal
  });

  if (!response.ok) {
    throw new Error(`Server returned status ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const lines = buffer.split('\n\n');
    buffer = lines.pop();

    for (const line of lines) {
      if (line.startsWith('data: ')) {
        try {
          const jsonStr = line.slice(6);
          const data = JSON.parse(jsonStr);
          if (onEvent) onEvent(data);
        } catch (e) {
          console.warn('Failed to parse SSE line:', line);
        }
      }
    }
  }
};

export const getComplexPlan = async (prompt, executionMode = 'local', primarySelectedModel = null) => {
  const res = await api.post('/complex/plan', {
    prompt,
    execution_mode: executionMode,
    primary_selected_model: primarySelectedModel
  });
  return res.data;
};

export const getExperienceStatus = async () => {
  const res = await api.get('/experience/status');
  return res.data;
};

export const getRLStatus = async () => {
  const res = await api.get('/rl/status');
  return res.data;
};

export const getCostMetrics = async () => {
  const res = await api.get('/metrics/cost');
  return res.data;
};

export default api;
