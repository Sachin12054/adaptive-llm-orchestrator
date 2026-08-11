export function formatLatency(ms) {
  if (ms === null || ms === undefined || !Number.isFinite(Number(ms))) {
    return "N/A";
  }

  const totalSeconds = Number(ms) / 1000;

  if (totalSeconds < 60) {
    return `${totalSeconds.toFixed(2)} sec`;
  }

  const minutes = Math.floor(totalSeconds / 60);
  const seconds = Math.round(totalSeconds % 60);

  return `${minutes} min ${String(seconds).padStart(2, "0")} sec`;
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = { formatLatency };
}
