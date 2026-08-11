import React from 'react';
import { AlertTriangle, RefreshCw } from 'lucide-react';

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null, errorInfo: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error('React ErrorBoundary Caught Error:', error, errorInfo);
    this.setState({ errorInfo });
  }

  handleReload = () => {
    window.location.reload();
  };

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen w-screen bg-[#07101F] text-slate-100 flex flex-col items-center justify-center p-6 font-sans">
          <div className="max-w-xl w-full bg-slate-900/90 border border-red-500/40 rounded-xl p-6 shadow-2xl shadow-red-500/10 space-y-4">
            <div className="flex items-center gap-3 border-b border-slate-800 pb-3">
              <div className="w-10 h-10 rounded-lg bg-red-500/15 border border-red-500/30 flex items-center justify-center text-red-400">
                <AlertTriangle className="w-5 h-5" />
              </div>
              <div>
                <h2 className="text-base font-bold text-slate-100">Application Rendering Error</h2>
                <p className="text-xs text-slate-400">A frontend runtime exception was safely caught by ErrorBoundary.</p>
              </div>
            </div>

            <div className="bg-slate-950/80 p-3 rounded-lg border border-slate-800 font-mono text-xs text-red-300 whitespace-pre-wrap max-h-40 overflow-y-auto">
              {this.state.error?.toString() || 'Unknown React Exception'}
            </div>

            {this.state.errorInfo && (
              <details className="text-[11px] font-mono text-slate-400">
                <summary className="cursor-pointer hover:text-slate-200 transition-colors">View Component Stack Trace</summary>
                <pre className="mt-2 bg-slate-950/90 p-3 rounded border border-slate-800/80 overflow-x-auto text-[10px] text-slate-400">
                  {this.state.errorInfo.componentStack}
                </pre>
              </details>
            )}

            <div className="flex items-center justify-end pt-2">
              <button
                onClick={this.handleReload}
                className="flex items-center gap-2 px-4 py-2 rounded-lg bg-red-600 hover:bg-red-500 text-white font-semibold text-xs transition-all shadow-lg shadow-red-600/20 active:scale-95"
              >
                <RefreshCw className="w-4 h-4" />
                <span>Reload Application</span>
              </button>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
