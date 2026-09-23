from app.services.resource_analyzer import ResourceAnalyzer

class ResourceMetricsEvaluator:
    """
    Hardware Telemetry & Resource Efficiency Evaluator.
    Captures CPU %, RAM %, VRAM %, and local vs cloud concurrency bounds.
    """

    def __init__(self, resource_analyzer: ResourceAnalyzer = None):
        self.resource_analyzer = resource_analyzer or ResourceAnalyzer()

    def capture_snapshot(self) -> dict:
        state = self.resource_analyzer.get_system_state()
        return {
            "cpu_utilization_pct": state.cpu_utilization,
            "ram_utilization_pct": state.ram_utilization,
            "gpu_utilization_pct": state.gpu_utilization,
            "vram_utilization_pct": state.vram_utilization,
            "local_mode_concurrency_bound": state.recommended_local_concurrency
        }
