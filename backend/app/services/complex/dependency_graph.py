import logging
from typing import List, Dict, Set, Any
from app.schemas.complex import SubTask

logger = logging.getLogger("orchestrator")

class DependencyGraphBuilder:
    """
    Constructs a Dependency Directed Acyclic Graph (DAG) for subtasks.
    - Validates subtask IDs for uniqueness.
    - Performs Cycle Detection via DFS.
    - Computes Parallel Execution Levels (topological level grouping).
    """

    def validate_unique_task_ids(self, subtasks: List[SubTask]):
        seen_ids = set()
        for task in subtasks:
            if task.task_id in seen_ids:
                raise ValueError(f"Duplicate task ID detected: '{task.task_id}'")
            seen_ids.add(task.task_id)

    def detect_cycles(self, subtasks: List[SubTask]):
        # Graph mapping: task_id -> list of prerequisite task_ids
        task_ids = {t.task_id for t in subtasks}
        dep_graph = {t.task_id: t.dependencies for t in subtasks}

        # Check for missing dependency IDs
        for task_id, deps in dep_graph.items():
            for dep in deps:
                if dep not in task_ids:
                    raise ValueError(f"Subtask '{task_id}' references non-existent dependency '{dep}'")

        # DFS Cycle Detection
        visited: Dict[str, int] = {t_id: 0 for t_id in task_ids}  # 0: unvisited, 1: visiting, 2: visited

        def dfs(node: str, path: List[str]):
            visited[node] = 1
            path.append(node)
            for dep in dep_graph.get(node, []):
                if visited[dep] == 1:
                    cycle_str = " -> ".join(path[path.index(dep):] + [dep])
                    raise ValueError(f"Cycle detected in subtask dependencies: {cycle_str}")
                elif visited[dep] == 0:
                    dfs(dep, path)
            path.pop()
            visited[node] = 2

        for t_id in task_ids:
            if visited[t_id] == 0:
                dfs(t_id, [])

    def compute_execution_levels(self, subtasks: List[SubTask]) -> List[List[str]]:
        """
        Groups subtasks into topological parallel execution levels.
        Level 0: Subtasks with 0 dependencies.
        Level 1: Subtasks whose dependencies are all satisfied in Level 0.
        ...
        """
        self.validate_unique_task_ids(subtasks)
        self.detect_cycles(subtasks)

        task_deps = {t.task_id: set(t.dependencies) for t in subtasks}
        resolved_tasks: Set[str] = set()
        remaining_tasks = set(task_deps.keys())

        execution_levels: List[List[str]] = []

        while remaining_tasks:
            # Find all tasks whose dependencies are fully resolved
            current_level = sorted([
                t_id for t_id in remaining_tasks
                if task_deps[t_id].issubset(resolved_tasks)
            ])

            if not current_level:
                raise ValueError("Failed to compute execution levels: unresolvable dependencies exist.")

            execution_levels.append(current_level)
            resolved_tasks.update(current_level)
            remaining_tasks.difference_update(current_level)

        return execution_levels
