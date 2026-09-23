import os
import sys
import json
import time
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

# Ensure backend is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
backend_dir = os.path.join(project_root, "backend")
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.schemas.rl import RLTrainRequest, RLTrainResponse
from app.services.experience_buffer import ExperienceBufferService, ACTION_MAP, REVERSE_ACTION_MAP

logger = logging.getLogger("orchestrator")

class PolicyTrainer:
    def __init__(self, output_path: Optional[str] = None, experience_buffer: Optional[ExperienceBufferService] = None):
        raw_path = output_path or os.path.join("data", "rl", "models", "rl_contextual_bandit_policy.json")
        self.output_path = raw_path if os.path.isabs(raw_path) else os.path.join(project_root, raw_path)
        self.experience_buffer = experience_buffer
        self.action_map = ACTION_MAP
        self.reverse_action_map = REVERSE_ACTION_MAP
        self.num_actions = len(ACTION_MAP)

    def train_policy(self, request: RLTrainRequest) -> RLTrainResponse:
        buffer_service = self.experience_buffer or ExperienceBufferService()
        experiences = buffer_service._buffer

        samples_count = len(experiences)
        logger.info(f"PolicyTrainer loaded {samples_count} experience records for offline training.")

        if samples_count < request.minimum_samples:
            msg = f"Insufficient Step 20 experience samples ({samples_count} available, minimum required is {request.minimum_samples})."
            logger.warning(msg)
            return RLTrainResponse(
                success=False,
                status="insufficient_data",
                message=msg,
                samples_used=samples_count,
                minimum_required=request.minimum_samples,
                final_mse=None,
                policy_path=None
            )

        # Validate state dimension = 12 and extract data
        valid_records = []
        for rec in experiences:
            if len(rec.state) == 12 and rec.action >= 0 and rec.action < self.num_actions:
                valid_records.append(rec)

        if len(valid_records) < request.minimum_samples:
            msg = f"Insufficient valid 12-dim experience samples ({len(valid_records)} valid, minimum required is {request.minimum_samples})."
            logger.warning(msg)
            return RLTrainResponse(
                success=False,
                status="insufficient_data",
                message=msg,
                samples_used=len(valid_records),
                minimum_required=request.minimum_samples,
                final_mse=None,
                policy_path=None
            )

        # Initialize linear model parameters: Weights W (K, 12), Bias b (K,)
        K = self.num_actions
        D = 12
        weights = np.zeros((K, D), dtype=np.float32)
        bias = np.zeros(K, dtype=np.float32)

        # Prepare numpy data arrays
        S = np.array([r.state for r in valid_records], dtype=np.float32)       # Shape (N, 12)
        A = np.array([r.action for r in valid_records], dtype=np.int32)        # Shape (N,)
        R = np.array([r.reward for r in valid_records], dtype=np.float32)       # Shape (N,) exact Step 20 rewards

        # Train linear contextual bandit via Regularized Gradient Descent
        lr = request.learning_rate
        l2 = request.l2_lambda
        epochs = request.epochs
        N = len(valid_records)

        for epoch in range(epochs):
            for i in range(N):
                s_i = S[i]
                a_i = A[i]
                r_i = R[i]

                # Prediction Q_hat(s_i, a_i) = w_{a_i}^T s_i + b_{a_i}
                q_pred = float(np.dot(weights[a_i], s_i) + bias[a_i])
                err = q_pred - r_i

                # Gradients with L2 regularization
                grad_w = err * s_i + l2 * weights[a_i]
                grad_b = err

                # Weight updates
                weights[a_i] -= lr * grad_w
                bias[a_i] -= lr * grad_b

        # Calculate final Mean Squared Error loss
        total_sq_error = 0.0
        for i in range(N):
            q_pred = float(np.dot(weights[A[i]], S[i]) + bias[A[i]])
            total_sq_error += (q_pred - R[i]) ** 2

        final_mse = round(float(total_sq_error / N), 4)

        # Serialize policy file to disk
        policy_data = {
            "policy_name": "rl_contextual_bandit_policy",
            "version": "2.0.0",
            "state_dim": D,
            "action_map": self.action_map,
            "weights": weights.tolist(),
            "bias": bias.tolist(),
            "training_samples_count": N,
            "mean_squared_error": final_mse,
            "timestamp": time.time()
        }

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(policy_data, f, indent=2)

        msg = f"Successfully trained linear contextual bandit policy on {N} experiences (Final MSE: {final_mse:.4f}). Saved to '{self.output_path}'."
        logger.info(msg)

        return RLTrainResponse(
            success=True,
            status="completed",
            message=msg,
            samples_used=N,
            minimum_required=request.minimum_samples,
            final_mse=final_mse,
            policy_path=self.output_path
        )
