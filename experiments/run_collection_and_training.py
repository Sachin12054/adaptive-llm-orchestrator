import os
import sys
import time

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
backend_dir = os.path.join(project_root, "backend")
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from experiments.collect_step20_experiences import main as collect_main
from app.services.experience_buffer import ExperienceBufferService
from app.schemas.rl import RLTrainRequest
from rl.policy_trainer import PolicyTrainer

def run_collection_and_training():
    print("=" * 80)
    print(" EXECUTING 100 REAL OLLAMA STEP 20 EXPERIENCE COLLECTION")
    print("=" * 80)

    # Execute experience collection
    collect_main()

    buffer_service = ExperienceBufferService()
    current_size = buffer_service.get_status().current_size
    print(f"\nFinal Experience Buffer Count: {current_size}")

    if current_size >= 100:
        print("\n" + "=" * 80)
        print(" EXECUTING STEP 21 OFFLINE LINEAR CONTEXTUAL BANDIT TRAINING")
        print("=" * 80)

        trainer = PolicyTrainer()
        train_res = trainer.train_policy(RLTrainRequest(minimum_samples=100, epochs=15, learning_rate=0.01))

        print(f"Training Success   : {train_res.success}")
        print(f"Training Status    : {train_res.status.upper()}")
        print(f"Message            : {train_res.message}")
        print(f"Samples Used       : {train_res.samples_used}")
        print(f"Final MSE Loss     : {train_res.final_mse}")

        if train_res.success:
            print("\nSTEP 21 OFFLINE POLICY TRAINING SUCCESSFULLY COMPLETED!")
    else:
        print(f"\n[NOTICE] Buffer size ({current_size}) < 100. Ensure local Ollama server is running with target models installed.")

if __name__ == "__main__":
    run_collection_and_training()
