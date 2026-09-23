import json
import os

def main():
    raw_dir = 'validation/results/complex_task_comparison/raw'
    orch_file = os.path.join(raw_dir, 'orchestrated_results.jsonl')
    
    with open(orch_file, 'r', encoding='utf-8') as f:
        records = [json.loads(l) for l in f if l.strip()]

    print(f"Total orchestrated records: {len(records)}")
    r0 = records[0]
    print("Keys in record 0:", list(r0.keys()))
    
    # Check if subtasks or subtask_texts are present in records
    has_subtasks = 'subtasks' in r0 or 'subtask_results' in r0 or 'subtask_texts' in r0
    print(f"Are subtask texts explicitly present in raw records? {has_subtasks}")

    # Inspect other raw files in raw_dir
    files = os.listdir(raw_dir)
    print("Files in raw dir:", files)

if __name__ == '__main__':
    main()
