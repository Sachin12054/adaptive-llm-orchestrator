import json
import os

def main():
    raw_path = 'validation/results/complex_task_comparison/raw/orchestrated_results.jsonl'
    if not os.path.exists(raw_path):
        print(f"File not found: {raw_path}")
        return

    with open(raw_path, 'r', encoding='utf-8') as f:
        records = [json.loads(l) for l in f if l.strip()]

    print(f"Loaded {len(records)} orchestrated records.")
    r0 = records[0]
    print("Top-level keys in record 0:")
    for k in r0.keys():
        print(f" - {k}: {type(r0[k])}")

    if 'subtasks' in r0:
        print(f"\nSubtasks count in record 0: {len(r0['subtasks'])}")
        st0 = r0['subtasks'][0]
        print("Subtask 0 keys:", list(st0.keys()))
        print("Subtask 0 task_id:", st0.get('task_id'))
        print("Subtask 0 execution_success:", st0.get('execution_success'))
        text = st0.get('generated_text', '')
        print(f"Subtask 0 text length: {len(text)}")

if __name__ == '__main__':
    main()
