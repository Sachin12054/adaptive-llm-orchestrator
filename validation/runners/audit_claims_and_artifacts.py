import os
import hashlib
import json
import re

def sha256_file(filepath):
    if not os.path.exists(filepath):
        return None
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def check_file_integrity():
    raw_files = {
        "RUN_1_FULL_PROVIDER": "validation/results/raw",
        "RUN_2_LOCAL_ONLY": "validation/results/local_only/raw",
        "RUN_3_COMPLEX_TASK": "validation/results/complex_task_comparison/raw",
        "RUN_4_SYNTHESIS": "validation/results/complex_task_synthesis_optimization/raw"
    }

    integrity_report = {}
    for run_name, rdir in raw_files.items():
        if os.path.exists(rdir):
            files = os.listdir(rdir)
            integrity_report[run_name] = {}
            for fname in files:
                fpath = os.path.join(rdir, fname)
                if os.path.isfile(fpath):
                    fstat = os.stat(fpath)
                    integrity_report[run_name][fname] = {
                        "size_bytes": fstat.st_size,
                        "mtime": fstat.st_mtime,
                        "sha256": sha256_file(fpath)
                    }
    return integrity_report

def search_claims():
    reports_dir = 'validation/reports'
    report_files = sorted([f for f in os.listdir(reports_dir) if f.endswith('.md')])

    target_terms = [
        "proven", "guaranteed", "true speedup", "100% improvement",
        "superior", "significant", "equivalent", "state of the art",
        "best", "eliminated"
    ]

    claims_found = []
    pattern = re.compile(r'\b(' + '|'.join(re.escape(t) for t in target_terms) + r')\b', re.IGNORECASE)

    for rfile in report_files:
        rpath = os.path.join(reports_dir, rfile)
        with open(rpath, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for idx, line in enumerate(lines, 1):
            matches = pattern.findall(line)
            if matches:
                claims_found.append({
                    "report_file": rfile,
                    "line_number": idx,
                    "terms": list(set(m.lower() for m in matches)),
                    "line_text": line.strip()
                })

    return claims_found

def main():
    integrity = check_file_integrity()
    claims = search_claims()

    out_file = 'validation/results/audit_integrity_data.json'
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump({"raw_integrity": integrity, "claims_count": len(claims), "claims": claims}, f, indent=2)

    print(f"File integrity checked across all 4 runs.")
    print(f"Total statistical/qualitative claim occurrences found across reports: {len(claims)}")

if __name__ == '__main__':
    main()
