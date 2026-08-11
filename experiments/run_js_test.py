import subprocess
import sys

def main():
    cmd = ["node", "experiments/test_format_latency_js.js"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print("STDOUT:\n", res.stdout)
    print("STDERR:\n", res.stderr)
    print("EXIT CODE:", res.returncode)
    sys.exit(res.returncode)

if __name__ == "__main__":
    main()
