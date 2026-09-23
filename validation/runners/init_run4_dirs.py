import os

def main():
    base_dir = 'validation/results/complex_task_synthesis_optimization'
    subdirs = ['raw', 'processed', 'figures', 'reports']
    for sd in subdirs:
        p = os.path.join(base_dir, sd)
        os.makedirs(p, exist_ok=True)
        print(f"Directory ready: {p}")

if __name__ == '__main__':
    main()
