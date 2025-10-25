import sys
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description='Eagle Predictors Main Entry Point')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['baseline', 'metrics', 'light', 'new'],
                       help='Select which main file to run')
    
    # آرگومان‌های اصلی رو می‌گیریم
    args, remaining_args = parser.parse_known_args()
    
    # تعیین فایل بر اساس mode
    module_map = {
        'baseline': 'eagle.predictors.__main__baseline_saving',
        'metrics': 'eagle.predictors.__main__baseline_metrics',
        'light': 'eagle.predictors.__main__light_gcn', 
        'new': 'eagle.predictors.__main__new'
    }
    
    module_name = module_map[args.mode]
    print(f"Running {module_name} in {args.mode} mode...")
    
    # اجرای فایل انتخاب شده با subprocess
    cmd = [sys.executable, '-m', module_name] + remaining_args
    print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"Error: {module_name} exited with code {result.returncode}")
        sys.exit(result.returncode)

if __name__ == "__main__":
    main()