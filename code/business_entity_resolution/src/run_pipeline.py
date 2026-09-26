import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def parse_args():
    parser = argparse.ArgumentParser(description="End-to-End Business Entity Resolution Pipeline")
    parser.add_argument("--mode", type=str, choices=["train", "predict", "all"], default="all",
                        help="Pipeline mode to execute")
    parser.add_argument("--data-dir", type=str, default=None, help="Dataset directory")
    parser.add_argument("--sample-size", type=int, default=50000,
                        help="Sample size for training (0 for entire train set)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Explicit prediction threshold (optional)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_script = os.path.join(script_dir, "train.py")
    inference_script = os.path.join(script_dir, "inference.py")
    
    python_cmd = sys.executable
    
    if args.mode in ("train", "all"):
        print("=" * 60)
        print("STAGE 1: TRAINING & THRESHOLD OPTIMIZATION")
        print("=" * 60)
        cmd = f'"{python_cmd}" "{train_script}" --sample-size {args.sample_size}'
        if args.data_dir:
            cmd += f' --data-dir "{args.data_dir}"'
        print(f"Executing: {cmd}")
        ret = os.system(cmd)
        if ret != 0:
            print("Training failed with exit code:", ret)
            sys.exit(ret)
            
    if args.mode in ("predict", "all"):
        print("\n" + "=" * 60)
        print("STAGE 2: TEST INFERENCE & TSV SUBMISSION GENERATION")
        print("=" * 60)
        cmd = f'"{python_cmd}" "{inference_script}"'
        if args.data_dir:
            cmd += f' --data-dir "{args.data_dir}"'
        if args.threshold is not None:
            cmd += f' --threshold {args.threshold}'
        print(f"Executing: {cmd}")
        ret = os.system(cmd)
        if ret != 0:
            print("Inference failed with exit code:", ret)
            sys.exit(ret)
            
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("Next step: Run the validator:")
    print("python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir student_resource/dataset/test")
    print("=" * 60)

if __name__ == "__main__":
    main()
