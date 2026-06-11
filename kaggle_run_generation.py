import subprocess
import os

def main():
    models = ['Facenet512', 'ArcFace', 'GhostFaceNet', 'VGG-Face']
    
    # Automatically search Kaggle's input directory for the actual dataset root
    dataset_base = '/kaggle/input'
    real_dataset_root = None
    
    for root, dirs, files in os.walk(dataset_base):
        if 'lfw_pairs' in dirs:
            real_dataset_root = root
            break
            
    print(f"Auto-detected dataset root at: {real_dataset_root}")
    
    for model in models:
        print(f"\n{'='*50}\nStarting generation for {model} using ATT_CNN...\n{'='*50}")
        cmd = [
            "python", "experiments/run_vanilla_subset_generation.py",
            "--input-csv", "docs/subset_input_pairs.csv",
            "--dataset-root", real_dataset_root,
            "--output-root", "/kaggle/working/results_new_attack",
            "--attacker-model", model,
            "--attacks", "ATT_CNN"
        ]
        
        subprocess.run(cmd, check=True)
        
    print("\nAll generations finished successfully on Kaggle!")

if __name__ == '__main__':
    main()
