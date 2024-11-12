import os
import time
from tqdm import tqdm

def run_fold(fold_name, model, code):
    print(f"Start Running {model}: {fold_name}")
    start_time = time.time()
    work_dir = os.getcwd()

    # Construct the path to mymain.py
    script_path = os.path.join(work_dir, code)

    # Change directory to the current fold
    data_dir = os.path.join('proj1', fold_name)
    os.chdir(data_dir)

    # Execute the script
    try:
        os.system(f"python3 {script_path}")  # Replace with appropriate Python execution command
        print(f"Successfully executed mymain.py in {fold_name}")
    except Exception as e:
        print(f"Error running mymain.py in {fold_name}: {e}")

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time for {fold_name}: {execution_time:.2f} seconds")
    # print(f"Complete Running: {fold_name}\n")

    # Change back to the original working directory
    os.chdir(work_dir)  # return to the initial working directory

# Both Models: mymain.py
if __name__ == "__main__":
    # Run both Tree and LM at the same time:
    folds = [f"fold{i}" for i in range(1, 11)]
    model = 'XGBoost and Ridge' 
    code = 'mymain.py'

    for fold_name in tqdm(folds, desc='Running Models'):
        run_fold(fold_name, model, code)
        print()

    print("All folds processed!")