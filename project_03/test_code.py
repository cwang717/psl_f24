import os  # Provides functions to interact with the operating system (e.g., change directories, run system commands)
import time  # Provides time-related functions (e.g., track elapsed time)
from tqdm import tqdm  # A module for showing progress bars in loops


folds = [f"split_{i}" for i in range(1, 6)]  # ['fold_1', 'fold_2', ..., 'fold_5']

cwd = os.getcwd()
# print(f"cwd: {cwd}")

project_data = os.path.join(cwd, 'F24_Proj3_data')
# print(f"Project Data: {project_data}")

code_path = os.path.join(cwd, 'mymain.py')
# print(f"code path: {code_path}")

for fold in tqdm(folds, desc=f'Running Splits'):
    fold_data = os.path.join(project_data, fold)
    os.chdir(fold_data)

    print(f"Start Running: in {fold}")
    start_time = time.time()

    try:
        os.system(f"python3 {code_path}")
        print(f"Successfully executed {fold}")
    
    # Handle any exceptions that might occur during the execution of the script
    except Exception as e:
        # Print an error message if the script fails to run (e.g., file not found, script error)
        print(f"Error running {fold}: {e}")

    # Record the ending time after the script has finished executing
    end_time = time.time()

    # Calculate the total execution time by subtracting the start time from the end time
    execution_time = end_time - start_time

    # Print the execution time, formatted to 2 decimal places, indicating how long the script took to execute for each split
    print(f"Execution time for {fold}: {execution_time:.2f} seconds")
    
    print()

os.chdir(cwd)
print("All splits processed!")