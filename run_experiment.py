import os 

def run_experiment(save, dir):
    if save:
        os.makedirs(dir, exist_ok=True)
        print(f"Directory {dir} created")
    else:
        print("Directory not created")
