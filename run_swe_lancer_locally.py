from datetime import date
import pandas as pd
import subprocess
import time
import json
import argparse


"""
README: How to Use `run_swe_lancer.py`
----------------------------------------

This script is designed to run SWE manager and IC SWE tasks for the Swelancer project.

**Usage:**

Run this script from the root of the repository using Python 3.12+:

    python project/swelancer/run_swe_lancer.py --model <MODEL_NAME> [--task_types <TYPE>] [--individual_scenarios <ID1,ID2,...>]

**Arguments:**

- `--model` (required): The model to use for the solver. Must be one of: gpt-5, gpt-5-mini, gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-4, gpt-3.5-turbo.
- `--task_types` (optional): One of `ic_swe`, `swe_manager`, or `both`. Determines which set of tasks to run.
- `--individual_scenarios` (optional): Comma-separated list of scenario IDs to run (e.g., 123,456,789).

**Examples:**

- Run all SWE manager tasks with GPT-4o:
  
      python project/swelancer/run_swe_lancer.py --model gpt-4o --task_types swe_manager

- Run all IC SWE tasks with GPT-4o-mini:
  
      python project/swelancer/run_swe_lancer.py --model gpt-4o-mini --task_types ic_swe

- Run both types and combine results:
  
      python project/swelancer/run_swe_lancer.py --model gpt-4o --task_types both

- Run specific scenarios:
  
      python project/swelancer/run_swe_lancer.py --model gpt-4o --individual_scenarios 101,102,103

**Requirements:**

- Python 3.12 or higher
- The following Python packages must be installed:
    - pandas
    - argparse

**Notes:**

- The script expects the file `all_swelancer_tasks.csv` to be present in the current working directory.
- Output files will be saved in the same directory as the script, with filenames indicating the model and date.
- Model costs are defined in the `model_cost` dictionary at the top of the script.

"""

csvv = pd.read_csv("all_swelancer_tasks.csv")
swe_managers = csvv[csvv['variant'] == 'swe_manager'].sort_values(by="price", ascending=True)
all_swe_manager_ids = list(swe_managers['question_id'])

print("Data read.", flush=True)

# Model costs per 1M tokens (as of 2024-06, in USD)
# Source: https://openai.com/api/pricing/
model_cost = {
    "gpt-5": {
        "input": 20.0,    # Placeholder, update with actual price when available
        "output": 60.0    # Placeholder, update with actual price when available
    },
    "gpt-5-mini": {
        "input": 8.0,     # Placeholder, update with actual price when available
        "output": 24.0    # Placeholder, update with actual price when available
    },
    "gpt-4o": {
        "input": 5.0,     # $5.00 per 1M input tokens
        "output": 15.0    # $15.00 per 1M output tokens
    },
    "gpt-4o-mini": {
        "input": 2.0,     # Placeholder, update with actual price when available
        "output": 6.0     # Placeholder, update with actual price when available
    },
    "gpt-4-turbo": {
        "input": 10.0,    # $10.00 per 1M input tokens
        "output": 30.0    # $30.00 per 1M output tokens
    },
    "gpt-4": {
        "input": 30.0,    # $30.00 per 1M input tokens
        "output": 60.0    # $60.00 per 1M output tokens
    },
    "gpt-3.5-turbo": {
        "input": 0.5,     # $0.50 per 1M input tokens
        "output": 1.5     # $1.50 per 1M output tokens
    }
}

def run_one_manager(manager_id: int, task_value: int, model:str):
    cmd = [
            "uv", "run", "python", "swelancer/run_swelancer.py",
            "swelancer.split=diamond",
            "swelancer.task_type=swe_manager",
            f"swelancer.taskset=[{repr(manager_id)}]",
            "swelancer.solver=swelancer.solvers.swelancer_agent.solver:SimpleAgentSolver",
            f"swelancer.solver.model={model}",
            "swelancer.solver.computer_runtime=nanoeval_alcatraz.alcatraz_computer_interface:AlcatrazComputerRuntime",
            "swelancer.solver.computer_runtime.env=alcatraz.clusters.local:LocalConfig",
            "swelancer.solver.computer_runtime.env.pull_from_registry=False",
            "swelancer.docker_image_prefix=swelancer/swelancer_x86",
            "swelancer.docker_image_tag=releasev1",
            "swelancer.use_single_image=True",
            "runner.concurrency=4",
            "runner.experimental_use_multiprocessing=False",
            "runner.enable_slackbot=False",
            "runner.recorder=nanoeval.recorder:dummy_recorder",
            "runner.max_retries=3",
            "swelancer.disable_internet=False"
        ]
    # Time the command
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end_time = time.time()
    time_elapsed = end_time - start_time
    # Moved JSON dump to the very end so that all times are saved after all runs
    # (Remove this block from here; see end of file for the new location)
    with open('swelancer_eval_report.json', 'r') as f:
        data = json.load(f)
    manager_data = data[0]['reports'][-1]
    metrics_with_errors = manager_data['metrics_including_errors']
    accuracy_of_model = metrics_with_errors['accuracy']

    input_tokens, output_tokens, reasoning_tokens = manager_data['total_input_tokens'], manager_data['total_output_tokens'], manager_data['total_reasoning_tokens']
    input_cost = (input_tokens / 1_000_000) * model_cost[model]['input']
    output_cost = ((output_tokens + reasoning_tokens) / 1_000_000) * model_cost[model]['output']
    total_cost = input_cost + output_cost
    manager_data_formatted = {
        'question_id':manager_id,
        'accuracy_of_model':accuracy_of_model,
        'task_monetary_value':task_value,
        'cost_of_tokens_used':total_cost,
        'time_to_run':time_elapsed
    }

    return manager_data_formatted 

def run_all_swe_managers(model:str, date_formatted:str):
    # Initialize empty list if file doesn't exist, or load existing data
    try:
        with open("manager_time_spent.json", "r") as f:
            manager_data_collection = json.load(f)
            if isinstance(manager_data_collection, dict):
                # Convert dict to list format for consistency
                manager_data_collection = list(manager_data_collection.values())
    except (FileNotFoundError, json.JSONDecodeError):
        manager_data_collection = []

    for _, row in swe_managers.iterrows():
        task_value, manager_id = row['price'], row['question_id']
        manager_data_formatted = run_one_manager(manager_id, task_value, model)

        # Add to list and save incrementally
        manager_data_collection.append(manager_data_formatted)
        with open(f"manager_time_spent_{date_formatted}.json", "w") as f:
            json.dump(manager_data_collection, f, indent=2)

def run_one_ic_swe(ic_swe_id:int, task_value:int, model:str, variant:str):
    # INSERT_YOUR_CODE
    cmd = [
        "uv", "run", "python", "swelancer/run_swelancer.py",
        "swelancer.split=diamond",
        f"swelancer.task_type=ic_swe",
        f"swelancer.taskset=[{repr(ic_swe_id)}]",
        "swelancer.solver=swelancer.solvers.swelancer_agent.solver:SimpleAgentSolver",
        f"swelancer.solver.model={model}",
        "swelancer.solver.computer_runtime=nanoeval_alcatraz.alcatraz_computer_interface:AlcatrazComputerRuntime",
        "swelancer.solver.computer_runtime.env=alcatraz.clusters.local:LocalConfig",
        "swelancer.solver.computer_runtime.env.pull_from_registry=True",
        "swelancer.docker_image_prefix=swelancer/swelancer_x86",
        "swelancer.docker_image_tag=releasev1",
        "runner.concurrency=4",
        "runner.experimental_use_multiprocessing=False",
        "runner.enable_slackbot=False",
        "runner.recorder=nanoeval.recorder:dummy_recorder",
        "runner.max_retries=1"
    ]
    # Time the command
    # Time the command
    start_time = time.time()
    subprocess.run(cmd, capture_output=True, text=True)
    end_time = time.time()
    time_elapsed = end_time - start_time
    # Moved JSON dump to the very end so that all times are saved after all runs
    # (Remove this block from here; see end of file for the new location)
    with open('swelancer_eval_report.json', 'r') as f:
        data = json.load(f)
    ic_swe_data = data[0]['reports'][-1]
    metrics_with_errors = ic_swe_data['metrics_including_errors']
    accuracy_of_model = metrics_with_errors['accuracy']

    input_tokens, output_tokens, reasoning_tokens = ic_swe_data['total_input_tokens'], ic_swe_data['total_output_tokens'], ic_swe_data['total_reasoning_tokens']
    input_cost = (input_tokens / 1_000_000) * model_cost[model]['input']
    output_cost = ((output_tokens + reasoning_tokens) / 1_000_000) * model_cost[model]['output']
    total_cost = input_cost + output_cost
    ic_swe_data_formatted = {
        'question_id':ic_swe_id,
        'accuracy_of_model':accuracy_of_model,
        'task_monetary_value':task_value,
        'cost_of_tokens_used':total_cost,
        'time_to_run':time_elapsed
    }

    return ic_swe_data_formatted

def run_all_ic_swe(model:str, date_formatted:str):
    # Initialize empty list if file doesn't exist, or load existing data
    try:
        with open("ic_swe_results.json", "r") as f:
            ic_swe_data_collection = json.load(f)
            if isinstance(ic_swe_data_collection, dict):
                # Convert dict to list format for consistency
                ic_swe_data_collection = list(ic_swe_data_collection.values())
    except (FileNotFoundError, json.JSONDecodeError):
        ic_swe_data_collection = []

    # Load IC SWE tasks instead of using swe_managers
    ic_swe_tasks = csvv[csvv['variant'] == 'ic_swe'].sort_values(by="price", ascending=True)
    for _, row in ic_swe_tasks.iterrows():
        task_value, ic_swe_id = row['price'], row['question_id']
        ic_swe_data_formatted = run_one_ic_swe(ic_swe_id, task_value, model, 'ic_swe')

        # Add to list and save incrementally
        ic_swe_data_collection.append(ic_swe_data_formatted)
        with open(f"ic_swe_results_{date_formatted}.json", "w") as f:
            json.dump(ic_swe_data_collection, f, indent=2)

def run_list_of_scenarios(scenarios: list[int], model:str, date:str):
    swelancer_df=pd.read_csv("all_swelancer_tasks.csv")
    for scenario_id in scenarios:
        task_value = swelancer_df[swelancer_df['question_id']==scenario_id]['price'].iloc[0]
        # Check if this is a manager task by looking up the variant
        variant = swelancer_df[swelancer_df['question_id']==scenario_id]['variant'].iloc[0]
        if variant == 'swe_manager':
            _data_formatted = run_one_manager(scenario_id, task_value, model)
        else:
            _data_formatted = run_one_ic_swe(scenario_id, task_value, model, variant)
        with open(f"swelancer_run_{model}_{date}.json", "w") as f:
            json.dump(_data_formatted, f, indent=2)
    print(f"Run done. Results saved to swelancer_run_{model}_{date}.json")
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SWE managers with specified model")
    parser.add_argument("--model", required=True, help="Model to use for the solver")
    parser.add_argument(
        "--task_types",
        required=False,
        choices=["ic_swe", "swe_manager", "both"],
        default=None,
        help="Optional: choose one of 'ic_swe', 'swe_manager', or 'both' to specify which tasks to run."
    )
    parser.add_argument(
        "--individual_scenarios",
        required=False,
        type=lambda s: [item.strip() for item in s.split(",") if item.strip()],
        default=None,
        help="Comma-separated list of individual scenario IDs to run (e.g. 123,456,789)."
    )
    # If individual_scenarios argument is provided, save the values in a list
    today = date.today()
    args = parser.parse_args()

    if args.model not in list(model_cost.keys()):
        raise ValueError("must add model cost ")

    date_formatted = str(today.year)+"_"+str(today.month)+"_"+str(today.day)
    if args.individual_scenarios is not None:
        run_list_of_scenarios(args.individual_scenarios, args.model, date_formatted)
    elif args.task_types == "ic_swe":
        run_all_ic_swe(model=args.model, date_formatted=date_formatted)
    elif args.task_types == 'swe_manager':
        run_all_swe_managers(model=args.model, date_formatted=date_formatted)
    elif args.task_types == "both":
        run_all_ic_swe(model=args.model, date_formatted=date_formatted)
        run_all_swe_managers(model=args.model, date_formatted=date_formatted)
        # Combine the json files of the two runs together into one, and print out the filepath
        ic_swe_file = f"ic_swe_results_{date_formatted}.json"
        swe_manager_file = f"manager_time_spent_{date_formatted}.json"
        combined_file = f"combined_results_{date_formatted}.json"
        try:
            with open(ic_swe_file, "r") as f1, open(swe_manager_file, "r") as f2:
                ic_swe_data = json.load(f1)
                swe_manager_data = json.load(f2)
            # Combine the data; if both are lists, concatenate, else merge dicts
            if isinstance(ic_swe_data, list) and isinstance(swe_manager_data, list):
                combined_data = ic_swe_data + swe_manager_data
            elif isinstance(ic_swe_data, dict) and isinstance(swe_manager_data, dict):
                combined_data = {**ic_swe_data, **swe_manager_data}
            else:
                # fallback: put both under keys
                combined_data = {
                    "ic_swe": ic_swe_data,
                    "swe_manager": swe_manager_data
                }
            with open(combined_file, "w") as fout:
                json.dump(combined_data, fout, indent=2)
            print(f"Combined results saved to {combined_file}")
        except Exception as e:
            print(f"Error combining result files: {e}")


