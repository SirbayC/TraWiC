import argparse
import json
import logging
import logging.config
import os

import pandas as pd
import torch
import yaml
from tqdm import tqdm

from checker import Checker
from models import SantaCoder

# load logging configuration
with open(os.path.join(os.getcwd(), "src", "logging_config.yaml"), "r") as f:
    config = yaml.safe_load(f.read())

logging.config.dictConfig(config)

parser = argparse.ArgumentParser(description="Trained Without My Consent")
parser.add_argument(
    "--language",
    type=str,
    default="py",
    help="language of the code",
)  # programming language
parser.add_argument(
    "--dataset_path",
    type=str,
    default="data",
    help="path to the dataset",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=32,
    help="batch size",
)  # batch size
parser.add_argument(
    "--model",
    type=str,
    default="santa_coder",
    help="model to use",
)
parser.add_argument(
    "--sorted",
    type=bool,
    default=False,
    help="sort the dataset",
)
parser.add_argument(
    "--run_num",
    type=str,
    default="4",
    help="run number",
)
parser.add_argument(
    "--working_dir",
    type=str,
    default=os.getcwd(),
    help="working directory",
)

args = parser.parse_args()

WORKING_DIR = os.getcwd() if args.working_dir == os.getcwd() else args.working_dir
print("\033[93m" + f"Working directory: {WORKING_DIR}" + "\033[0m")

model = SantaCoder() if args.model == "santa_coder" else None


def get_model_output(file_path) -> int:
    results = []
    file_checker = Checker(file_path)
    model_inputs = [
        file_checker.prepare_inputs_for_infill(level=i)
        for i in [
            "function_names",
            "variable_names",
            "class_names",
            "comments",
            "docstrings",
            "strings",
        ]
    ]
    model_inputs = [input for sublist in model_inputs for input in sublist]

    if model_inputs == []:
        return None
    for candidate_input in tqdm(model_inputs):
        model_output = model.infill(
            (
                candidate_input["infill"],
                candidate_input["prefix"],
                candidate_input["suffix"],
                candidate_input["level"],
            )
        )
        if model_output == "too_many_tokens":
            f = open(os.path.join(os.getcwd(), "run_results", "too_many_tokens.txt"), "a")
            f.write(file_path + "\n")
            return 400
        else:
            try:
                result = file_checker.check_similarity(
                    model_output,
                    candidate_input,
                    similiarity_metric="exact"
                    if candidate_input["level"]
                    in ["function_names", "variable_names", "class_names"]
                    else "fuzzy",
                )
                results.append(
                    {
                        "file_path": file_path,
                        "level": candidate_input["level"],
                        "similarity_metric": "exact"
                        if candidate_input["level"]
                        in ["function_names", "variable_names", "class_names"]
                        else "fuzzy",
                        "result": result,
                        "similarity_objective": candidate_input["infill"],
                        "model_output": model_output,
                    }
                )
            except Exception as e:
                logging.error(e)
                return 500
    with open(
        os.path.join(
            os.getcwd(), "run_results", f"TokensRun{args.run_num}", "results.jsonl"
        ),
        "a",
    ) as f:
        json_results = json.dumps(results)
        f.write(json_results)
        f.write("\n")
    return 200


if __name__ == "__main__":
    print(args.sorted)
    print(type(args.run_num))
    print("Available devices: ", torch.cuda.device_count())

    if torch.cuda.is_available():
        logging.info(f"GPU is available. Running on {torch.cuda.get_device_name(0)}")
    else:
        logging.info("GPU is not available. Running on CPU")

    if not os.path.exists(
        os.path.join(os.getcwd(), "run_results", f"TokensRun{args.run_num}")
    ):
        os.mkdir(os.path.join(os.getcwd(), "run_results", f"TokensRun{args.run_num}"))

    dataset_files = []
    for dirpath, dirnames, filenames in os.walk(
        os.path.join(WORKING_DIR, args.dataset_path)
    ):
        python_files = [file for file in filenames if file.endswith(".py")]
        if python_files:
            dataset_files.extend(
                [os.path.join(WORKING_DIR, dirpath, file) for file in python_files]
            )

    dataset_files = (
        sorted(dataset_files) if args.sorted else sorted(dataset_files, reverse=True)
    )

    files_generated_blocks = open(
        os.path.join(os.getcwd(), "run_results", "generated.txt"), "r"
    ).readlines()  # read already processed files

    files_generated_blocks = [file.rstrip("\n") for file in files_generated_blocks]

    files_generated_blocks = (
        sorted(files_generated_blocks)
        if args.sorted
        else sorted(files_generated_blocks, reverse=True)
    )

    already_processed_files = open(
        os.path.join(os.getcwd(), "run_results", "processed_tokens.txt"), "r"
    ).readlines()  # read already processed files
    already_processed_files = [file.rstrip("\n") for file in already_processed_files]

    dangerous_files = open(
        os.path.join(WORKING_DIR, "run_results", f"assert_errors.txt"),
        "r",
    ).readlines()
    dangerous_files = [file.rstrip("\n") for file in dangerous_files]

    large_files = open(
        os.path.join(WORKING_DIR, "run_results", f"too_many_tokens.txt"),
        "r",
    ).readlines()
    large_files = [file.rstrip("\n") for file in large_files]

    stack_count = 0
    repo_count = 0
    LIMIT = 50  # Process 50 files from each class (Total 100)

    for file_path in dataset_files:
        # 1. Check if we should skip this file based on logs
        if file_path in dangerous_files or file_path in large_files:
            print("\033[91m" + file_path + "\033[0m")
            print("Skipping...")
            continue

        # 2. Determine if file is Member (Stack) or Non-Member (Repos)
        is_stack = "the_stack" in file_path

        # 3. Check class-specific limits
        if is_stack and stack_count >= LIMIT:
            continue
        if not is_stack and repo_count >= LIMIT:
            continue
        
        # 4. Process the file
        print("\033[91m" + file_path + "\033[0m")
        print("Processing...")
        
        if file_path in already_processed_files:
            print("Already processed")
            if is_stack:
                stack_count += 1
                print("Stack: " + str(stack_count))
            else:
                repo_count += 1
                print("Repo: " + str(repo_count))
            continue
            
        # Run the model (This function handles writing the results to JSONL)
        output_code = get_model_output(file_path)

        if output_code == 200:
            print("Successfully processed")
            if is_stack:
                stack_count += 1
                print("Stack: " + str(stack_count))
            else:
                repo_count += 1
                print("Repo: " + str(repo_count))
        else:
            print("Error: " + str(output_code))

        # 5. Log that we finished this file
        with open(
            os.path.join(WORKING_DIR, "run_results", "processed_tokens.txt"), "a"
        ) as f:
            f.write(file_path + "\n")

        # 6. Global Stop Condition
        if stack_count >= LIMIT and repo_count >= LIMIT:
            print("Reached limit for both classes. Stopping.")
            break