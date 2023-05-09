'''
 * Copyright (c) 2023, Salesforce, Inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

import argparse
import json
import os
from src.auditnlg.safety.exam import safety_scores
from src.auditnlg.factualness.exam import factual_scores
from src.auditnlg.constraint.exam import constraint_scores
from src.auditnlg.regeneration.prompt_helper import prompt_engineer
from src.auditnlg.explain import llm_explanation
from src.auditnlg.utils import setup_global_api_key, get_global_knowledge

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json_file", type=str, required=True, help="A json file that contains the input samples.")
    parser.add_argument("--shared_knowledge_file", type=str, default=None, help="A file that contains all the background knowledge for all samples in the input json file.")
    parser.add_argument("--run_factual", action="store_true", help="Run factualness evaluation.")
    parser.add_argument("--run_safety", action="store_true", help="Run safety evaluation.")
    parser.add_argument("--run_constraint", action="store_true", help="Run constraint evaluation.")
    parser.add_argument("--run_prompthelper", action="store_true", help="Run prompt helper.")
    parser.add_argument("--run_explanation", action="store_true", help="Run explanation.")
    parser.add_argument("--factual_method", type=str, default="openai/gpt-3.5-turbo", help="The model used for factualness evaluation.")
    parser.add_argument("--safety_method", type=str, default="Salesforce/safety-flan-t5-base", help="The model used for safety evaluation.")
    parser.add_argument("--constraint_method", type=str, default="openai/gpt-3.5-turbo", help="The model used for constraint evaluation.")
    parser.add_argument("--prompthelper_method", type=str, default="openai/gpt-3.5-turbo/#critique_revision", help="The model used for prompt helper.")
    parser.add_argument("--prompthelper_only_better", action="store_true", help="Only use the prompt helper if it improves the trust score.")
    parser.add_argument("--explanation_method", type=str, default="openai/gpt-3.5-turbo", help="The model used for explanation.")
    parser.add_argument("--output_path", type=str, default="./output", help="The path to save the output file.")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size for local model inference")
    parser.add_argument("--use_cuda", action="store_true", help="Use GPU for local model inference.")
    parser.add_argument("--gpu_device", type=int, default=0)
    return parser.parse_args()


def run_evaluation(args, data, knowledge):
    '''
        data: List of dict
        knowledge: List of str
    '''

    results = [{"id": data[i]["id"]} for i in range(len(data))]

    if args.run_factual:
        print("Running Factualness Evaluation...")
        factual_s, meta_data = factual_scores(args.factual_method, data, knowledge, args.use_cuda, args.gpu_device) 
        for i, x in enumerate(factual_s):
            results[i]["factualness_score"] = x
            if "aspect_explanation" in meta_data.keys(): 
                results[i].setdefault("aspect_explanation", {})["factual"] = meta_data["aspect_explanation"][i]

    if args.run_safety:
        print("Running Safety Evaluation...")
        safety_s, meta_data = safety_scores(args.safety_method, data, knowledge, args.batch_size, args.use_cuda)
        for i, x in enumerate(safety_s):
            results[i]["safety_score"] = x
            if "all_scores" in meta_data.keys(): results[i]["safety_meta"] = meta_data["all_scores"][i]
            if "aspect_explanation" in meta_data.keys(): 
                results[i].setdefault("aspect_explanation", {})["safety"] = meta_data["aspect_explanation"][i]

    if args.run_constraint:
        print("Running Constraint Evaluation...")
        constraint_s, meta_data = constraint_scores(args.constraint_method, data, knowledge)
        for i, x in enumerate(constraint_s):
            results[i]["constraint_score"] = x
            if "aspect_explanation" in meta_data.keys(): 
                results[i].setdefault("aspect_explanation", {})["constraint"] = meta_data["aspect_explanation"][i]

    return results


def main():
    args = parse_args()
    # print(args)

    # setup api keys
    setup_global_api_key(args)

    # Read and parse input into a list of dict objects
    if args.input_json_file.endswith(".json"):
        with open(args.input_json_file, "r") as f:
            data = json.load(f)
    else:
        try:
            data = [json.loads(args.input_json_file)]
        except:
            raise ValueError("Input file must be a .json file or a json string.")

    # Read and parse knowledge file into a list of string objects
    global_knowledge = get_global_knowledge(args)

    # Trust Detection
    results = run_evaluation(args, data, global_knowledge)

    # Trust Improvement
    candidates = []
    if args.run_prompthelper:
        print("Running Prompt Helper...")
        candidates = prompt_engineer(data, results, global_knowledge, args.factual_method, args.safety_method, args.constraint_method, args.prompthelper_method, args.prompthelper_only_better, args.batch_size, args.use_cuda, args.gpu_device)
        for i, candidate_list in enumerate(candidates):
            results[i]["candidates"] = candidate_list
    if args.run_explanation:
        print("Running Explanation...")
        explanations = llm_explanation(args.explanation_method, data, global_knowledge)
        for i, explanation in enumerate(explanations):
            results[i]["general_explanation"] = explanation

    # Save prediction to an output folder
    print("Done.")
    dumped = json.dumps({
        "args": vars(args),
        "results": results,
        }, indent=4)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    output_file = os.path.join(args.output_path, "report.json")
    with open(output_file, "w") as f:
        f.write(dumped)
        print("Report saved to {}".format(output_file))
    print("One Output Example:\n{}".format(json.dumps(results[0], indent=4)))

if __name__ == "__main__":
    main()