'''
 * Copyright (c) 2023, Salesforce, Inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

import numpy as np

from ..safety.exam import safety_scores
from ..factualness.exam import factual_scores
from ..constraint.exam import constraint_scores
from ..utils import example_format_checker

from .modeling import create_model
from .constitutional_ai.critique_revise import run_pipeline, run_few_shot_pipeline, improve_factuality
from .self_refine.pipeline import refine
from .prompt_expansion.prompt_with_guidelines import instruction_with_guidelines_factuality


def format_model_inputs(sample):
    original_input = sample["prompt_all"]
    if original_input == "":
        original_input = sample["prompt_task"] + '\n\n' + sample["prompt_context"]
    original_output = sample["output"]
    return original_input, original_output


def rerun_evaluations(sample, global_knowledge, run_factual, run_safety, run_constraint, factual_method, safety_method, constraint_method, batch_size, use_cuda, gpu_device):
    result = {"text": sample["output"]}
    if run_factual:
        scores, _ = factual_scores(factual_method, [sample], global_knowledge, use_cuda, gpu_device) 
        result["factualness_score"] = scores[0]
    if run_safety:
        safety_score, _ = safety_scores(safety_method, [sample], global_knowledge, batch_size, use_cuda)
        result["safety_score"] = safety_score[0]
    if run_constraint:
        scores, _ = constraint_scores(constraint_method, [sample], global_knowledge)
        result["constraint_score"] = scores[0]
    return result


def is_better_candidate(baseline_result, candidate_result):
    for key, score in baseline_result.items():
        if "_score" in key:
            if baseline_result[key] > candidate_result[key]:
                return False
    return True


def self_refine(sample, original_input, original_output, model, global_knowledge, run_factual, run_safety, run_constraint, factual_method, safety_method, constraint_method, batch_size, use_cuda, gpu_device, max_attempts: int = 2) -> str:
    def is_refinement_sufficient(original_scores, new_scores, num_attempts) -> bool:
        # Define stopping criteria here
        if num_attempts >= max_attempts:
            return True
        baseline = np.mean([score for key, score in original_scores.items() if key.endswith("_score")])
        new_avg = np.mean([score for key, score in new_scores.items() if key.endswith("_score")])
        return new_avg > baseline * 1.2

    copy_sample = sample.copy()
    copy_sample["output"] = original_output
    original_scores = rerun_evaluations(copy_sample, global_knowledge, run_factual, run_safety, run_constraint, factual_method, safety_method, constraint_method, batch_size, use_cuda, gpu_device)
    answer = original_output
    refined = None

    num_attempts = 0
    while True:
        copy_sample = copy_sample.copy()
        copy_sample["output"] = answer
        feedback_scores = rerun_evaluations(copy_sample, global_knowledge, run_factual, run_safety, run_constraint, factual_method, safety_method, constraint_method, batch_size, use_cuda, gpu_device)
        refined = refine(original_input, answer, feedback_scores, model)

        print(f"Step {num_attempts}: {refined}")
        num_attempts += 1

        if is_refinement_sufficient(original_scores, feedback_scores, num_attempts):
            print("break")
            break

        answer = refined

    return refined


def prompt_engineer(
        data = [], 
        results = [], 
        global_knowledge = "", 
        factual_method = "openai/gpt-3.5-turbo", 
        safety_method = "Salesforce/safety-flan-t5-base", 
        constraint_method = "openai/gpt-3.5-turbo", 
        prompthelper_method = "openai/gpt-3.5-turbo/#critique_revision", 
        prompthelper_only_better = False, 
        batch_size = 8, 
        use_cuda = False, 
        gpu_device = 0
    ):
    data = example_format_checker(data)
    model = create_model(prompthelper_method)
    # print(f"Model: {model.model_name}")
    # print(model.model_max_length)
    candidates = []
    for i, sample in enumerate(data):
        new_outputs = []
        original_input, original_output = format_model_inputs(sample)

        run_factual = True if "factualness_score" in results[i].keys() else False
        run_safety = True if "safety_score" in results[i].keys() else False
        run_constraint = True if "constraint_score" in results[i].keys() else False

        # Method 1. Critique-revise pipeline
        if "critique_revision" in prompthelper_method.split("#"):
            updated_output = run_pipeline(original_input, original_output, model)
            if updated_output:
                copy_sample = sample.copy()
                copy_sample["output"] = updated_output
                new_output = rerun_evaluations(copy_sample, global_knowledge, run_factual, run_safety, run_constraint, factual_method, safety_method, constraint_method, batch_size, use_cuda, gpu_device)
                if not prompthelper_only_better or is_better_candidate(baseline_result=results[i], candidate_result=new_output):
                    new_outputs.append(new_output)

        # Method 2. Critique-revise pipeline with few-shot examples
        if "#critique_revision_with_few_shot" in prompthelper_method.split("#"):
            updated_output = run_few_shot_pipeline(original_input, original_output, model)
            if updated_output:
                copy_sample = sample.copy()
                copy_sample["output"] = updated_output
                new_output = rerun_evaluations(copy_sample, global_knowledge, run_factual, run_safety, run_constraint, factual_method, safety_method, constraint_method, batch_size, use_cuda, gpu_device)

                if not prompthelper_only_better or is_better_candidate(baseline_result=results[i], candidate_result=new_output):
                    new_outputs.append(new_output)

        # Method 3. Factuality pipeline
        if "#factuality_revision" in prompthelper_method.split("#"):
            factuality_score = results[i]["factualness_score"]
            knowledge = sample['knowledge']
            updated_output = improve_factuality(original_input, original_output, knowledge, factuality_score, model)
            if updated_output:
                copy_sample = sample.copy()
                copy_sample["output"] = updated_output
                new_output = rerun_evaluations(copy_sample, global_knowledge, run_factual, run_safety, run_constraint, factual_method, safety_method, constraint_method, batch_size, use_cuda, gpu_device)
                if not prompthelper_only_better or is_better_candidate(baseline_result=results[i], candidate_result=new_output):
                    new_outputs.append(new_output)

        # Method 4. Self-refine pipeline
        if "#self_refine_loop" in prompthelper_method.split("#"):
            updated_output = self_refine(sample, original_input, original_output, model, global_knowledge, run_factual, run_safety, run_constraint, factual_method, safety_method, constraint_method, batch_size, use_cuda, gpu_device, max_attempts = 2)
            if updated_output:
                copy_sample = sample.copy()
                copy_sample["output"] = updated_output
                new_output = rerun_evaluations(copy_sample, global_knowledge, run_factual, run_safety, run_constraint, factual_method, safety_method, constraint_method, batch_size, use_cuda, gpu_device)
                if not prompthelper_only_better or is_better_candidate(baseline_result=results[i], candidate_result=new_output):
                    new_outputs.append(new_output)

        # Method 5. Instruction-expansion through fine-grained guidelines
        if "#guideline_revision" in prompthelper_method.split("#"):
            knowledge = sample['knowledge']
            updated_output = instruction_with_guidelines_factuality(original_input, original_output, knowledge, global_knowledge, model)
            if updated_output:
                copy_sample = sample.copy()
                copy_sample["output"] = updated_output[0]
                new_output = rerun_evaluations(copy_sample, global_knowledge, run_factual, run_safety, run_constraint, factual_method, safety_method, constraint_method, batch_size, use_cuda, gpu_device) 
                if not prompthelper_only_better or is_better_candidate(baseline_result=results[i], candidate_result=new_output):
                    new_outputs.append(new_output)

        candidates.append(new_outputs)
    return candidates
