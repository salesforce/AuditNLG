'''
 * Copyright (c) 2023, Salesforce, Inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

from .prompts import constraint_identification_prompt, constraint_checking_prompt
from ..llm_wrapper import OpenAILLM, LocalLLM
from tqdm import tqdm


# Read all the possible inputs and decide in the evaluation function how to use them.
def format_model_inputs(sample):
    return sample["prompt_all"], sample["prompt_task"], sample["prompt_context"], sample["output"]

"""
main function that evaluates and scores how well the constraints in the prompt are satisfied by the output

TODO:
1. What score for no constraints? (full?) - current scores it as 1.0 if no constraints identified
2. How to use the local and global knowledge here? -- Initial thoughts: Local or Global knowledge is only useful for factual verification not for constraint check


Scoring the constraints in 2 steps: 
1. constraint identification using only prompt as the input
2. checking constraints as a set of rules against the LLM output
"""

def evaluate_2step_prompt(method, data, global_knowledge=''):

    constraint_scores = []
    score_reasoning = []

    if "openai" in method:
        model = OpenAILLM(model_name = method.split("/")[-1])
    else:
        model_task = "text-generation" if "gpt" in method else "text2text-generation"
        model =  LocalLLM(model_name = method, task = model_task)

    for sample in tqdm(data):
        prompt_all, task, input_doc, llm_output = format_model_inputs(sample)

        # populate the prompt
        if prompt_all:  # if instruction and document provided together, no option but to use both for identification
            prompt_identification = constraint_identification_prompt.format(instructions=prompt_all)
        else:
            prompt_identification = constraint_identification_prompt.format(instructions=task)

        # run the prompt 
        constraints_found = model.generate(prompt=prompt_identification, messages="")

        if 'No Constraints.' in constraints_found:
            constraint_scores.append(1.0)
            score_reasoning.append(constraints_found)
            continue

        # if specific constraints found
        prompt_checking = constraint_checking_prompt.format(llm_output=llm_output, constraints=constraints_found)

        constraints_checked = model.generate(prompt=prompt_checking, messages="")

        satisfied = []
        for constraint in constraints_checked.split("\n"):
            
            if constraint and 'Constraint' in constraint:

                # count the num of yes-es
                if "Yes" in constraint:
                    satisfied.append(1)
                else:
                    satisfied.append(0)
            
        score = sum(satisfied)/len(satisfied)
        
        # Use full response text as the reasoning
        constraint_scores.append(score)
        score_reasoning.append(constraints_found + "\n" + constraints_checked)

    # TODO: Add exact constraint checked back into the reasoning for it to make sense 
    meta_data = {
        "aspect_explanation": score_reasoning
    }

    # print ("\n==================\nconstraint_scores 2 step: ", constraint_scores)
    return constraint_scores, meta_data

