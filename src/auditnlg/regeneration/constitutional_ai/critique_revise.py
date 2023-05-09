'''
 * Copyright (c) 2023, Salesforce, Inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

from .constants import (
    critique_prompt_template, revise_prompt_template,
    critique_few_shot_prompt_template, revise_few_shot_prompt_template,
    factuality_prompt_template
)

gen_param = {
    "max_tokens": 128
}

def run_pipeline(model_input: str, model_output: str, model) -> str:
    critique_prompt = critique_prompt_template.format(model_input=model_input, model_output=model_output)
    critique = model.generate(critique_prompt, **gen_param)
    # print(f"Critique output: {critique}")

    revise_prompt = revise_prompt_template.format(model_input=model_input, model_output=model_output, critique=critique)
    revise = model.generate(revise_prompt, **gen_param)
    # print(f"Revision output: {revise}")
    return revise


def run_few_shot_pipeline(model_input: str, model_output: str, model) -> str:
    if model.model_max_length <= 2048:
        # TODO: vary few-shot examples based on what context window allows
        return ""
    critique_prompt_few_shot = critique_few_shot_prompt_template.format(model_input=model_input, model_output=model_output)
    critique = model.generate(critique_prompt_few_shot, **gen_param)
    # print(f"Few-shot Critique output: {critique}")

    revise_prompt_few_shot = revise_few_shot_prompt_template.format(model_input=model_input, model_output=model_output, critique=critique)
    revise = model.generate(revise_prompt_few_shot, **gen_param)
    print(f"Few-shot Revision output: {revise}")
    return revise


def improve_factuality(model_input: str, model_output: str, knowledge: str, factuality_score: float, model) -> str:
    if not knowledge:
        return ""
    prompt = factuality_prompt_template.format(
        model_input=model_input, model_output=model_output, knowledge=knowledge, factuality_score=factuality_score
    )
    revise = model.generate(prompt, **gen_param)
    # print(f"Factuality Revision output: {revise}")
    return revise