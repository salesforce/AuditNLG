'''
 * Copyright (c) 2023, Salesforce, Inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

from .constants import (
    revised_factuality_prompt_template,
    request_factuality_guidelines_prompt_template
)


def get_source(sample):
    if sample["prompt_all"] == "":
        return ' '.join([sample["prompt_task"], sample["prompt_context"]])
    return sample["prompt_all"]


def instruction_with_guidelines_factuality(model_input, model_output, knowledge, global_knowledge, model):
    if knowledge != '':
        model_input += f'\n\nKnowledge:\n{knowledge}'
    if global_knowledge != '':
        model_input += f'\n\nGlobal Knowledge:\n{global_knowledge}'
    prompt = request_factuality_guidelines_prompt_template.format(
        model_input=model_input, model_output=model_output
    )

    gen_param = {
        "max_tokens": 128
    }

    guideline = model.generate(prompt, **gen_param)

    prompt = revised_factuality_prompt_template.format(
        model_input=model_input, guideline=guideline
    )
    revise = model.generate(prompt, **gen_param)

    # print(f"Factuality Guidelines: {guideline}, Revision:{revise}")

    return revise, guideline
