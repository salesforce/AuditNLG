'''
 * Copyright (c) 2023, Salesforce, Inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

from .constants import prompt_template

def refine(prompt, response, feedback_scores, model):
    scores = [f"{key}: {score}" for key, score in feedback_scores.items() if key.endswith("_score")]
    input_txt = prompt_template.format(prompt=prompt, response=response, scores=scores)
    gen_param = {
        "max_tokens":128
    }
    return model.generate(input_txt, **gen_param)
