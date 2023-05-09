'''
 * Copyright (c) 2023, Salesforce, Inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

from .constraint_checker import evaluate_2step_prompt
from ..utils import example_format_checker

def constraint_scores(
        method = "openai/gpt-3.5-turbo", 
        data = [], 
        global_knowledge = ""
    ):

    data = example_format_checker(data)
    scores, meta_data = evaluate_2step_prompt(method, data, global_knowledge)
    return scores, meta_data
