'''
 * Copyright (c) 2023, Salesforce, Inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

from ..utils import example_format_checker

def factual_scores(
        method = "openai/gpt-3.5-turbo", 
        data = [], 
        global_knowledge = "", 
        use_cuda = False, 
        gpu_device = 0
    ):
    
    scores = []
    meta_data = {}
    
    data = example_format_checker(data)

    if method == "openai/text-davinci-003":
        from .classifier import gpt3_score
        scores, meta = gpt3_score("text-davinci-003", data, global_knowledge)
        meta_data["aspect_explanation"] = meta
    elif method == 'openai/gpt-3.5-turbo':
        from .classifier import chatgpt_score
        scores = chatgpt_score("gpt-3.5-turbo", data, global_knowledge)
    elif method == "unieval":
        from .classifier import unieval
        scores = unieval(data, global_knowledge, use_cuda, gpu_device)
    elif method in ["facebook/opt-125m", "facebook/opt-350m", "facebook/opt-1.3b", "facebook/opt-2.7b",
                    "facebook/opt-6.7b", "facebook/opt-13b", "facebook/opt-66b"]:
        from .classifier import opt_score
        scores, meta = opt_score(method, data, global_knowledge)
        meta_data["aspect_explanation"] = meta
    elif method == "summac":
        from .classifier import summac_metric
        scores = summac_metric(data, global_knowledge, use_cuda, gpu_device)
    elif method == "qafacteval":
        from .classifier import qafacteval_metric
        scores, tmp_meta = qafacteval_metric(data, global_knowledge, use_cuda, gpu_device)
        meta_data["aspect_explanation"] = tmp_meta
    else:
        from .classifier import localllm_score
        scores = localllm_score(method, data, global_knowledge)
    
    return scores, meta_data
