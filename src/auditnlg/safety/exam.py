'''
 * Copyright (c) 2023, Salesforce, Inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

import os
from ..utils import example_format_checker

def safety_scores(
        method = "Salesforce/safety-flan-t5-base", 
        data = [], 
        global_knowledge = "", 
        batch_size = 8, 
        use_cuda = False
    ):

    meta_data = {}
    scores = []

    data = example_format_checker(data)

    if "Salesforce/safety-flan-t5" in method:
        from .classifier import safety_generation
        scores, predictions = safety_generation(data, method, batch_size, use_cuda)
        meta_data["aspect_explanation"] = predictions
    
    elif method == "openai_moderation":
        from .classifier import openai_moderation_API
        scores, all_scores = openai_moderation_API(data)
        meta_data["all_scores"] = all_scores

    elif "openai/" in method:
        from .classifier import openai_score
        scores, predictions = openai_score(data, method)
        meta_data["aspect_explanation"] = predictions

    elif "self_diagnosis" in method:
        from .classifier import self_diagnosis_Schick_et_al
        model_name = method.split("self_diagnosis_")[1]
        scores, all_scores = self_diagnosis_Schick_et_al(data, model_name, batch_size, use_cuda)
        meta_data["all_scores"] = all_scores
    
    elif method == "detoxify":
        from .classifier import detoxify
        scores, all_scores = detoxify(data, use_cuda)
        meta_data["all_scores"] = all_scores
        
    elif method == "perspective":
        from .classifier import perspective_API
        scores, all_scores = perspective_API(data, os.environ.get("PERSPECTIVE_API_KEY"))
        meta_data["all_scores"] = all_scores

    elif method == "safetykit":
        from .classifier import safetykit
        scores = safetykit(data)
    
    elif method == "hive":
        from .classifier import run_hive
        scores, all_scores = run_hive(data, os.environ.get("HIVE_API_KEY"))
        meta_data["all_scores"] = all_scores

    elif method == "sensitive_topics":
        from .classifier import sensitive_topics_classifier
        scores, all_scores = sensitive_topics_classifier(batch_size, data)
        meta_data["all_scores"] = all_scores

    else:
        print("[Warning] No method [{}] found".format(method))

    assert len(scores) == len(data)

    return scores, meta_data
