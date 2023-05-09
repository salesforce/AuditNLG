'''
 * Copyright (c) 2023, Salesforce, Inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

from ..llm_wrapper import OpenAILLM, LocalLLM

def create_model(method):
    if "openai" in method:
        return OpenAILLM(model_name = method.split("/")[1])
    else:
        task = "text-generation" if "gpt" in method else "text2text-generation"
        return LocalLLM(model_name = method.split("/#")[0], task = task)
