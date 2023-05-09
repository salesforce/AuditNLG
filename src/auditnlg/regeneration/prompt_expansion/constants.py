'''
 * Copyright (c) 2023, Salesforce, Inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

from langchain import PromptTemplate

request_factuality_guidelines_template = """Instruction: {model_input}

Output: {model_output}

The model-generated output may contain facts that are inconsistent with the facts in the instruction. To ensure the model's output is accurate and does not contain any false information, create guidelines that require the model to reproduce all facts from the instruction exactly as they are stated.

Guidelines:
"""

request_factuality_guidelines_prompt_template = PromptTemplate(input_variables=["model_input", "model_output"], template=request_factuality_guidelines_template)

revised_factuality_template = """Instruction: {model_input}

Follow the guidelines below:
{guideline}

"""

revised_factuality_prompt_template = PromptTemplate(input_variables=["model_input", "guideline"], template=revised_factuality_template)
