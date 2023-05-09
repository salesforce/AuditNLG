from langchain import PromptTemplate
'''
 * Copyright (c) 2023, Salesforce, Inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

gpt3_score_template = """{instruction}

Ensure that the generated text is factually accurate and consistent. Refer to local knowledge or general knowledge for missing or additional factual information.
Local Knowledge: {local_knowledge}
Global Knowledge: {global_knowledge}

Generated text: {output}"""

chatgpt_evaluator_template = """Score the output text, given the corresponding instruction, with respect to consistency using a scale of one to five stars. One star indicates "inconsistency" while five stars means "perfect consistency". Note that consistency measures whether the facts in the output are consistent with the facts in the instruction. Consider whether the output does reproduce all facts accurately and does not make up untrue information. The contextual text may include instruction and source text, local knowledge that is specific to the source text, and global knowledge.

Instruction: {instruction}
Local Knowledge: {local_knowledge}
Global Knowledge: {global_knowledge}
Output: {output}
Stars:"""

localllm_evaluator_template = """Is the Output factual given the corresponding instruction and knowledge? Give of score of 1, 2, 3 where 1 means seriously inconsistent, 2 means somewhat inconsistent, and 3 means consistent.

Instruction: {instruction}
Local Knowledge: {local_knowledge}
Global Knowledge: {global_knowledge}
Output: {output}

The score is:"""

opt_score_instruction_template = """{instruction}

Local Knowledge: {local_knowledge}
Global Knowledge: {global_knowledge}
"""

prompt_templates = {
    'chatgpt_evaluator': PromptTemplate(input_variables=["instruction", "local_knowledge", "global_knowledge", "output"], template=chatgpt_evaluator_template),
    'gpt3_score': PromptTemplate(input_variables=["instruction", "local_knowledge", "global_knowledge", "output"], template=gpt3_score_template),
    'localllm_score': PromptTemplate(input_variables=["instruction", "local_knowledge", "global_knowledge", "output"], template=localllm_evaluator_template),
    'opt_score': PromptTemplate(input_variables=["instruction", "local_knowledge", "global_knowledge"], template=opt_score_instruction_template),
}
