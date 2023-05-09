'''
 * Copyright (c) 2023, Salesforce, Inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

from langchain import PromptTemplate

template = """Conversation history:

{prompt}
Response: {response}
Scores:
{scores}

Use the feedback scores to improve the response.

Conversation history:

{prompt}
Response:"""

prompt_template = PromptTemplate(input_variables=["prompt", "response", "scores"], template=template)
