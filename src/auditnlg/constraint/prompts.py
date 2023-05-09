from langchain import PromptTemplate
'''
 * Copyright (c) 2023, Salesforce, Inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

"""
TODO: Update such that the constraints are retrieved from a database according to the relevance?
TODO: Structure constraint (Constraint 11 is very open ended, consider breaking it down into is_bullets, is_dialogue etc.,), 
in case it is difficult for the LLM to understand the structure constraint as a whole
"""
identification_only = """
Given the instruction and the list of available constraints below, select which constraints should be checked for this particular instruction. Only select constraints/rules from the given list that are explicitly mentioned in the instruction. Ignore optional or implied constraints. If there are no constraints found from the above list, the output will be 'No Constraints'. For each constraint, specify what the filled function call should look like. See the examples below and follow their structure.

List of constraints:

* Constraint 1: Number of characters. Function template: constraint_num_characters(<output_text>, <min_num_characters>, <max_num_characters>)
* Constraint 2: Number of words. Function template: constraint_num_words(<output_text>, <min_num_words>, <max_num_words>)
* Constraint 3: Number of sentences. Function template: constraint_num_sentences(<output_text>, <min_num_sentences>, <max_num_sentences>)
* Constraint 4: Number of paragraphs. Function template: constraint_num_paragraphs(<output_text>, <min_num_paragraphs>, <max_num_paragraphs>)
* Constraint 5: Check lowercase. Function template: constraint_check_lowercase(<output_text>)
* Constraint 6: Check uppercase. Function template: constraint_check_uppercase(<output_text>)
* Constraint 7: Starts with. Function template: constraint_starts_with(<output_text>, <start_string>)
* Constraint 8: Ends with. Function template: constraint_ends_with(<output_text>, <end_string>)
* Constraint 9: Contains. Function template: constraint_contains(<output_text>, <contains_string>)
* Constraint 10: Does not contain. Function template: constraint_does_not_contain(<output_text>, <does_not_contain_string>)
* Constraint 11: Structure constraint. Function template: constraint_structure(<output_text>, <style_requested>)


Instruction: Summarize the given document in 1 sentence.
Output:
1. Constraint 3: constraint_num_sentences(<output_text>, 1, 1).

Instruction: Summarize the given document in 2-3 sentences and keep it lowercase.
Output:
1. Constraint 3: constraint_num_sentences(<output_text>, 2, 3).
2. Constraint 5: constraint_check_lowercase(<output_text>).

Instruction: Write a summary of the given document in upper case characters only, with not more than 50 words.
Output:
1. Constraint 2: constraint_num_words(<output_text>, 1, 50). 
2. Constraint 6: constraint_check_uppercase(<output_text>). 

Instruction: Write an email to a marketing professional in less than 200 words and 3 bullet points
Output:
1. Constraint 2: constraint_num_words(<output_text>, 1, 200). 
2. Constraint 11: constraint_structure(<output_text>, bullet points). 
3. Constraint 3: constraint_num_sentences(<output_text>, 3, 3). 

Instruction: Hello, Can you tell me some good strategies for maintaining good work life balance?
Output:
No Constraints.

Instruction: Write me a short article on the effects on the Covid pandemic?
Output:
No Constraints.

Instruction: {instructions}
Output:
"""

constraint_identification_prompt = PromptTemplate(
    input_variables=["instructions"],
    template=identification_only,
)

checking_only = """
Your job is to assess the quality of a text document, given a set of rules or constraints. You will be provided with a text document and a list of rules or constraints to check one by one. For each constraint answer Yes/No if the given text satisfies it. Only check for constaints that you are completely sure about. See the examples below and follow their structure.

List of constraints:

* Constraint 1: Number of characters. Function template: constraint_num_characters(<output_text>, <min_num_characters>, <max_num_characters>)
* Constraint 2: Number of words. Function template: constraint_num_words(<output_text>, <min_num_words>, <max_num_words>)
* Constraint 3: Number of sentences. Function template: constraint_num_sentences(<output_text>, <min_num_sentences>, <max_num_sentences>)
* Constraint 4: Number of paragraphs. Function template: constraint_num_paragraphs(<output_text>, <min_num_paragraphs>, <max_num_paragraphs>)
* Constraint 5: Check lowercase. Function template: constraint_check_lowercase(<output_text>)
* Constraint 6: Check uppercase. Function template: constraint_check_uppercase(<output_text>)
* Constraint 7: Starts with. Function template: constraint_starts_with(<output_text>, <start_string>)
* Constraint 8: Ends with. Function template: constraint_ends_with(<output_text>, <end_string>)
* Constraint 9: Contains. Function template: constraint_contains(<output_text>, <contains_string>)
* Constraint 10: Does not contain. Function template: constraint_does_not_contain(<output_text>, <does_not_contain_string>)
* Constraint 11: Structure constraint. Function template: constraint_structure(<output_text>, <style_requested>)


Input Text: FOXO factors play a critical role in hematopoietic stem cell regeneration and stress resistance, with potential implications for aging and malignant stem cell biology.
Constraints:
1. Constraint 3: constraint_num_sentences(<input_text>, 1, 1)
2. Constraint 7: constraint_starts_with(<input_text>, 'FOXO')
3. Constraint 11: constraint_structure(<input_text>, 'dialogue')
Output:
1. Constraint 3: Yes
2. Constraint 7: Yes
3. Constraint 11: No


Input Text: The FOXO family of transcription factors regulates stress resistance, cell-cycle arrest, and metabolism. Recent studies have highlighted the essential role of FOXO-dependent signaling in the long-term regeneration potential of hematopoietic stem cells. These findings have implications for understanding the mechanisms of aging and malignant stem cell biology.
Constraints:
1. Constraint 3: constraint_num_sentences(<input_text>, 1, 2)
2. Constraint 5: constraint_check_lowercase(<input_text>)
3. Constraint 9: constraint_contains(<input_text>, "hematopoietic stem cells")

Output:
1. Constraint 3: No
2. Constraint 5: No
3. Constraint 9: Yes


Input Text: After two injury-plagued seasons, Nadal made a stellar return in one of the greatest comeback seasons of all time in 2013; reaching 14 finals, winning two majors and five Masters events including the US Open Series sweep (Summer Slam). He continued his dominance at the French Open, securing six titles, two US Open titles, an Australian Open title, and an Olympic doubles gold at the 2016 Rio Olympics with Marc López
Constraints:
1. Constraint 2: constraint_num_words(<input_text>, 1, 100)
2. Constraint 6: constraint_check_uppercase(<input_text>)
Output:
1. Constraint 2: Yes
2. Constraint 6: No

Instruction: Write an email to a marketing professional in less than 200 words and 3 bullet points
Input Text: Dear [Marketing Professional],

I hope this email finds you well. I am reaching out to you today because I believe there may be a mutually beneficial collaboration opportunity between our companies.

Here are a few reasons why I think this could be a great opportunity:

- Our companies have complementary products/services that could be leveraged for a joint marketing campaign.
- We share a similar target audience, which could allow us to reach new customers and increase our brand exposure.
- By working together, we could potentially pool our resources and expertise to create something truly unique and valuable for our customers.
If this sounds interesting to you, I would love to set up a call to discuss this further and explore potential ways we could work together. Please let me know if you are available next week to chat.

Thank you for your time, and I look forward to hearing back from you soon.

Best regards,
[Your Name]
Constraints:
1. Constraint 2: constraint_num_words(<input_text>, 1, 200)
2. Constraint 11: constraint_structure(<input_text>, bullets)
3. Constraint 3: constraint_num_sentences(<input_text>, 3, 3)
Output:
1. Constraint 2: Yes.
2. Constraint 11: Yes

Input Text: {llm_output}
{constraints}
Output: 

"""

constraint_checking_prompt = PromptTemplate(
    input_variables=["llm_output", "constraints"],
    template=checking_only,
)




