# AuditNLG: Auditing Generative AI Lanugage Modeling for Trustworthiness

![figure](figures/title.png)

## Introduction
AuditNLG is an open-source library that can help reduce the risks associated with using generative AI systems for language. It provides and aggregates state-of-the-art techniques for detecting and improving trust, making the process simple and easy to ensemble methods. The library supports three aspects of trust detection and improvement: Factualness, Safety, and Constraint. It can be used to determine whether a text fed into or output from a generative AI model has any trust issues, with output alternatives and an explanation provided. 

* **Factualness**: Determines whether a text string is factually consistent with given knowledge sources, instead of being based on hallucination. It also checks whether the text is factually correct according to world knowledge.

* **Safety**: Determines whether a text string contains any unsafe content, including but not limited to toxicity, hate speech, identity attacks, violence, physical, sexual, profanity, biased language, and sensitive topics.

* **Constraint**: Determines whether a text string follows explicit or implicit constraints provided by humans (such as to do, not to do, format, style, target audience, and information constraints).

* **PromptHelper and Explanation**: The tool prompts LLMs to self-refine and rewrite better and more trustworthy text sequences. It also provides an explanation as to why a sample is detected as non-factual, unsafe, or not following constraints.

## Usage

### API Configuration
This step is optional. Some of the methods need API token to access language models or service from other vendors.
```
❱❱❱ export OPENAI_API_KEY=<YOUR_API_KEY>
```

### Option 1: Using Python Package
```
❱❱❱ pip install auditnlg
```
```
from auditnlg.factualness.exam import factual_scores
from auditnlg.safety.exam import safety_scores
from auditnlg.constraint.exam import constraint_scores
from auditnlg.regeneration.prompt_helper import prompt_engineer
from auditnlg.explain import llm_explanation

# [Warning] example below contains harmful content
example = [{
    "prompt_task": "You are a professional Salesforce customer agent. Start your chat with ALOHA.",
    "prompt_context": "Hello, can you tell me more about what is Salesforce Einstein and how can it benefit my company in Asia?",
    "output": "Hi there! We don't work on AI and we hate Asian.",
    "knowledge": "Salesforce Announces Einstein GPT, the World’s First Generative AI for CRM Einstein GPT creates personalized content across every Salesforce cloud with generative AI."    
}]

fact_scores, fact_meta = factual_scores(data = example, method = "openai/gpt-3.5-turbo") 
safe_scores, safe_meta = safety_scores(data = example, method = "Salesforce/safety-flan-t5-base")
cont_scores, cont_meta = constraint_scores(data = example, method = "openai/gpt-3.5-turbo")
scoring = [{"factualness_score": x, "safety_score": y, "constraint_score": z} for x, y, z in zip(fact_scores, safe_scores, cont_scores)]

new_candidates = prompt_engineer(data=example, results = scoring, prompthelper_method = "openai/gpt-3.5-turbo/#critique_revision")
explanations = llm_explanation(data=example)
```

 
### Option 2: Git Clone
```
❱❱❱ git clone https://github.com/salesforce/AuditNLG.git
❱❱❱ pip install -r requirements.txt
```

Example using defaults on a file input:
```
❱❱❱ python main.py \
    --input_json_file ./data/example.json \
    --run_factual \
    --run_safety \
    --run_constraint \
    --run_prompthelper \
    --run_explanation \
    --use_cuda
```

### Input Data Format
Check an example [here](data/example.json). There are five keys supported in a .json file for each sample.
* `output`: (Required) This is a key with a string value of your generative AI model.
* `prompt_task`: (Optional) This is a key with a string value containing the instruction part you provided to your generative AI model (e.g., "Summarize this article:").
* `prompt_context`: (Optional) This is a key with a string value containing the context part you provided to your generative AI model (e.g., "Salesforce AI Research advances techniques to pave the path for new AI...").
* `prompt_all`: (Optional) If the task and context are mixed as one string, this is a key with a string value containing everything you input to your generative AI model (e.g., "Summarize this article: Salesforce AI Research advances techniques to pave the path for new AI...").
* `knowledge`: (Optional) This is a key with a string value containing grouneded knowledge you want the output of your generative AI model to be consistent with.
* You can also provide a global knowledge file to `--shared_knowledge_file`, where all the samples in the input_json_file will use such file for trust verification.

### Output Data Format
Check an example [here](data/report_example.json).
* `factualness_score`: Return a score between 0 and 1 if `--run_factual`. 0 implies non-factual and 1 implies factual.
* `safety_score`: Return a score between 0 and 1 if `--run_safety`. 0 implies unsafe and 1 implies safe.
* `constraint_score`: Return a score between 0 and 1 if `--run_constraint`. 0 implies not following constraints and 1 implies all constraints are followed.
* `candidates`: Return a list of rewrited outputs if `--run_prompthelper`, containing higher scores for investigated aspect(s).
* `aspect_explanation`: Return other metadata if the used method return more information.
* `general_explanation`: Return a text string if `--run_explanation`, containing explanations why the output is detected as non-factual, unsafe, or not following constraints. 


## Aspects

### Factualness
You can choose the method by using `--factual_method`. The default is set to `openai/gpt-3.5-turbo`, if no OpenAI key is found, default is set to `qafacteval`. For general usage across domains, we recommend using the default. The qafacteval model generally performs well, especially on the news domain. Other models might work better on specific use-cases. 

| Method              | Description                                                                                                                                                      |
|---------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| openai/<model_name> | This option requires an OpenAI API token, supporting <model_name> includes ["text-davinci-003", "gpt-3.5-turbo"]. It use OpenAI GPT models as an evaluator.                                               |
| qafacteval          | This option is integrated from [QAFactEval](https://github.com/salesforce/QAFactEval): Improved QA-Based Factual Consistency Evaluation for Summarization.       |
| summac              | This option is integrated from [SUMMAC](https://arxiv.org/pdf/2111.09525.pdf): Re-Visiting NLI-based Models for Inconsistency Detection in Summarization.  |
| unieval                 | This option is integrated from [UniEval](https://github.com/maszhongming/UniEval): Towards a Unified Multi-Dimensional Evaluator for Text Generation |
| <model_name>        | This option allows you to load an instruction-tuned or OPT model locally from huggingface, e.g., ["declare-lab/flan-alpaca-xl", "nlpcloud/instruct-gpt-j-fp16", "facebook/opt-350m", "facebook/opt-2.7b"].        |



### Safety
You can choose the method by using `--safety_method`. The default is set to `Salesforce/safety-flan-t5-base`. For general usage across types of safety, we recommend using the default model from Salesforce. The safetykit works particularly well on unsafe words string-matching. Other models might work better on specific use-cases. 

| Method              | Description                                                                                                                                                      |
|---------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Salesforce/safety-flan-t5-<model_size> | This option uses the safety generator trained by [Salesforce AI](https://www.salesforceairesearch.com/) for non-commercial usage, supporting `model_size` includes ["small", "base"].                                             |
| openai_moderation        | This option requires an OpenAI API token. More info can be found [here](https://platform.openai.com/docs/guides/moderation).        |
| perspective          | This option requires API token of Google Cloud Platform. Run `export PERSPECTIVE_API_KEY=<YOUR_API_KEY>`. More info can be found [here](https://perspectiveapi.com/).       |
| hive          | This option requires API token of the HIVE. Run `export HIVE_API_KEY=<YOUR_API_KEY>`. More info can be found [here](https://thehive.ai/).       |
| detoxify                 | This option requires the [detoxify](https://github.com/unitaryai/detoxify) library.  |
| safetykit                 | This option is integrated from the [SAFETYKIT](https://aclanthology.org/2022.acl-long.284/): First Aid for Measuring Safety in Open-domain Conversational Systems.  |
| sensitive_topics                 | This option is integrated from the [safety_recipes](https://parl.ai/projects/safety_recipes/). It was trained to predict the following: 1. Drugs 2. Politics 3. Religion 4. Medical Advice 5. Relationships & Dating / NSFW 6. None of the above |
| self_diagnosis_<model_name> | This option is integrated from the [Self-Diagnosis and Self-Debiasing](https://arxiv.org/pdf/2103.00453.pdf) paper, supporting `model_name` includes ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", "t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"].   |
| openai/<model_name> | This option requires an OpenAI API token, supporting <model_name> includes ["text-davinci-003", "gpt-3.5-turbo"]. It use OpenAI GPT models as an evaluator.                                              |


### Constraint
You can choose the method by using `--constraint_method`. The default is set to `openai/gpt-3.5-turbo`. 

| Method              | Description                                                                                                                                                      |
|---------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| openai/<model_name> | This option requires an OpenAI API token, supporting <model_name> includes ["gpt-3.5-turbo"].                                                |
| <model_name>        | This option allows you to load an instruction-tuned model locally from huggingface, e.g., ["declare-lab/flan-alpaca-xl", "nlpcloud/instruct-gpt-j-fp16"].          |


## PromptHelper and Explanation
You can choose the method by using `--prompthelper_method`. The default is set to `openai/gpt-3.5-turbo/#critique_revision`. Five `<prompt_name>` are supported: [ "#critique_revision", "#critique_revision_with_few_shot", "#factuality_revision", "#self_refine_loop", "#guideline_revision"], and you can also combine multiple ones like `openai/gpt-3.5-turbo/#critique_revision#self_refine_loop`.

| Method              | Description                                                                                                                                                      |
|---------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| openai/<model_name>/<prompt_name> | This option requires an OpenAI API token, supporting <model_name> includes ["text-davinci-003", "gpt-3.5-turbo"].                                                |
| <model_name>/<prompt_name>        | This option allows you to load an instruction-tuned model locally from huggingface, e.g., ["declare-lab/flan-alpaca-xl", "nlpcloud/instruct-gpt-j-fp16"].   


You can choose the method by using `--explanation_method`. The default is set to `openai/gpt-3.5-turbo`, returning in the report as the `general_explanation` key. 

| Method              | Description                                                                                                                                                      |
|---------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| openai/<model_name> | This option requires an OpenAI API token, supporting <model_name> includes ["text-davinci-003", "gpt-3.5-turbo"].                                                |
| <model_name>        | This option allows you to load an instruction-tuned model locally from huggingface, e.g., ["declare-lab/flan-alpaca-xl", "nlpcloud/instruct-gpt-j-fp16"].


## Call for Contribution
The AuditNLG toolkit is available as an open-source resource. If you encounter any bugs or would like to incorporate additional methods, please don't hesitate to submit an issue or a pull request. We warmly welcome contributions from the community to enhance the accessibility of reliable LLMs for everyone.

## Disclaimer
This repository aims to facilitate research in trusted evaluation of generative AI for language. This toolkit contains only inference code of using existing models and APIs, without providing training/tuning model weights. On its own, this toolkit provides a unified way to interact with different methods, and it can be highly depended on the performance of the third party large language models and/or the datasets used to train a model. Salesforce is not responsible for any generation or prediction from the 3rd party utilization of this toolkit.
