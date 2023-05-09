'''
 * Copyright (c) 2023, Salesforce, Inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

import openai
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM, OPTForCausalLM


class LocalLLM(object):
    def __init__(self, model_name: str, task: str) -> None:
        self.model_name = model_name
        self.task = task
        if task == "text-generation":
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map="auto")
        elif task == "text2text-generation":
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, device_map="auto")
        elif task == "llm-scorer":
            self.model = OPTForCausalLM.from_pretrained(self.model_name, device_map="auto")
        else:
            self.model = AutoModel.from_pretrained(self.model_name, device_map="auto")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model_max_length = self.tokenizer.model_max_length
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = 2000

    def generate(self, prompt: str, **gen_kwargs) -> str:
        input_ids = self.tokenizer([prompt], truncation=True, padding=True, return_tensors="pt").to(self.device)
        output_ids = self.model.generate(
            **input_ids,
            do_sample=True,
            temperature=0.9,
            max_new_tokens=64,
            # min_new_tokens=5,
            # **gen_kwargs
            )
        output_text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        if self.task == 'text2text-generation':
            output = output_text
        else:
            output = output_text[len(prompt):]
        return output

    def score(self, src: str, tgt: str, prompt_text: str) -> float:

        def trunk_input(inputs, outputs, reduce_seq, max_length):
            input_ids = self.tokenizer.encode(inputs)[1:-1]
            output_ids = self.tokenizer.encode(outputs)[1:-1]
            reduce_seq_ids = self.tokenizer.encode(reduce_seq)[1:-1]
            total_len = len(input_ids) + len(output_ids)
            if total_len > max_length:
                del_len = len(input_ids) + len(output_ids) - max_length
                reduce_seq_ids = reduce_seq_ids[:len(reduce_seq_ids) - del_len]
                reduce_seq = self.tokenizer.decode(reduce_seq_ids)
            return reduce_seq

        new_src = trunk_input(src, tgt, src, max_length=self.max_length)
        src = new_src
        text = src + prompt_text + tgt
        input_ids = self.tokenizer.encode(text)
        tgt_ids = self.tokenizer.encode(tgt)[1:]
        output_ids = [-100] * len(input_ids)
        output_ids[len(input_ids) - len(tgt_ids):] = tgt_ids
        input_ids = torch.LongTensor(input_ids).unsqueeze(0).to(self.device)
        output_ids = torch.LongTensor(output_ids).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                labels=output_ids,
                output_hidden_states=True
            )
        loss, logits, hidden_states = outputs[0], outputs[1], outputs.hidden_states[0]
        loss = loss.item()
        return loss


class OpenAILLM(object):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model_max_length = 4097
        self.usages = []
        self.completions = []
        self.responses = []

    def generate(self, prompt: str, messages: list = [], **gen_kwargs) -> str:
        if self.model_name == "gpt-3.5-turbo":
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": prompt}
                ] if len(messages) == 0 else messages
            )
            message = response["choices"][0]['message']['content']
        else:
            response = openai.Completion.create(
                model=self.model_name,
                prompt=prompt,
                **gen_kwargs
            )
            message = response["choices"][0]["text"]

        self.responses.append(response)
        self.usages.append(response["usage"])
        self.completions.append(response["choices"])
        return message
