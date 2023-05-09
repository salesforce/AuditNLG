'''
 * Copyright (c) 2023, Salesforce, Inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

import os
import json
import openai
from .factualness.utils import retrieve_global_knowledge            

def example_format_checker(data):
    for i, x in enumerate(data):
        if "output" not in x.keys():
            print("[ERROR] Missing output in sample {}".format(i))
            exit(1)
        if "id" not in x.keys():
            x["id"] = i
        
        for key in ["prompt_all", "prompt_task", "prompt_context", "knowledge"]:
            if key not in x.keys():
                x[key] = ""
        
    return data


def knowledge_parser(knowledge):
    # TODO: We need to provide guidance to acceptable knowledge formats
    return list(filter(lambda x: x != "", knowledge.split("\n")))


def get_global_knowledge(args):
    global_knowledge = ""
    if args.shared_knowledge_file is not None:
        with open(args.shared_knowledge_file, "r") as f:
            global_knowledge = f.read()
        global_knowledge = knowledge_parser(global_knowledge)
    return global_knowledge


def setup_global_api_key(args):
    # setup api keys for openai and perspective

    openai.api_key = os.environ.get("OPENAI_API_KEY")
    openai.organization = os.environ.get("OPENAI_ORGANIZATION")

    if openai.api_key == None or openai.api_key == "":
        print("[Warning] No OpenAI API key found.")
        print("[Warning] Default methods will change to local methods.")
        if "openai" in args.factual_method and args.run_factual: 
            args.factual_method = "qafacteval" 
            print("[Warning] Default factual method changed to 'qafacteval'")
        if "openai" in args.safety_method and args.run_safety: 
            args.safety_method = "safety_generation/flan-t5-base"
            print("[Warning] Default safety method changed to 'safety_generation/flan-t5-base'")
        if "openai" in args.constraint_method and args.run_constraint: 
            args.constraint_method = "e2e/declare-lab/flan-alpaca-xl"
            print("[Warning] Default constraint method changed to 'e2e/declare-lab/flan-alpaca-xl'")
        if "openai" in args.prompthelper_method and args.run_prompthelper: 
            args.prompthelper_method = "declare-lab/flan-alpaca-xl"
            print("[Warning] Default prompthelper method changed to 'declare-lab/flan-alpaca-xl'")
        if "openai" in args.explanation_method and args.run_explanation: 
            args.explanation_method = "declare-lab/flan-alpaca-xl"
            print("[Warning] Default explanation method changed to 'declare-lab/flan-alpaca-xl'")

