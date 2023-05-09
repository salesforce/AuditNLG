'''
 * Copyright (c) 2023, Salesforce, Inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

# Using Sensitive Topics Classifier from safety_recipes https://parl.ai/projects/safety_recipes/
# parlai eval_model -mf zoo:sensitive_topics_classifier/model -t sensitive_topics_evaluation -dt valid -bs 16
import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from scipy.special import softmax

from parlai.core.opt import Opt
from parlai.core.message import Message
from parlai.core.worlds import World
from parlai.core.build_data import modelzoo_path
from parlai.agents.transformer.transformer import TransformerClassifierAgent as TCA


class_list = ['none', 'politics', 'religion', 'medical', 'nsfw', 'drugs']


def get_batch(data, batch_size=1):
    """
    Return a batch of data
    """
    l = len(data)
    for ndx in range(0, l, batch_size):
        yield data[ndx:min(ndx + batch_size, l)]


def get_model(data_path="/tmp"):
    """
    Get the model instance
    """
    # download model
    model_name = "zoo:sensitive_topics_classifier/model"
    model_file = modelzoo_path(data_path, model_name)

    # load opt
    optfile = model_file + '.opt'
    opt = Opt.load(optfile)
    TCA.upgrade_opt(opt)

    # overide
    opt["model_file"] = model_file
    opt["dict_file"] = model_file + '.dict'
    # opt['history_size'] = 0

    # load model
    model = TCA(opt)
    model.model.eval()
    # model.histsz = opt['history_size']
    return model


def sensitive_scorer(batch_size, data):
    with torch.no_grad():
        model = get_model(data_path="/tmp")
        batch_world = BatchWorld(batch_size, model)
        scores, meta_data = batch_world.parley(data)
        return scores, meta_data


class BatchWorld(World):
    """
    Run batch inference in parlai
    """
    def __init__(self, batch_size, model):
        # store the originals
        self.model = model
        self.batch_size = batch_size

        self.model_copies = []
        for i in range(self.batch_size):
            self.model_copies.append(self.model.clone())

    def clear_history(self):
        for i in range(self.batch_size):
            # reset after batch
            self.model_copies[i].reset()

    def parley(self, data):
        safety_scores = []
        meta_data = []
        for i, batch_data in tqdm(enumerate(get_batch(data, batch_size=self.batch_size))):
            batch_observations = []
            for agent_idx, sample in enumerate(batch_data):
                # convert txt to message
                response = sample["output"]
                message = Message(text=response)
                #print("message", message)

                # get agent
                agent = self.model_copies[agent_idx]
                # observe message
                observation = agent.observe(message)
                #print("observations", observation)
                batch_observations.append(observation)
            #print("batch observations", batch_observations)
            batch = self.model.batchify(batch_observations)

            # batch inference
            output = self.model.score(batch)
            # get score
            scores = softmax(output.cpu(), axis=1)
            pred_classes = [class_list[i] for i in np.argmax(scores, axis=1)]
            # safety scores
            safety_scores.extend([s[0] for k, s in enumerate(scores)])
            # get metadata
            for k, s in enumerate(scores):
                str_scores = [float_s for float_s in list(s)]
                meta_scores = [{i: j.item()} for (i, j) in zip(class_list, str_scores)]
                meta_data.append({'pred_class': pred_classes[k], 'class_scores':meta_scores})

            # clear history after batch
            self.clear_history()
        return safety_scores, meta_data

