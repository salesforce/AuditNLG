'''
 * Copyright (c) 2023, Salesforce, Inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

from summac.model_summac import SummaCConv
from ..utils import retrieve_global_knowledge, get_instruction

class SummacEvaluator:
    def __init__(self, device='cuda:0'):
        """ Set up evaluator for factual consistency detection """
        self.model_conv = SummaCConv(models=["vitc"], bins='percentile', granularity="sentence", \
            nli_labels="e", device=device, start_file="default", agg="mean")

    def evaluate(self, data, global_knowledge):
        sources = []
        generateds = []
        for sample in data:
            source = '\n'.join([get_instruction(sample),
                                sample['knowledge'],
                                retrieve_global_knowledge(sample['output'], global_knowledge)])
            sources.append(source)
            generateds.append(sample['output'])
        scores = self.model_conv.score(sources, generateds)["scores"]
        return scores
