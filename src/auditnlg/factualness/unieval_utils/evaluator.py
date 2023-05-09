# Borrowed from https://github.com/maszhongming/UniEval/blob/main/metric/evaluator.py

from .scorer import UniEvaluator
from ..utils import retrieve_global_knowledge, get_instruction
from nltk import sent_tokenize

class FactEvaluator:
    def __init__(self, max_length=1024, device='cuda:0', cache_dir=None):
        """ Set up evaluator for factual consistency detection """
        self.scorer = UniEvaluator(model_name_or_path='MingZhong/unieval-fact',
                                   max_length=max_length,
                                   device=device, cache_dir=cache_dir)
        self.task = 'fact'
        self.dim = 'consistency'

    def evaluate(self, data, global_knowledge):
        """
            Get the factual consistency score (only 1 dimension for this task)

            print_result: whether to print the average factual score on the screen
        """
        n_data = len(data)

        # Calculate average sentence-level scores for facutal consistency
        input_list = []
        n_sents = [] # the number of sentences in the claim
        for sample in data:
            source = '\n'.join([get_instruction(sample),
                                sample['knowledge'],
                                retrieve_global_knowledge(sample['output'], global_knowledge)])
            system_outputs = sent_tokenize(sample['output'])
            n_sents.append(len(system_outputs))
            for system_output in system_outputs:
                input_list.append('question: Is this claim consistent with the document? </s> claim: ' + system_output + ' </s> document: ' + source)
        sent_score = self.scorer.score(input_list)

        # Get average score for each sample
        start_idx = 0
        scores = []
        for cur_n_sent in n_sents:
            scores.append(sum(sent_score[start_idx: start_idx + cur_n_sent]) / cur_n_sent)
            start_idx += cur_n_sent

        return scores
