# Code borrowed from https://github.com/danieldeutsch/qaeval/tree/master
# This file was edited from the run_squad.py file in the experiment repository
import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    squad_convert_examples_to_features,
)
from transformers.data.processors.squad import SquadResult, SquadExample

from typing import Dict, List, Tuple, Union

from .utils import compute_predictions_logits_with_null, fix_answer_span, SpanFixError

Prediction = Union[
    Tuple[str, float, float],
    Tuple[str, float, float, Tuple[int, int]],
    Dict[str, Union[str, float, Tuple[int, int]]],
]


class QuestionAnsweringModel(object):
    def __init__(self,
                 model_dir: str,
                 cuda_device: int = 0,
                 batch_size: int = 8,
                 silent: bool = True) -> None:
        self.config = AutoConfig.from_pretrained(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, do_lower_case=True, use_fast=False)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_dir, config=self.config)
        # if "cpu" not in cuda_device and int(cuda_device) >= 0:
        if int(cuda_device) >= 0:
            self.model.to(cuda_device)

        self.model_type = 'electra'
        self.cuda_device = cuda_device
        self.batch_size = batch_size
        self.max_seq_length = 384
        self.doc_stride = 128
        self.silent = silent

    def _to_list(self, tensor):
        return tensor.detach().cpu().tolist()

    def _try_fixing_offsets(
        self,
        contexts: List[str],
        predictions: Dict[str, str],
        offsets_dict: Dict[str, Tuple[int, int]],
    ) -> Dict[str, Tuple[int, int]]:
        """
        Tries to fix the potentially noisy character offsets of the predictions in the `contexts`.
        The input and output end indices are exclusive.
        """
        new_offsets = {}

        for i, context in enumerate(contexts):
            index = str(i)

            prediction = predictions[index]
            pred_start, pred_end = offsets_dict[index]
            if context is None or prediction is None or pred_start is None or pred_end is None:
                new_offsets[index] = (pred_start, pred_end)
            else:
                span = context[pred_start:pred_end]
                if span != prediction:
                    try:
                        pred_start, pred_end = fix_answer_span(prediction, span, pred_start, pred_end)
                    except SpanFixError:
                        pass
                new_offsets[index] = (pred_start, pred_end)
        return new_offsets

    def answer(
        self,
        question: str,
        context: str,
        return_offsets: bool = False,
        try_fixing_offsets: bool = True,
        return_dict: bool = False,
    ) -> Prediction:
        """
        Returns a tuple of (prediction, probability, null_probability). If `return_offsets = True`, the tuple
        will include rough character offsets of where the prediction is in the context. Because the tokenizer that
        the QA model uses does not support returning the character offsets from the BERT tokenization, we cannot
        directly provide exactly where the answer came from. However, the offsets should be pretty close to the
        prediction, and the prediction should be a substring of the offsets (modulo whitespace). If
        `return_offsets` and `try_fixing_offsets` are `True`, we will try to fix the character offsets via
        an alignment. See below.

        The `SquadExample` class maintains a list of whitespace separated tokens `doc_tokens` and a mapping
        from the context string characters to the token indices `char_to_word_offset`. Whitespace
        is included in the previous token. The `squad_convert_example_to_features` method takes each of these
        tokens and breaks it into the subtokens with the transformers tokenizer, which are passed into the model.
        It also keeps a mapping from the subtokens to the `doc_tokens` called `tok_to_orig_index`. The QA model
        predicts a span in the subtokens. In the `_get_char_offsets` method, we use these data structures to map
        from the subtoken span to character offsets. However, we cannot separate subtokens, so they are merged together.
        See the below example

            context: " My name is  Dan!"
            doc_tokens: [My, name, is, Dan!]
            char_to_word_offset: [-1, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
            subtokens: [My, name, is, Dan, ##!]
            tok_to_orig_index: [0, 1, 2, 3, 3]

            prediction: "name is Dan"
            prediction subtokens: [name, is, Dan]
            prediction in doc_tokens: [name, is, Dan!]
            prediction in context: "name is  Dan!"

        The prediction includes the extra whitespace between "is" and "Dan" as well as the "!"

        If `try_fixing_offsets=True`, we will try to fix the character offsets to be correct based on an alignment
        algorithm. We use the `edlib` python package to create a character alignment between the actual prediction
        string and the span given by the original offsets. We then update the offsets based on the alignment. If
        this procedure fails, the original offsets will be returned.
        """
        return self.answer_all(
            [(question, context)], return_offsets=return_offsets,
            try_fixing_offsets=try_fixing_offsets, return_dicts=return_dict
        )[0]

    def answer_all(
        self,
        input_data: List[Tuple[str, str]],
        return_offsets: bool = False,
        try_fixing_offsets: bool = True,
        return_dicts: bool = False,
    ) -> List[Prediction]:
        # Convert all of the instances to squad examples
        examples = []
        
        for i, (question, context) in enumerate(input_data):
            examples.append(SquadExample(
                qas_id=str(i),
                question_text=question,
                context_text=context,
                answer_text=None,
                start_position_character=None,
                title=None,
                is_impossible=True,
                answers=[]
            ))
        
        features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=self.tokenizer,
            max_seq_length=self.max_seq_length,
            doc_stride=self.doc_stride,
            max_query_length=64,
            is_training=False,
            return_dataset="pt",
            threads=1,
            tqdm_enabled=not self.silent
        )

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.batch_size)

        self.model.eval()
        all_results = []
        generator = eval_dataloader
        if not self.silent:
            generator = tqdm(generator, desc='Evaluating')

        for batch in generator:
            if int(self.cuda_device) >= 0:
                batch = tuple(t.to(int(self.cuda_device)) for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                }

                feature_indices = batch[3]
                outputs = self.model(**inputs)

            for i, feature_index in enumerate(feature_indices):
                eval_feature = features[feature_index.item()]
                unique_id = int(eval_feature.unique_id)
                # output_ = []
                # for output in outputs:
                #     import pdb;pdb.set_trace()
                #     output_.append(self._to_list(output[i]))
                # # output_ = [self._to_list(output[i]) for output in outputs]
                # start_logits, end_logits = output_
                start_logits = outputs.start_logits
                end_logits = outputs.end_logits
                result = SquadResult(unique_id, start_logits, end_logits)

                all_results.append(result)

        model_predictions = compute_predictions_logits_with_null(
            self.tokenizer,
            examples,
            features,
            all_results,
            20,
            30,
            True,
            False,
            True,
            return_offsets=return_offsets
        )

        if return_offsets:
            predictions, prediction_probs, no_answer_probs, offsets = model_predictions
            if try_fixing_offsets:
                contexts = [context for _, context in input_data]
                offsets = self._try_fixing_offsets(contexts, predictions, offsets)
        else:
            predictions, prediction_probs, no_answer_probs = model_predictions

        results = []
        for i in range(len(input_data)):
            i = str(i)
            r = (predictions[i], prediction_probs[i], no_answer_probs[i])
            if return_dicts:
                r = {
                    'prediction': r[0],
                    'probability': r[1],
                    'null_probability': r[2],
                }

            if return_offsets:
                if return_dicts:
                    r['start'] = offsets[i][0]
                    r['end'] = offsets[i][1]
                else:
                    r = r + (offsets[i],)
            results.append(r)
        return results
