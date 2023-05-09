# from https://github.com/danieldeutsch/qaeval/tree/master and https://github.com/salesforce/QAFactEval/tree/master
from typing import Any, Dict, List, Tuple, Union
import hashlib
import logging
import math
from tqdm import tqdm
from transformers.data.metrics.squad_metrics import compute_f1
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from .utils import LERCQuipScorer, MetaScorer, ExactMatchScorer, F1Scorer, IsAnsweredScorer
from .qa_model import QuestionAnsweringModel
from .answer_selection import NP_CHUNKS_STRATEGY, AnswerSelector

MetricsDict = Dict[str, float]
SummaryType = Union[str, List[str]]
logger = logging.getLogger(__name__)


def get_filter(qa_summ, answers_ref):
    answerable = []
    a_orig = []
    for qa_summ_ in qa_summ:
        answerability = (qa_summ_[1] > qa_summ_[2])
        answerable.append(answerability)
        a_orig.append(qa_summ_[0])

    f1s = [compute_f1(answer, prediction) for \
        answer, prediction in zip(answers_ref, a_orig)]
    bool_f1 = [x > 0.60 for x in f1s]
    bool_total = [x and y for x, y in zip(bool_f1, answerable)]
    return bool_total


class QAFactEval:
    def __init__(
            self,
            lerc_quip_path: str,
            use_lerc_quip: bool,
            lerc_batch_size: int,
            cuda_device: int,
            answering_model_dir: str,
            answer_selection_strategy: str = NP_CHUNKS_STRATEGY,
            generation_batch_size: int = 8,
            answering_batch_size: int = 8,
            verbose: bool = False):
        try:
            import spacy
            spacy.load("en_core_web_sm")
        except:
            import spacy.cli
            spacy.cli.download("en_core_web_sm")

        self.answer_selector = AnswerSelector(answer_selection_strategy)
        self.question_tokenizer = AutoTokenizer.from_pretrained('Salesforce/mixqg-base')
        self.question_generator = AutoModelForSeq2SeqLM.from_pretrained('Salesforce/mixqg-base')
        if cuda_device >= 0:
            self.question_generator.to(cuda_device)

        self.cuda_device = cuda_device
        self.question_answerer = QuestionAnsweringModel(
            answering_model_dir,
            cuda_device=cuda_device,
            batch_size=answering_batch_size,
            silent=not verbose,
        )
        self.verbose = verbose
        self.generation_batch_size = generation_batch_size

        scorers = [IsAnsweredScorer(), ExactMatchScorer(), F1Scorer()]
        self.scorer = MetaScorer(scorers)
        if use_lerc_quip:
            lerc_quip = LERCQuipScorer(lerc_quip_path=lerc_quip_path, \
                cuda_device=cuda_device, batch_size=lerc_batch_size)
            self.scorer.scorers.append(lerc_quip)
        

    def _get_question_id(
        self, instance_index: int, reference_index: int, start: int, end: int
    ) -> str:
        m = hashlib.md5()
        m.update(str(instance_index).encode())
        m.update(str(reference_index).encode())
        m.update(str(start).encode())
        m.update(str(end).encode())
        return m.hexdigest()

    def _get_empty_summary_mask(
        self, summaries: List[str], references_list: List[List[str]]
    ) -> Tuple[List[str], List[List[str]], List[bool]]:
        # This will identify any summaries that have empty text. The output will be the list of non-empty summaries
        # with their corresponding references plus a list of booleans that is parallel will the input `summaries`
        # which mark whether or not they are empty
        is_empty_list = []
        non_empty_summaries = []
        non_empty_references_list = []

        for summary, references in zip(summaries, references_list):
            if len(summary.strip()) > 0:
                is_empty_list.append(False)
                non_empty_summaries.append(summary)
                non_empty_references_list.append(references)
            else:
                is_empty_list.append(True)
        return non_empty_summaries, non_empty_references_list, is_empty_list

    def _flatten_summaries(self, summaries: List[SummaryType]) -> List[str]:
        flat_summaries = []
        for summary in summaries:
            if isinstance(summary, list):
                summary = " ".join(summary)
            flat_summaries.append(summary)
        return flat_summaries

    def _flatten_references_list(
        self, references_list: List[List[SummaryType]]
    ) -> List[List[str]]:
        # Flattens all of the summaries so they are `str` instead of `List[str]`
        flat_references_list = []
        for references in references_list:
            flat_references_list.append([])
            for reference in references:
                if isinstance(reference, list):
                    reference = " ".join(reference)
                flat_references_list[-1].append(reference)
        return flat_references_list
    
    def generate_all(self, input_jsons):
        final_inputs = []
        outputs = []
        num_batches = int(math.ceil(len(input_jsons) / self.generation_batch_size))

        generator = range(0, len(input_jsons), self.generation_batch_size)
        # if not self.silent:
            # generator = tqdm(generator, total=num_batches, desc='Generating questions')

        def format_inputs(x):
            answer = x[0][x[1]:x[2]]
            return f"{answer} \\n {x[0]}"
        
        for i in generator:
            batch = input_jsons[i:i + self.generation_batch_size]
            batch = [format_inputs(x) for x in batch]
            batch = self.question_tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
            if self.cuda_device >= 0:
                batch.to(self.cuda_device)
            gen = self.question_generator.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                num_beams=4,
                max_length=32
            )
            gen_dec = self.question_tokenizer.batch_decode(gen, skip_special_tokens=True)
            outputs.extend(gen_dec)

        return outputs

    def _generate_qa_pairs(
        self, references_list: List[List[str]]
    ) -> List[List[List[Dict[str, Any]]]]:
        # This will generate the question-answer pairs for each reference. Since references may be repeated,
        # we first deduplicate the references to minimize the expensive work.
        #
        # `reference_to_index` keeps track of where each of the unique references are in `distinct_references_list`
        reference_to_index = {}
        distinct_references_list = []

        # Maps from (i, j) to the index in the `distinct_references_list`
        mapping = {}
        for i, references in enumerate(references_list):
            for j, reference in enumerate(references):
                if reference not in reference_to_index:
                    reference_to_index[reference] = len(distinct_references_list)
                    distinct_references_list.append(reference)
                mapping[(i, j)] = reference_to_index[reference]

        # Select the answers
        logger.info(
            f"Selecting answers from {len(distinct_references_list)} distinct summaries"
        )
        answers_list = self.answer_selector.select_all(distinct_references_list)
        num_answers = sum(len(answers) for answers in answers_list)
        logger.info(f"Selected {num_answers} answers in total")

        # Generate the questions
        generation_inputs = []
        for reference, answers in zip(distinct_references_list, answers_list):
            for answer in answers:
                sentence = reference[answer.sent_start : answer.sent_end]
                start = answer.start - answer.sent_start
                end = answer.end - answer.sent_start
                generation_inputs.append((sentence, start, end))

        logger.info(f"Generating questions for {len(generation_inputs)} answers")
        question_list = self.generate_all(generation_inputs)
        logger.info("Finished generating questions")

        # Remap the questions to align with the answers
        index = 0
        remapped_questions = []
        for i, answers in enumerate(answers_list):
            remapped_questions.append([])
            for _ in answers:
                remapped_questions[-1].append(question_list[index])
                index += 1
            assert len(remapped_questions[i]) == len(answers_list[i])
        assert len(remapped_questions) == len(answers_list)

        # Remap output to align with the inputs
        # qa_pairs_lists[summary_index][reference_index] = [(q, a)]
        qa_pairs_lists = []
        for i, references in enumerate(references_list):
            qa_pairs_lists.append([])
            for j, reference in enumerate(references):
                index = mapping[(i, j)]
                qa_pairs_lists[-1].append([])
                for question, answer in zip(
                    remapped_questions[index], answers_list[index]
                ):
                    question_id = self._get_question_id(i, j, answer.start, answer.end)
                    qa_pairs_lists[-1][-1].append(
                        {
                            "question_id": question_id,
                            "question": question,
                            "answer": answer.text,
                            "sent_start": answer.sent_start,
                            "sent_end": answer.sent_end,
                            "answer_start": answer.start,
                            "answer_end": answer.end,
                        }
                    )
        return qa_pairs_lists
    
    def _answer_questions(
        self, summaries: List[str], qa_pairs_lists: List[List[List[Dict[str, Any]]]]
    ) -> List[List[List[Dict[str, Any]]]]:
        # Answers all of the questions. Some of the (question, context) pairs may be duplicates, for instance because
        # of jackknifing. It will be a lot faster to deduplicate them first.
        #
        # `qa_inputs` will contain the unique inputs, `context_to_input_index` maps from the (question, context) pair
        # to its index in `qa_inputs`, and `mapping` will map from the i-th summary, j-th reference, and k-th question
        # to the index of the corresponding data in `qa_inputs`
        qa_inputs = []
        context_to_input_index = {}
        mapping = {}

        for i, (summary, qa_pairs_list) in enumerate(zip(summaries, qa_pairs_lists)):
            for j, qa_pairs in enumerate(qa_pairs_list):
                for k, qa in enumerate(qa_pairs):
                    question = qa["question"]
                    key = (question, summary)
                    if key not in context_to_input_index:
                        context_to_input_index[key] = len(qa_inputs)
                        qa_inputs.append(key)
                    mapping[(i, j, k)] = context_to_input_index[key]

        logger.info(f"Answering {len(qa_inputs)} distinct (question, context) pairs")
        predictions = self.question_answerer.answer_all(qa_inputs, return_offsets=True)
        logger.info("Finished answering questions")

        # Remap from the distinct answers back to the original QA lists
        predictions_lists = []
        for i, (summary, qa_pairs_list) in enumerate(zip(summaries, qa_pairs_lists)):
            predictions_lists.append([])
            for j, qa_pairs in enumerate(qa_pairs_list):
                predictions_lists[-1].append([])
                for k, qa in enumerate(qa_pairs):
                    index = mapping[(i, j, k)]
                    prediction, probability, null_probability, offsets = predictions[
                        index
                    ]
                    predictions_lists[-1][-1].append(
                        {
                            "prediction_id": self._get_prediction_id(index),
                            "prediction": prediction,
                            "probability": probability,
                            "null_probability": null_probability,
                            "start": offsets[0],
                            "end": offsets[1],
                        }
                    )
        return predictions_lists

    def _get_prediction_id(self, prediction_index: int):
        m = hashlib.md5()
        m.update(str(prediction_index).encode())
        return m.hexdigest()

    def _combine_outputs(
        self,
        metrics_list: List[MetricsDict],
        qa_pairs_lists: List[List[List[Dict[str, Any]]]],
        predictions_lists: List[List[List[Dict[str, Any]]]],
        scores_lists: List[List[List[Dict[str, float]]]],
    ) -> List[List[List[Dict[str, Any]]]]:
        # This method will combine the metrics and QA pair metadata together into a tuple so they can
        # both be returned together
        combined = []
        for metrics, qa_pairs_list, predictions_list, scores_list in zip(
            metrics_list, qa_pairs_lists, predictions_lists, scores_lists
        ):
            # This is for 1 (summary, reference) pair
            combined.append((metrics, []))
            for qa_pairs, predictions, scores in zip(
                qa_pairs_list, predictions_list, scores_list
            ):
                # This is for 1 reference
                combined[-1][1].append([])
                for qa, prediction, score in zip(qa_pairs, predictions, scores):
                    prediction = dict(**prediction)
                    for key in self.scorer.keys():
                        prediction[key] = score[key]
                    combined[-1][1][-1].append(
                        {"question": qa, "prediction": prediction}
                    )
        return combined
    
    def _insert_empty_outputs(
        self,
        metrics_list: List[MetricsDict],
        is_empty_list: List[bool],
        include_qa_list: bool,
    ) -> List[Any]:
        full_metrics_list = []
        index = 0
        for is_empty in is_empty_list:
            if is_empty:
                empty_metrics = {"qa-eval": self.scorer.default_scores()}
                if include_qa_list:
                    full_metrics_list.append((empty_metrics, []))
                else:
                    full_metrics_list.append(empty_metrics)
            else:
                full_metrics_list.append(metrics_list[index])
                index += 1
        return full_metrics_list
    
    def _score_predictions(
        self,
        summaries: List[str],
        qa_pairs_lists: List[List[List[Dict[str, Any]]]],
        predictions_lists: List[List[List[Dict[str, Any]]]],
    ) -> Tuple[List[MetricsDict], List[List[List[Dict[str, float]]]]]:
        logger.info("Scoring predictions")
        metrics_list = []
        scores_list = []

        generator = tqdm(
            zip(summaries, qa_pairs_lists, predictions_lists),
            total=len(summaries),
            disable=not self.verbose,
        )
        for summary, qa_pairs_list, predictions_list in generator:
            # This is for 1 (summary, references) pair
            input_questions_list = []
            input_answers_list = []
            input_predictions_list = []
            input_probabilities_list = []
            input_null_probabilities_list = []
            for qa_pairs, predictions in zip(qa_pairs_list, predictions_list):
                # This is the set of QA pairs for 1 reference
                input_questions_list.append([])
                input_answers_list.append([])
                input_predictions_list.append([])
                input_probabilities_list.append([])
                input_null_probabilities_list.append([])
                for qa, prediction in zip(qa_pairs, predictions):
                    input_questions_list[-1].append(qa["question"])
                    input_answers_list[-1].append(qa["answer"])
                    input_predictions_list[-1].append(prediction["prediction"])
                    input_probabilities_list[-1].append(prediction["probability"])
                    input_null_probabilities_list[-1].append(
                        prediction["null_probability"]
                    )

            metrics, scores = self.scorer.score_multi_ref(
                summary,
                input_questions_list,
                input_answers_list,
                input_predictions_list,
                input_probabilities_list,
                input_null_probabilities_list,
            )
            metrics = {"qa-eval": metrics}
            metrics_list.append(metrics)
            scores_list.append(scores)

        logger.info("Finished scoring predictions")
        return metrics_list, scores_list
    
    def score_batch_qafacteval(
        self,
        source: List[SummaryType],
        summaries: List[List[SummaryType]],
        qa_pairs_precomputed: List = None,
        predictions_lists: List = None,
        return_qa_pairs: bool = False,
            ) -> List[List[MetricsDict]]:
        source = self._flatten_summaries(source)
        summaries = self._flatten_references_list(summaries)

        # Remove any input source that are empty. They mess up the processing otherwise
        (
            source,
            summaries,
            is_empty_list,
        ) = self._get_empty_summary_mask(source, summaries)


        if qa_pairs_precomputed:
            qa_pairs_lists = qa_pairs_precomputed
        else:
            qa_pairs_lists = self._generate_qa_pairs(summaries)

        # question_consistency
        # TODO: only uses one summary here
        summaries_cons = [x[0] for x in summaries]
        predictions_lists_consistency = self._answer_questions(summaries_cons, qa_pairs_lists)

        qa_pairs_lists_cons = []
        for x, cur_qa_pair in zip(predictions_lists_consistency, qa_pairs_lists):
            qa_summ_new = [[x["prediction"], x["probability"], \
                x["null_probability"]] for x in x[0]]
            answers_ref = [x["answer"] for x in cur_qa_pair[0]]

            bool_total = get_filter(qa_summ_new, answers_ref)

            cur_qa_pair_keep = [x for count, x in enumerate(cur_qa_pair[0]) if bool_total[count]]
            if not cur_qa_pair_keep:
                cur_qa_pair_keep = []
            qa_pairs_lists_cons.append([cur_qa_pair_keep])

        if predictions_lists:
            predictions_lists = predictions_lists
        else:
            predictions_lists = self._answer_questions(source, qa_pairs_lists_cons)

        metrics_list, scores_lists = self._score_predictions(
            source, qa_pairs_lists_cons, predictions_lists
        )

        if return_qa_pairs:
            output = self._combine_outputs(
                metrics_list, qa_pairs_lists_cons, predictions_lists, scores_lists
            )
        else:
            output = metrics_list

        output = self._insert_empty_outputs(output, is_empty_list, return_qa_pairs)
        output_final = []
        # Add consistency info for analysis
        for out, qa_pairs_list, predictions_cons in \
                zip(output, qa_pairs_lists, predictions_lists_consistency):
            output_final.append((out[0], out[1], qa_pairs_list, predictions_cons))
        return output_final