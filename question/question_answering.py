from typing import Optional, List
from transformers import pipeline
from util.logger import setup_logger
from util.nlp import split_sentences, determine_most_frequent

from dynaconf import settings
from transformers import pipeline

from context.context import ContextRetrieval
from processing.translator import Translator


log = setup_logger(__name__)


class QuestionAnswering:
    """
    This class takes care of answering questions. Should be initialized with
    context as args, so the embeddings have to be computed only once.
    """

    def __init__(self, context_list=None):
        self.answerer = pipeline("text2text-generation", model="allenai/unifiedqa-t5-base")
        self.paraphraser = pipeline("text2text-generation", model="tuner007/pegasus_paraphrase")
        self.from_to_translator = Translator(from_lang=settings["input_language"], to_lang="en")
        self.to_from_translator = Translator(from_lang="en", to_lang=settings["input_language"])
        self.qa_evaluator = pipeline("text-classification", model="iarfmoose/bert-base-cased-qa-evaluator")
        self.context = ContextRetrieval()
        self.embedding_state = False  # this bool keeps track wether embeddings have been created in ContextRetrieval()
        if isinstance(context_list, list):
            self._embed(context_list)

    def _embed(self, context_list):
        translated_context_list = self.from_to_translator.translate(context_list)
        self.context.embed(translated_context_list)
        self.embedding_state = True

    def answer_question(self, question, context_list=None):
        # check if embeddings exist yet, create them if not. For backward compatibility only
        if self.embedding_state == False:
            if context_list == None:
                log.error("No embeddings exist yet but no context_list was provided.")
            self._embed(context_list)

        ## TRANSLATE ##
        translated_question = self.from_to_translator.translate(question)

        ## SEMANTIC SEARCH ##
        most_relevant_context_score_pairs = self.context.retrieve_context(
            translated_question,
            n=settings["number_of_retrieved_contexts"],
            method="euclid")

        ## QUESTION ANSWERING ##
        answer = self.retrieve_answer(
            question, [x[0] for x in most_relevant_context_score_pairs],
            num_generated_answers=settings["num_generated_answers"],
            num_correct_answers_required=settings["num_correct_answers_required"])

        ## TRANSLATE BACK ##
        translated_answer = self.to_from_translator.translate(answer)

        ## OUTPUT ##
        result_objects = {
            "translated_answer":
            translated_answer,
            "translated_question":
            translated_question,
            "question":
            question,
            "answer":
            answer,
            "qa_pair_score":
            self.qa_evaluator(f"[CLS] {translated_question} [SEP] {answer} [SEP]"),
            "most_relevant_contexts":
            [x[0] for x in most_relevant_context_score_pairs],
            "context_scores": [x[1] for x in most_relevant_context_score_pairs],
        }

        return result_objects


    def _split_context(self, context):
        split_context = []
        if context != None:
            for text in context:
                split_context += split_sentences(text)
        return split_context

    def _generate_paraphrases(self, question, num_generated_answers):
        log.debug("Generating paraphrases of question...")
        paraphrased_questions = self.paraphraser(question, num_return_sequences=num_generated_answers)
        return paraphrased_questions

    def _format_query(self, question: str, split_context: Optional[List[str]] = None, choices: Optional[List[str]] = None):
        query = ""
        # question
        query += question
        query += ' \n'

        # choices if available
        if choices != None:
            for idx, choice in enumerate(choices):
                query += f" ({chr(97 + idx)}) {choice}"
                query += ' \n '

        # context if available
        if split_context != None:
            for sentence in split_context:
                query += sentence
        return query

    def _generate_answers(self, paraphrased_questions, split_context, choices):
        log.debug("Attempting to answer questions...")
        answers = []
        for paraphrased_question in paraphrased_questions:
            query = self._format_query(paraphrased_question['generated_text'], split_context, choices)
            answers.append(self.answerer(query)[0]['generated_text'])
        return answers

    def retrieve_answer(self, question: str, context: Optional[List[str]] = None, choices: Optional[List[str]] = None, num_generated_answers=5, num_correct_answers_required=3):
        log.debug(f"Starting to answer the question: {question}")

        # Prepare input data by splitting the context snippet in sentences
        split_context = self._split_context(context)

        # Generate paraphrases of the question
        paraphrased_questions = self._generate_paraphrases(question, num_generated_answers)

        # Generate potential answers
        answers = self._generate_answers(paraphrased_questions, split_context, choices)

        # Return the most common answer if we get it right often enough, translate it back to original language
        most_common_answer, answer_count = determine_most_frequent(answers)

        if answer_count >= num_correct_answers_required:
                return most_common_answer
        else:
            return "No common answer found"
