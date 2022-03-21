from typing import Optional, List
from setuptools import setup
from transformers import pipeline
from collections import Counter
from util.logger import setup_logger
from util.loader import Loader
from nlp_util import split_sentences, determine_most_frequent

log = setup_logger(__name__)


class QuestionAnswering:

    def __init__(self):
        pass

    def split_context(self, context):
        split_context = []
        if context != None:
            for text in context:
                split_context += split_sentences(text)
        return split_context

    def generate_paraphrases(self, question, num_generated_answers):
        log.info("Generating paraphrases of question...")
        paraphraser = pipeline("text2text-generation", model="tuner007/pegasus_paraphrase")
        paraphrased_questions = paraphraser(question, num_return_sequences=num_generated_answers)
        return paraphrased_questions

    def format_query(self, question: str, split_context: Optional[List[str]] = None, choices: Optional[List[str]] = None):
        # One of the possible options:
        # <QUESTION> \n (a) <CHOICE_A> (b) <CHOICE_B> ... \n <CONTEXT> 
        
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

    def generate_answer(self, paraphrased_questions, split_context, choices):
        log.info("Attempting to answer questions...")
        answers = []
        for paraphrased_question in paraphrased_questions:
            generator = pipeline("text2text-generation",
                                model="allenai/unifiedqa-t5-base")
            query = self.format_query(
                paraphrased_question['generated_text'], split_context, choices)
            answers.append(generator(query)[0]['generated_text'])
        return answers

    def retrieve_answer(self, question: str, context: Optional[List[str]] = None, choices: Optional[List[str]] = None, num_generated_answers=5, num_correct_answers_required=3):

        log.info(f"Starting to answer the question: {question}")
        
        # Prepare input data
        split_context = self.split_context(context)

        # Generate paraphrases of the question
        paraphrased_questions = self.generate_paraphrases(question, num_generated_answers)

        # Generate an answer using AllenAI
        answers = self.generate_answer(paraphrased_questions, split_context, choices)

        # Return the most common answer if we get it right often enough, translate it back to original language
        most_common_answer, answer_count = determine_most_frequent(answers)

        if answer_count >= num_correct_answers_required:
                return most_common_answer
        else:
            return "No common answer found"        
        