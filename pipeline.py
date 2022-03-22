from inp_out.io import IO
from processing.processing import Processing
from context.context import Context
from question_answering.question_answering import QuestionAnswering
from util.logger import setup_logger
from dynaconf import settings

log = setup_logger(__name__)

def give_answer_to_question_pipeline():
    log.info("Starting 'give answer to question' pipeline...")
    ## SETUP ##
    # Setup objects with their config
    io = IO()
    processing = Processing()
    context = Context()
    question_answering = QuestionAnswering()
    ##

    ## INPUT ##
    question, context_list = io.get_raw_question_and_contexts()

    ## PREPROCESSING ##
    # TODO: automatically detect language
    translated_question = processing.translate(question, from_lang=settings["input_language"], to_lang='en')
    translated_context_list = processing.translate(context_list, from_lang=settings["input_language"], to_lang='en')

    ## CONTEXT ##
    most_relevant_context_score_pairs = context.retrieve_context(translated_question, translated_context_list, n=settings["number_of_retrieved_contexts"])

    ## QUESTION ANSWERING ##
    answer = question_answering.retrieve_answer(question, [x[0] for x in most_relevant_context_score_pairs])

    ## POSTPROCESSING ##
    translated_answer = processing.translate(
        answer, from_lang='en', to_lang=settings["input_language"])

    ## OUTPUT ##
    result_objects = {
        "translated_answer": translated_answer, 
        "translated_question": translated_question, 
        "question": question,
        #"context_list": context_list,
        #"translated_context_list": translated_context_list, 
        "answer": answer, 
        "most_relevant_contexts": [x[0] for x in most_relevant_context_score_pairs], 
        "context_scores": [x[1] for x in most_relevant_context_score_pairs],
        }
    io.print_results(result_objects)


def generate_questions_pipeline():
    pass
    #
