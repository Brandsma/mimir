from dynaconf import settings
from transformers import pipeline

from context.context import Context
from data_pipeline import Data_pipeline
from inp_out.io import IO
from processing.processing import Processing
from question_answering.question_answering import QuestionAnswering
from question_generation import generate_question
from questiongenerator import QuestionGenerator, print_qa
from summarize_text import one_line_summary, summarize_text
from util.logger import setup_logger

log = setup_logger(__name__)


def give_answer_to_question_pipeline(question, context_list):
    log.info("Starting 'give answer to question' pipeline...")
    ## SETUP ##
    processing = Processing()
    context = Context()
    question_answering = QuestionAnswering()
    ##

    ## INPUT ##
    question = io.get_question()
    context_list = io.get_context_list()

    ## PREPROCESSING ##
    # TODO: automatically detect language
    translated_question = processing.translate(
        question, from_lang=settings["input_language"], to_lang='en')
    translated_context_list = processing.translate(
        context_list, from_lang=settings["input_language"], to_lang='en')

    #data.add("translation score", translation_score)
    ## CONTEXT ##
    # TODO: Maybe more types of scores?
    # Like what?
    # - How good was this context for the question answering? --> link answer score to context
    # -
    most_relevant_context_score_pairs = context.retrieve_context(
        translated_question,
        translated_context_list,
        n=settings["number_of_retrieved_contexts"])

    ## QUESTION ANSWERING ##
    # TODO: Score for certainty about answer
    answer = question_answering.retrieve_answer(
        question, [x[0] for x in most_relevant_context_score_pairs],
        num_generated_answers=settings["num_generated_answers"],
        num_correct_answers_required=settings["num_correct_answers_required"])
    # TODO: Score for the context and answer pair with the qa evaluator pair

    ## POSTPROCESSING ##
    translated_answer = processing.translate(
        answer, from_lang='en', to_lang=settings["input_language"])

    qa_evaluator = pipeline("text-classification",
                            model="iarfmoose/bert-base-cased-qa-evaluator")

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
        qa_evaluator(f"[CLS] {translated_question} [SEP] {answer} [SEP]"),
        "most_relevant_contexts":
        [x[0] for x in most_relevant_context_score_pairs],
        "context_scores": [x[1] for x in most_relevant_context_score_pairs],
    }

    return result_objects


def generate_questions_pipeline():
    ## SETUP ##
    # Setup objects with their config
    io = IO()
    data = Data_pipeline()
    question_answering = QuestionAnswering()
    ##

    _, context_list = io.get_raw_question_and_contexts()

    qg = QuestionGenerator()

    qa_list = qg.generate(' '.join(context_list),
                          num_questions=10,
                          answer_style="sentences",
                          use_evaluator=True)
    print_qa(qa_list, show_answers=True)


def summarize_text_pipeline():
    io = IO()
    data = Data_pipeline()
    paragraphs = io.get_paragraphs()

    for idx, text in enumerate(paragraphs):
        if idx == 2:
            break
        ## Summarize text
        summary = summarize_text(text)

        ## Single line summary
        one_line = one_line_summary(text)

        ## Evaluation
        ## ??

        results_object = {
            "raw_text": text,
            "summary": summary,
            "one_line": one_line
        }
        data.add_summarize_results(results_object)
        io.print_results(results_object)
    print(data.summarize_to_df())


if __name__ == "__main__":
    generate_questions_pipeline()
