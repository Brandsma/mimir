from matplotlib.style import context
from inp_out.io import IO
from processing.processing import Processing
from context.context import Context
from question_answering.question_answering import QuestionAnswering
from util.logger import setup_logger
from dynaconf import settings
from question_generation import generate_question
from transformers import pipeline
from summarize_text import summarize_text, one_line_summary
from questiongenerator import QuestionGenerator, print_qa
from data_pipeline import Data_pipeline

log = setup_logger(__name__)

def give_answer_to_question_pipeline():
    log.info("Starting 'give answer to question' pipeline...")
    ## SETUP ##
    # Setup objects with their config
    io = IO()
    data = Data_pipeline()
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

    #data.add("translation score", translation_score)
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
    data.add_qa_results(result_objects)
    io.print_results(result_objects)

def generate_questions_pipeline():
    ## SETUP ##
    # Setup objects with their config
    io = IO()
    data = Data_pipeline()
    question_answering = QuestionAnswering()
    ##

    _, context_list = io.get_raw_question_and_contexts()

    qg = QuestionGenerator()

    qa_list = qg.generate(
        ' '.join(context_list[50:100]),
        num_questions=10,
        answer_style="sentences",
        use_evaluator=True
    )
    print_qa(qa_list, show_answers=True)



    # for chosen_context in context_list[15:20]:
    #     ## QUESTION GENERATION ##
    #     generated_question = generate_question(chosen_context)

    #     ## QUESTION ANSWERING ##
    #     answer = question_answering.retrieve_answer(generated_question, chosen_context)

    #     ## QUESTION/ANSWER EVALUATION ##
    #     qa_evaluator = pipeline(model = "iarfmoose/bert-base-cased-qa-evaluator")

    #     results_object = {
    #         "context": chosen_context,
    #         "generated_question": generated_question,
    #         "answer": answer,
    #         "qa_score": qa_evaluator(f"[CLS] {generated_question} [SEP] {answer} [SEP]")
    #     }
    #     data.add_q_gen_results(results_object)
    #     io.print_results(results_object)

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

if __name__=="__main__":
    generate_questions_pipeline()