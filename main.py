from sentence_transformers import SentenceTransformer, util

from data_pipeline import Data_pipeline
from inp_out.io import IO
from pipeline import (generate_questions_pipeline,
                      give_answer_to_question_pipeline,
                      summarize_text_pipeline)
from util.loader import Loader
from util.logger import setup_logger

log = setup_logger(__name__)


def evaluate_answering_pipeline():
    ## SETUP ##
    # Setup objects with their config
    data = Data_pipeline()
    io = IO()
    embedding_creator = SentenceTransformer(
        'sentence-transformers/multi-qa-MiniLM-L6-cos-v1')

    ## INPUT ##
    # TODO: Human input
    questions = io.get_all_questions()
    answers = io.get_all_answers()
    paragraphs_related_to_answers = io.get_all_paragraphs_for_answers()
    answer_embeddings = embedding_creator.encode(answers)
    # TODO: Challenge, better clean up for the paragraphs
    context_list = io.get_paragraphs()

    try:
        with Loader("Answering Questions..."):
            for idx, question in enumerate(questions):
                log.info(
                    f"Answering question '{question}' ({idx}/{len(questions)})..."
                )
                result_objects = give_answer_to_question_pipeline(
                    question, context_list)

                result_objects["true_answer"] = answers[idx]

                result_objects[
                    "true_answer_similarity_score"] = util.dot_score(
                        embedding_creator.encode(
                            result_objects["translated_answer"]),
                        answer_embeddings[idx])

                result_objects["true_context_similarity_score"] = util.dot_score(embedding_creator.encode(
                    paragraphs_related_to_answers[idx]), embedding_creator.encode(result_objects["most_relevant_contexts"][0]))
                
                data.add_qa_results(result_objects)
                io.print_results(result_objects)
    except KeyboardInterrupt:
        df = data.to_df()
        df.to_csv("./first_interrupted_run.csv")
        return
    df = data.to_df()
    df.to_csv("./first_run.csv")
    # generate_questions_pipeline()
    # summarize_text_pipeline()


if __name__ == "__main__":
    main()
