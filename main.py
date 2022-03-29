from util.loader import Loader
from util.logger import setup_logger

from inp_out.io import IO
from data_pipeline import Data_pipeline
from pipeline import give_answer_to_question_pipeline, generate_questions_pipeline, summarize_text_pipeline
from sentence_transformers import SentenceTransformer, util

log = setup_logger(__name__)

def main():
    ## SETUP ##
    # Setup objects with their config
    data = Data_pipeline()
    io = IO()
    embedding_creator = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')

    ## INPUT ##
    # TODO: Human input
    questions = io.get_all_questions()
    answers = io.get_all_answers()
    answer_embeddings = embedding_creator.encode(answers)
    context_list = io.get_paragraphs()

    try:
        with Loader("Answering Questions..."):
            for idx, question in enumerate(questions):
                log.info(f"Answering question '{question}' ({idx}/{len(questions)})...")
                result_objects = give_answer_to_question_pipeline(question, context_list)
                result_objects["true_answer"] = answers[idx]
                result_objects["true_answer_similarity_score"] = util.dot_score(embedding_creator.encode(result_objects["translated_answer"]), answer_embeddings[idx])
                data.add_qa_results(result_objects)
                io.print_results(result_objects)
    except KeyboardInterrupt:
        df = data.to_df()
        df.to_csv("./first_interrupted_run.csv")
        return
    df = data.to_df()
    df.to_csv("./first_run.csv")
    #generate_questions_pipeline()
    # summarize_text_pipeline()
    
if __name__=="__main__":
    main()