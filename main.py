from util.logger import setup_logger
from pipeline import give_answer_to_question_pipeline, generate_questions_pipeline, summarize_text_pipeline

log = setup_logger(__name__)

def main():
    #give_answer_to_question_pipeline()
    #generate_questions_pipeline()
    summarize_text_pipeline()
    
if __name__=="__main__":
    main()