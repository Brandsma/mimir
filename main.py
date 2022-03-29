from util.logger import setup_logger
from pipeline import give_answer_to_question_pipeline, generate_questions_pipeline

log = setup_logger(__name__)

def main():
    generate_questions_pipeline()

if __name__=="__main__":
    main()
