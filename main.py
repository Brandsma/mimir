import data
from answer_retrieval import retrieve_answer
from context_retrieval import retrieve_context
from logger import setup_logger

log = setup_logger(__name__)

def answer_question_pipeline(question):
    # Input (raw) source text
    source = data.retrieve_paragraphs()
    contexts = [x for x in source['train']['text']]

    # retrieve context from source text based on question
    context = retrieve_context(question, contexts)
    print(context)
    exit()

    # Answer question with context
    answer = retrieve_answer(question, context, language='en', num_generated_answers=5, num_correct_answers_required=3)

    # Return answer
    print("")
    print(f"QUESTION: {question}\nANSWER: {answer}")

def main():
    # Input question 
    questions = data.retrieve_questions()
    print(f"trying the following example question: \n {questions['test'][0]}")
    question = questions['test'][0]['question']
    

    answer_question_pipeline(question)

    
if __name__=="__main__":
    main()