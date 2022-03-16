import data
from answer_retrieval import retrieve_answer
from context_retrieval import retrieve_context, retrieve_context_FAISS
from logger import setup_logger

log = setup_logger(__name__)

def analyze_contexts(contexts):
    print("--- Contexts")
    print(contexts)
    print("--- tpyes")
    print([type(x) for x in contexts])
    print("--- lens")
    print([len(x) for x in contexts])

def answer_question_pipeline(question):
    # Input (raw) source text
    # Normal
    source = data.retrieve_paragraphs()
    contexts = [x for x in source['train']['text']]
    # FAISS
    #contexts = data.retrieve_paragraphs()
    #print(contexts)

    analyze_contexts(contexts)
    exit()

    # retrieve context from source text based on question
    context = retrieve_context(question, contexts, n=5)
    print(context)
    exit()
    #context = [x[0] for x in context]

    # Answer question with context
    answer = retrieve_answer(question, context, language='en', num_generated_answers=5, num_correct_answers_required=3)

    # Return answer
    print("")
    print(f"QUESTION: {question}\nANSWER: {answer}")

def main():
    # Input question 
    questions = data.retrieve_questions()
    question = questions['test'][0]['question']
    print(f"trying the following example question: \n {question}")
    

    answer_question_pipeline(question)

    
if __name__=="__main__":
    main()