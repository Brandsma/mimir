import data
from answer_retrieval import retrieve_answer
from context_retrieval import retrieve_context, retrieve_context_FAISS
from logger import setup_logger

log = setup_logger(__name__)

def analyze_contexts(contexts):
    result = [(x, len(x)) for x in contexts if len(x) > 512]
    for thing in result:
        print(f"{thing[1]} : {thing[0]}")
        print("")



def answer_question_pipeline(question):
    # Input (raw) source text
    # Normal
    log.info("Retrieving Dataset...")
    source = data.retrieve_paragraphs()
    contexts = [x for x in source['train']['text']]

    # log.info("Sizing down sequence lengths that are too large...")
    # contexts, context_groupings = chop_up_dem_contexts(contexts, max_sequence_length)
        
    # retrieve context from source text based on question
    log.info("Retrieving the most relevant context paragraphs...")
    context = retrieve_context(question, contexts, n=5)
    context = [x[0] for x in context]
    print([len(x) for x in context if len(x) > 512])

    # Answer question with context
    log.info("Trying to answer the question with the found context...")
    answer = retrieve_answer(question, context, language='en', num_generated_answers=5, num_correct_answers_required=3)

    # Return answer
    print("")
    print(f"QUESTION: {question}\nANSWER: {answer}")

def main():
    # Input question 
    questions = data.retrieve_questions()
    question = questions['test'][1]['question']
    print(f"trying the following example question: \n {question}")
    

    answer_question_pipeline(question)

    
if __name__=="__main__":
    main()