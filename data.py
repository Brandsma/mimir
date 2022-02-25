from datasets import load_dataset

def retrieve_questions():
    return load_dataset("GroNLP/ik-nlp-22_slp", name="questions")

def retrieve_paragraphs():
    return load_dataset("GroNLP/ik-nlp-22_slp")