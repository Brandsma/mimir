import pprint
from datasets import load_dataset
from dynaconf import settings

class IO:
    def get_raw_question_and_contexts(self): 
        questions = load_dataset("GroNLP/ik-nlp-22_slp", name="questions", split="test")
        paragraphs = load_dataset("GroNLP/ik-nlp-22_slp", name="paragraphs", split="train")
        return (questions[settings["question_number"]]['question'], [x for x in paragraphs['text']])

    def print_results(self, results_object):
        pp = pprint.PrettyPrinter(indent=4, depth=4)
        pp.pprint(results_object)