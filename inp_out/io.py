import pprint
from datasets import load_dataset
from dynaconf import settings

class IO:

    def __init__(self):
        self.questions = load_dataset("GroNLP/ik-nlp-22_slp", name="questions", split="test")
        self.paragraphs = load_dataset("GroNLP/ik-nlp-22_slp", name="paragraphs", split="train")
        
    def get_raw_question_and_contexts(self):
        return (self.questions[settings["question_number"]]['question'], [x for x in self.paragraphs['text']])

    def get_paragraphs(self):
        return [x for x in self.paragraphs['text']]
        
    def get_question(self, question_number):
        return self.questions[question_number]['question']

    def print_results(self, results_object):
        pp = pprint.PrettyPrinter(indent=4, depth=4)
        pp.pprint(results_object)