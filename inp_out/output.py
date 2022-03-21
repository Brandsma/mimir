import pprint
from datasets import load_dataset
from dynaconf import settings

class IO:
    def __init__(self):
        print(settings["colors"])
        
    def get_raw_question_and_contexts(self): 
        return load_dataset("GroNLP/ik-nlp-22_slp", name="questions"), load_dataset("GroNLP/ik-nlp-22_slp")

    def print_results(self, results_object):
        pp = pprint.PrettyPrinter(indent=4, depth=4)
        pp.pprint(results_object)