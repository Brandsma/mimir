import pprint

from datasets import load_dataset
from dynaconf import settings
from util.logger import setup_logger

log = setup_logger(__name__)


class IO:
    def __init__(self, dataset_name = "GroNLP/ik-nlp-22_slp"):
        self.dataset_name = dataset_name
        if dataset_name == "GroNLP/ik-nlp-22_slp":
            self.questions = load_dataset(dataset_name,
                                        name="questions",
                                        split="test")
            self.paragraphs = load_dataset(dataset_name,
                                        name="paragraphs",
                                        split="train")
        elif dataset_name == "squad" or dataset_name == "squad_v2":
            self.dataset = load_dataset(dataset_name, split="train[0:3500]")
        else:
            log.error(f"Invalid dataset was given to IO. Should be one of [GroNLP/ik-nlp-22_slp, squad, squad_v2], but found {dataset_name}")


    def get_paragraphs(self):
        if self.dataset_name == "GroNLP/ik-nlp-22_slp":
            return [x for x in self.paragraphs['text']]
        elif self.dataset_name == "squad" or self.dataset_name == "squad_v2":
            return [x for x in self.dataset['context']]

    def get_single_question(self, question_number):
        if self.dataset_name == "GroNLP/ik-nlp-22_slp":
            return self.questions[question_number]['question']
        elif self.dataset_name == "squad" or self.dataset_name == "squad_v2":
            return self.dataset['question'][question_number]

    def get_all_questions(self):
        if self.dataset_name == "GroNLP/ik-nlp-22_slp":
            return [x['question'] for x in self.questions]
        elif self.dataset_name == "squad" or self.dataset_name == "squad_v2":
            return [x['question'] for x in self.dataset]

    def get_all_true_answers(self):
        if self.dataset_name == "GroNLP/ik-nlp-22_slp":
            return [x['answer'] for x in self.questions]
        elif self.dataset_name == "squad" or self.dataset_name == "squad_v2":
            return [x['answers']["text"][0] for x in self.dataset]
            

    def get_all_true_paragraphs(self):
        if self.dataset_name == "GroNLP/ik-nlp-22_slp":
            return [x['paragraph'] for x in self.questions]
        elif self.dataset_name == "squad" or self.dataset_name == "squad_v2":
            return [x['context'] for x in self.dataset]

    def print_results(self, results_object):
        pp = pprint.PrettyPrinter(indent=4, depth=4)
        pp.pprint(results_object)
