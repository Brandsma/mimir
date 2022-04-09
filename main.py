from inp_out.io import IO
from mimir.installer.installer import install_required_packages
from mimir.viewer import Viewer
from util.logger import setup_logger

import argparse

from dynaconf import settings
from transformers import pipeline

from context.context import ContextRetrieval
from processing.translator import Translator
from question.question_answering import QuestionAnswering
from question.question_generation import QuestionGenerator, print_qa

log = setup_logger(__name__)

# Pipelines


def give_answer_to_question_pipeline(question, context_list):
    log.info("Starting 'give answer to question' pipeline...")
    qa = QuestionAnswering(context_list)
    results = qa.answer_question(question)
    return results

def generate_questions_pipeline(text, subjects, answering_style):
    qg = QuestionGenerator()
    qa_list = qg.generate(text, num_questions=settings["num_generated_questions"], subjects=subjects, answer_style=answering_style)
    print_qa(qa_list, show_answers=True)

# Main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mimir - A Natural Language Study Assistant")
    parser.add_argument(
        "-m", "--mode",
        default="answering",
        type=str,
        help="The mode in which mimir runs. Choose from ['interactive', 'answering', 'generation']",
    )
    # General
    parser.add_argument("--text_file", type=str, help="The text file from which questions are generated OR which is used as a context for question answering")

    # Question Generation
    parser.add_argument("--subjects", nargs="+", help="The subjects that the question generation has to try and look for")
    parser.add_argument(
        "--answering_style",
        default="sentences",
        type=str,
        help="The type of question-answer pairs the model generates. Choose from ['sentences', 'multiple_choice']",
    )

    # Question Answering
    parser.add_argument("--question", type=str, help="The question that has to be answered in 'answering' mode. If none, then a random question is selected.")

    # Installer
    parser.add_argument(
        "-i", "--installer", 
        action = "store_true",
        help="Run with this argument to install all required packages."
    )

    # GUI
    parser.add_argument(
        "-g", "--GUI", 
        action = "store_true",
        help="Run the program with a (limited usability) GUI."
    )

    # Parse a text that is inputted
    #parser.add_argument("--use_nlp_dataset", dest="use_nlp_dataset", action="store_true", default=True, help="Whether or not to use the provided dataset")
    return parser.parse_args()

def main():
    args = parse_args()
    if args.mode == "interactive":
        print("Running in interactive mode - TODO")
        return

    if args.installer:
        install_required_packages()

    if args.GUI:
        v = Viewer()
        v.run()
        return

    if args.mode == "answering":
        io = IO()
        question = args.question
        if args.text_file is None:
            if question is None:
                log.info("No question is given, picking a random question from dataset...")
                question = io.get_random_question()
            result_object = give_answer_to_question_pipeline(question, io.get_paragraphs())
            io.print_results(result_object)
        else:
            if question is None:
                log.error("When providing a text file, please also provide a questiorn")
                return
            with open(args.text_file, 'r') as f:
                text_file = f.read()
            result_object = give_answer_to_question_pipeline(question, text_file.split("\n\n"))
            io.print_results(result_object)



    if args.mode == "generation":
        if args.text_file is None:
            log.error("Please provide a text file from which to generate the questions")
            return
        with open(args.text_file, 'r') as f:
            text_file = f.read()
        generate_questions_pipeline(text_file, args.subjects, args.answering_style)



def old_main():
    args = parse_args()
    with open("./data/input/hiroshima_article.txt", 'r') as file:
        text_file = file.read()
    qg = QuestionGenerator()
    qa_list = qg.generate(
        text_file,
        subjects=["Hiroshima", "children"]
    )
    print_qa(qa_list, show_answers=args.show_answers)


if __name__ == "__main__":
    main()
