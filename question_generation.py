from util.logger import setup_logger
from transformers import pipeline

log = setup_logger(__name__)


def generate_question(context):
    if not isinstance(context, str): log.error("context is not a string.")
    qgen = pipeline(model = "mrm8488/t5-base-finetuned-question-generation-ap")
    return qgen("context: " + context)[0]['generated_text']


if __name__ == "__main__":
    context = "Cows can weigh up to 700 kilograms."
    log.info(generate_question(context))
