from typing import Optional, List
from setuptools import setup
from transformers import pipeline
from collections import Counter
from logger import setup_logger
from query_builder import format_query
from loader import Loader
from nlp_util import split_sentences


log = setup_logger(__name__)


def determine_most_frequent(list):
    occurence_count = Counter(list)
    most_common = occurence_count.most_common(1)[0][0]
    return most_common, occurence_count[most_common]


def retrieve_answer(question: str, context: Optional[List[str]] = None, choices: Optional[List[str]] = None, language: Optional[str] = 'en', num_generated_answers=5, num_correct_answers_required=3):
    #loader = Loader("Loading with object...", "That was fast!", 0.05).start()
    log.info(f"Starting to answer the question: {question}")
    # Prepare input data
    split_context = []

    if context != None:
        for text in context:
            split_context += split_sentences(text)
            
    # Translate everything
    if language != 'en':
        log.info(f"Translating everything from {language} to en...")
        nl_en_translator = pipeline(
            "text2text-generation", model=f"Helsinki-NLP/opus-mt-{language}-en")
        question = nl_en_translator(question)[0]['generated_text']
        if choices != None:
            for idx, choice in enumerate(choices):
                choices[idx] = nl_en_translator(choice)[0]['generated_text']
        if context != None:
            for idx, context_sentence in enumerate(split_context):
                split_context[idx] = nl_en_translator(context_sentence)[
                    0]['generated_text']

    # Generate paraphrases of the question
    log.info("Generating paraphrases of question...")
    paraphraser = pipeline("text2text-generation",
                           model="tuner007/pegasus_paraphrase")
    paraphrased_questions = paraphraser(
        question, num_return_sequences=num_generated_answers)

    # Generate an answer using AllenAI
    log.info("Attempting to answer questions...")
    answers = []
    for paraphrased_question in paraphrased_questions:
        generator = pipeline("text2text-generation",
                             model="allenai/unifiedqa-t5-base")
        query = format_query(
            paraphrased_question['generated_text'], split_context, choices)
        answers.append(generator(query)[0]['generated_text'])

    # Return the most common answer if we get it right often enough, translate it back to original language
    if language != 'en':
        en_nl_translator = pipeline(
            "text2text-generation", model=f"Helsinki-NLP/opus-mt-en-{language}")
    most_common_answer, answer_count = determine_most_frequent(answers)
    if answer_count >= num_correct_answers_required:
        if language != 'en':
            return en_nl_translator(most_common_answer)[0]['generated_text']
        else:
            return most_common_answer
    else:
        if language != 'en':
            return en_nl_translator("No common answer found")[0]['generated_text']
        else:
            return "No common answer found"


if __name__ == "__main__":
    context = "De Eiffeltoren is een monument in Parijs en een van bekendste en meest bezochte bezienswaardigheden van Frankrijk. Hij staat aan de linkeroever van de Seine in het 7e arrondissement van Parijs. De Eiffeltoren is hét symbool van Parijs en wordt door velen gezien als een van de Niet klassieke wereldwonderen. De toren ontving tussen 2011 en 2017 jaarlijks meer dan zes miljoen bezoekers en is daarmee het meest bezochte monument ter wereld waar een toegangskaartje voor gekocht moet worden. Op 28 november 2002 verwelkomde de Eiffeltoren zijn 200 miljoenste gast. De toren is ontworpen door de ingenieurs Maurice Koechlin en Émile Nouguier, twee medewerkers van Gustave Eiffel, en werd gebouwd ter gelegenheid van de wereldtentoonstelling van 1889 in Parijs."
    choices = ["Parijs", "London", "Praag", "Berlijn"]
    log.info(retrieve_answer("Wat is de naam van de stad waar de eiffeltoren staat?",
             context=context, choices=choices, language='nl'))
