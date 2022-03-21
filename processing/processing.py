from typing import List
from dynaconf import settings
from setuptools import setup
from util.logger import setup_logger
from transformers import pipeline

log = setup_logger("Processing")

class Processing:

    def __init__(self):
        pass

    def translate(self, text, from_lang, to_lang):
        if from_lang == to_lang:
            log.info("Input and output language match, no translation required.")
            return text

        log.info(f"Translating everything from {from_lang} to {to_lang}...")
        nl_en_translator = pipeline(
            "text2text-generation", model=f"Helsinki-NLP/opus-mt-{from_lang}-{to_lang}")
        
        if type(text) == List:
            translated_text_elements = []
            for idx, split in enumerate(text):
                translated_text_elements[idx] = nl_en_translator(split)[0]['generated_text']
            return translated_text_elements
        else:
            return nl_en_translator(text)[0]['generated_text']
