from typing import List
from util.logger import setup_logger
from transformers import pipeline
from textblob import TextBlob


log = setup_logger("Processing")

class Translator:

    def __init__(self, from_lang="", to_lang="en"):
        self.from_lang = from_lang
        self.to_lang = to_lang
        if self.from_lang == self.to_lang:
            return
        
        if self.from_lang != "":
            self.translator = pipeline("text2text-generation", model=f"Helsinki-NLP/opus-mt-{from_lang}-{to_lang}")
    
    def auto_detect_language(self, text):
        # If no input language was given, then try to detect it
        b = TextBlob(text)
        lang = str(b.detect_language())
        if lang == "en":
            self.from_lang = "en"
            self.to_lang = "en"
        else:
            self.from_lang = lang
            self.translator = pipeline("text2text-generation", model=f"Helsinki-NLP/opus-mt-{self.from_lang}-{self.to_lang}")
            
    def translate(self, text):
        if self.from_lang == "":
            self.auto_detect_language(text)
            
        if self.from_lang == self.to_lang:
            log.debug("Input and output language match, no translation required.")
            return text
        
        log.debug(f"Translating everything from {self.from_lang} to {self.to_lang}...")
        
        if self.translator is None:
            log.error(f"Translator was not found for {self.from_lang}->{self.to_lang}, no translation was performed.\n Attempting without translation...")
            return text
        
        if type(text) == List:
            translated_text_elements = []
            for idx, split in enumerate(text):
                translated_text_elements[idx] = self.translator(split)[0]['generated_text']
            return translated_text_elements
        else:
            return self.translator(text)[0]['generated_text']
