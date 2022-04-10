# Mimir - A Natural Language Study Assistant

Mimir is capable of doing two distinct things:

1. Given a question and a text, generate an answer for the question
2. Given a text, generate questions and their associated answers

## How to run

### CLI
The model can be run in two different modes, an 'answering' mode and a 'generation' mode.

In order to answer a random question for the NLP dataset, run:
```bash
python main.py --mode answering
```

In order to answer a specific question for the NLP dataset, run:
```bash
python main.py --mode answering --question "What does the Kleene star in Regex mean?" 
```

In order to generate a question from a text, run:
```bash
python main.py --mode generation --text_file ./_data/input/hiroshima_article.txt
```

In order to generate a question from a text filtered with subjects, run:
```bash
python main.py --mode generation --text_file ./_data/input/hiroshima_article.txt --subjects fire bomb hospital
```

## Architecture

### Graphical Representation

In order to run the program with a graphical representation, run the program either from the command line with the argument GUI set to True, or run the command "python viewer.py".
The GUI has limited functionality, but it is a little easier to interface with.

To use the QA model, first load the QA model, then enter a question in the field, and click the Answer question! button. The model will load the dataset as set in the settings.toml to answer the question.
To use the QG model, load the QG model and load a .txt file in the Input filepath field. The user can specify the number of questions to be generated and the subjects that the questions should concern (for multiple, seperate subjects the subjects should be entered as a list of strings in python syntax.). The weighting for subject relevance versus QA quality can be set with the slider, where a lower value returns questions that are more relevant to the entered subjects and a higher value prioritizes QA pairs that are deemed of higher quality.
Pressing the Generate questions! button returns the QA pairs in the questions text field box as a dictionary.
### Modules

There are several modules in this Python project. A short description is given for each of the modules:

- inp_out
*An IO module that handles data objects, input and output.*

- context
*Everything related to context handling for questions, mostly deals with semantic search.*

- question
*Everything related to question handling. This means both question answering and question generation.*

- processing
*This module does any pre- and post-processing. At the moment it can do summarization of text and translation.*

- util
*A module containing any functions that did not clearly belong to some module, but are helpful in multiple occassions.*


## Notes

*Created by:*
- B.J. van Marum
- I.E. Steegstra
- A. Brandsma

