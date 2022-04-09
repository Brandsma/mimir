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

TODO
In order to run the program with a graphical representation, run the program either from the command line with the argument GUI set to True, or run the command "python viewer.py".
The GUI has limited functionality, but it is a little easier to interface with.

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

