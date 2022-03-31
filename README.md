# Mimir - A Natural Language Study Assistant

Mimir is capable of doing two distinct things:

1. Given a question and a text, generate an answer for the question
2. Given a text, generate questions and their associated answers

## How to run

```bash
python main.py
```

## Architecture

### Graphical Representation

TODO

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


TODO Write about this
List of features:
- Question Generation 
- Question Generation with subjects
- Question Answering
- Question translation
- Auto-detect language for translation
- Text Summarization
