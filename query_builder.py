from typing import Optional, List

def format_query(question: str, split_context: Optional[List[str]] = None, choices: Optional[List[str]] = None):
  # One of the possible options:
  # <QUESTION> \n (a) <CHOICE_A> (b) <CHOICE_B> ... \n <CONTEXT> 
  
  query = ""
  # question
  query += question
  query += ' \n'

  # choices if available
  if choices != None:
    for idx, choice in enumerate(choices):
      query += f" ({chr(97 + idx)}) {choice}"
    query += ' \n '

  # context if available
  if split_context != None:
    for sentence in split_context:
      query += (sentence + '. ')
  
  return query