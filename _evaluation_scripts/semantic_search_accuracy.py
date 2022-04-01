"""
This script computes the accuracy of our semantic search implementation
"""


if __name__ == "__main__":
    import sys
    sys.path.append("..")

from inp_out.io import IO
from context.context import ContextRetrieval

io = IO()
context = ContextRetrieval()

questions = io.get_all_questions()
true_context_list = io.get_all_true_paragraphs()
context_list = io.get_paragraphs()

context.embed(context_list)

counter = 0
trunc_idx = len(questions)
for i, question in enumerate(questions[:trunc_idx]):
    best_context_list = [x[0] for x in context.retrieve_context(question, 5)]
    print("X"*30)
    print(i)
    print(question)
    print("-"*30)
    for line in best_context_list:
        print("-->", line)
    print("-"*30)
    print(true_context_list[i])
    print("percentage correct so far: ", counter/(i+1), " ", counter, "/", i+1)

    if true_context_list[i] in best_context_list:
        counter += 1

print("percentage correct: ", counter/trunc_idx, " ", counter, "/", trunc_idx)
