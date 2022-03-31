if __name__=="__main__":
    import sys
    sys.path.append("..")

from inp_out.data_pipeline import DataManager
from inp_out.io import IO
from processing.summarize_text import one_line_summary, summarize_text

def summarize_text_pipeline():
    io = IO()
    data = DataManager()
    paragraphs = io.get_paragraphs()

    for idx, text in enumerate(paragraphs):
        if idx == 2:
            break
        ## Summarize text
        summary = summarize_text(text)

        ## Single line summary
        one_line = one_line_summary(text)

        ## Evaluation
        ## ??

        results_object = {
            "raw_text": text,
            "summary": summary,
            "one_line": one_line
        }
        data.add_summarize_results(results_object)
        io.print_results(results_object)
    print(data.summarize_to_df())
    
if __name__=="__main__":
    summarize_text_pipeline()