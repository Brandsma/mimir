import pandas as pd

class Data_pipeline:

    def __init__(self):
        self.qa_results = []
        self.q_gen_results = []
        self.summarize_results = []

    def add_qa_results(self, results_object):
        self.qa_results.append(results_object)

    def add_q_gen_results(self, results_object):
        self.q_gen_results.append(results_object)

    def add_summarize_results(self, results_object):
        self.summarize_results.append(results_object)

    def qa_to_df(self):
        df = pd.DataFrame(self.qa_results)
        return df

    def q_gen_to_df(self):
        df = pd.DataFrame(self.q_gen_results)
        return df

    def summarize_to_df(self):
        df = pd.DataFrame(self.summarize_results)
        return df

    def to_df(self):
        df = pd.concat([self.qa_to_df(), self.q_gen_to_df(), self.summarize_to_df()], axis = 1)
        return df