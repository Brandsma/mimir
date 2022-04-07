from guizero import App, Text, TextBox, PushButton, Slider
from inp_out.io import IO
from question.question_answering import QuestionAnswering

# This GUI is pure bonus, so it was written quickly without regard for making it look pretty or being good code

class Viewer():

    def __init__(self):
        self.app = App(title = "Project Mímir", layout = "grid")
        self.texts = {}
        self.buttons = {}
        self.inputs = {}
        self.settings = {}

    def init(self):
        self.make_text_elements()
        self.make_inputs()
        self.make_buttons()

    def display(self):
        self.app.display()

    def make_text_elements(self):
        self.texts = {
            "question" : Text(self.app, text = "Question: ", grid = [0,0]),
            "q_num" : Text(self.app, text = "Input Question Number: ", grid = [0,1]),
            "n_c" : Text(self.app, text = "Number of contexts", grid = [0,2]),
            "n_a" : Text(self.app, text = "Number of answers", grid = [0,3]),
            "n_gc" : Text(self.app, text = "Number of generated questions", grid = [0,4]),
            "q_subj" : Text(self.app, text = "Question subjects", grid = [0,5]),
            "qr_weight" : Text(self.app, text = "Quality vs relevance weight", grid = [0,6]),
            "inp_path" : Text(self.app, text = "Input filepath", grid = [0,7]),
            "ds_name" : Text(self.app, text = "Dataset name", grid = [0,8]),
        }

    def make_inputs(self):
        self.inputs = {
            "input_question" : TextBox(self.app, grid = [1,0]),
            "question_number" : Slider(self.app, end = 10, grid = [1,1]),
            "num_contexts" : Slider(self.app, end = 10, grid = [1,2]),
            "num_answers" : Slider(self.app, end = 10, grid = [1,3]),
            "num_gen_questions" : Slider(self.app, end = 10, grid = [1,4]),
            "question_subjects" : TextBox(self.app, grid = [1,5]),
            "quality_vs_relevance_weight": Slider(self.app, end = 100, grid = [1,6]),
            "input_filepath" : TextBox(self.app, grid = [1,7]),
            "dataset_name" : TextBox(self.app, grid = [1,8]),
        }
    

    def make_buttons(self):
        self.buttons = {
            "answer_question" : PushButton(self.app, text = "Answer question!", command = self.answer_question, grid = [2,0]),
            "generate_question" : PushButton(self.app, text = "Generate questions!", command = self.generate_questions, grid = [2,1]),
        }

    def get_settings(self):
        self.settings = {
            "input_question" : self.inputs["input_question"].value,
            "question_number" : self.inputs["question_number"].value,
            "number_of_retrieved_contexts" : self.inputs["num_contexts"].value,
            "num_generated_answers" : self.inputs["num_answers"].value,
            "num_generated_questions" : self.inputs["num_gen_questions"].value,
            "question_subjects" : self.inputs["question_subjects"].value,
            "quality_vs_relevance_weight" : float(self.inputs["quality_vs_relevance_weight"].value) / 100.0,
            "input_filepath" : self.inputs["input_filepath"].value,
            "dataset_name" : self.inputs["dataset_name"].value,
        }

    def answer_question(self):
        # Run the answer retrieval pipeline
        question = self.settings["input_question"]
        io = IO()
        qa = QuestionAnswering()
        results = qa.answer_question(question, io.get_paragraphs())
        self.texts["answer"] = Text(self.app, text = results["answer"], grid = [0,9])

    def generate_questions(self):
        # Run the question generation pipeline
        pass 


if __name__ == "__main__":
    v = Viewer()
    v.init()
    v.display()