from guizero import App, Text, TextBox, PushButton, Slider, Box
from inp_out.io import IO
from question.question_answering import QuestionAnswering
from question.question_generation import QuestionGenerator, print_qa

# This GUI is pure bonus, so it was written quickly without regard for making it look pretty or being good code

class Viewer():

    def __init__(self):
        self.app = App(title = "Project Mímir", width = 500, height = 600) #layout = "grid", 
        self.boxes = {}
        self.texts = {}
        self.buttons = {}
        self.inputs = {}
        self.settings = {}
        self.io = None
        self.qa = None
        self.qg = None
        self.text = None

    def init(self):
        self.make_boxes()
        self.make_text_elements()
        self.make_inputs()
        self.make_buttons()

    def display(self):
        self.app.display()

    def make_boxes(self):
        self.boxes = {
            "title_box" : Box(self.app, width = "fill", align = 'top'),
            "q_box" : Box(self.app, height = 'fill', align = 'left', layout = 'grid'),
            "data_box" : Box(self.app, height = "fill", align = 'right', layout = 'grid'),
        }

    def make_text_elements(self):
        self.texts = {
            "title" : Text(self.boxes["title_box"], text = "Project Mimir", size = 32),
            "qa_title" : Text(self.boxes["q_box"], text = "Question Answering", align = 'left', size = 18, grid = [0,0,2,1]),
            "question" : Text(self.boxes["q_box"], text = "question:", align = 'left', grid = [0,2]),
            "answer" : Text(self.boxes["q_box"], text = "answer:", align = 'left', grid = [0,4]),
            "question_answer" : TextBox(self.boxes["q_box"], width = 40, align = 'left', text = "", grid = [0,5,2,1]),

            "qg_title" : Text(self.boxes["q_box"], text = "Question Generation", align = 'left', size = 18, grid = [0,6,2,1]),
            "n_gc" : Text(self.boxes["q_box"], text = "generate n questions:", align = 'left', grid = [0,8,2,1]),
            "q_subj" : Text(self.boxes["q_box"], text = "subjects - for multiple enter as list.", align = 'left', grid = [0,10]),
            "qr_weight" : Text(self.boxes["q_box"], text = "relevance <- vs -> quality", align = 'left', grid = [0,12,2,1]),
            "questions" : Text(self.boxes["q_box"], text = "questions:", align = 'left', grid = [0,14]),
            "generated_question" : TextBox(self.boxes["q_box"], multiline = True, width = 40, height = 8, align = 'left', text = "", grid = [0,15,2,4]),
            
            "data_title" : Text(self.boxes["data_box"], text = "Data Input", align = 'left', size = 18, grid = [3,0]),
            "inp_path" : Text(self.boxes["data_box"], text = "Input filepath", align = 'left', grid = [3,2]),
            "file_loading" : Text(self.boxes["data_box"], text = "", align = 'left', grid = [3,5]),
        }

    def make_inputs(self):
        self.inputs = {
            "input_question" : TextBox(self.boxes["q_box"], width = 40, align = 'left', grid = [0,3,2,1]),
            "num_gen_questions" : Slider(self.boxes["q_box"], end = 10, align = 'left', width = 250, grid = [0,9,2,1]),
            "question_subjects" : TextBox(self.boxes["q_box"], align = 'left', width = 40, grid = [0,11,2,1]),
            "quality_vs_relevance_weight": Slider(self.boxes["q_box"], end = 100, align = 'left', width = 250, grid = [0,13,2,1]),
            "input_filepath" : TextBox(self.boxes["data_box"], align = 'left', width = 'fill', grid = [3,3]),
        }
    

    def make_buttons(self):
        self.buttons = {
            "load_QA_model" : PushButton(self.boxes["q_box"], text = "Load QA model", align = 'left', command = self.load_QA, grid = [0,1]),
            "load_QG_model" : PushButton(self.boxes["q_box"], text = "Load QG model", align = 'left', command = self.load_QG, grid = [0,7]),
            "answer_question" : PushButton(self.boxes["q_box"], text = "Answer question!", command = self.answer_question, grid = [1,1]),
            "generate_question" : PushButton(self.boxes["q_box"], text = "Generate questions!", command = self.generate_questions, grid = [1,7]),
            "load text file" : PushButton(self.boxes["data_box"], text = "Load file", command = self.load_input_file, grid = [3,4]),
        }

    def get_settings(self):
        self.settings = {
            "input_question" : self.inputs["input_question"].value,
            "num_generated_questions" : self.inputs["num_gen_questions"].value,
            "question_subjects" : self.inputs["question_subjects"].value,
            "quality_vs_relevance_weight" : float(self.inputs["quality_vs_relevance_weight"].value) / 100.0,
            "input_filepath" : self.inputs["input_filepath"].value,
        }

    def answer_question(self):
        # get the settings from the GUI
        self.get_settings()
        question = self.settings["input_question"]

        if self.io == None:
            self.load_IO()

        # If the model is not loaded. (better to not load both models at startup)
        if self.qa == None:
            self.texts["question_answer"].value = "Please load the QA model."
            return

        self.texts["question_answer"].value = f"Retrieving answer for \"{question}\""
        # Run the answer retrieval pipeline
        results = self.qa.answer_question(question, self.io.get_paragraphs())
        self.texts["answer"] = Text(self.app, text = results["answer"], grid = [0,9])

    def generate_questions(self):
        self.get_settings()

        if self.io == None:
            self.load_IO()

        if self.qg == None:
            self.texts["generated_question"].value = "Please load the QG model."
            return

        if self.text == None:
            self.texts["generated_question"].value = "Please load a text file."
            return

        # Run the question generation pipeline
        qa_list = self.qg.generate(self.text, num_questions=self.settings["num_generated_questions"], subjects=self.settings["question_subjects"])
        self.texts["generated_question"].value = qa_list
        print(qa_list)


    def load_QA(self):
        self.texts["question_answer"].value = "Loading QA model, please wait..."
        self.qa = QuestionAnswering()
        self.texts["question_answer"].value = "QA model loaded."

    def load_QG(self):
        self.texts["generated_question"].value = "Loading QG model, please wait..."
        self.qg = QuestionGenerator()
        self.texts["generated_question"].value = "QG model loaded."

    def load_IO(self):
        self.io = IO()

    def load_input_file(self):
        path = self.inputs["input_filepath"].value
        try:
            with open(path, 'r') as f:
                self.text = f.read()
        except Exception as e:
            err_msg = f"Error at reading file: {e}"
            self.texts["file_loading"].text_color = "red"
            self.texts["file_loading"].value = err_msg
            print(err_msg)
        self.texts["file_loading"].text_color = "green"
        self.texts["file_loading"].value = "file loaded"

if __name__ == "__main__":
    v = Viewer()
    v.init()
    v.display()