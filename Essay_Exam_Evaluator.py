import os
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import pandas as pd
import re
import random


class Exam:
    def __init__(self, exam_path, questions_column, folder_path):
        self.file_path = exam_path
        self.source_column = questions_column
        self.folder_path = folder_path
        self.loader = CSVLoader(file_path=self.file_path, source_column=self.source_column)
        self.data = self.loader.load()
        self.instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
        self.vectordb = FAISS.from_documents(documents=self.data, embedding=self.instructor_embeddings)
        self.retriever = self.vectordb.as_retriever(score_threshold=0.7)
        if "GOOGLE_API_KEY" not in os.environ:
            os.environ["GOOGLE_API_KEY"] = "AIzaSyC2UHXCocEIWVOSFXRX3_dWzBdR4j0kpI8"
        self.llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    def answer_corrector(self, question, answer, instructions="There are no specific instructions"):
        prompt_template = f"As the answer corrector Adhere to these instructions: {instructions}." \
                          f"Regarding the question {question}" \
                          + """provide an improved answer based on the given context and your knowledge.
                          .In your response, incorporate as much information as possible from the "Answer" section in the source document .
                          Rate the correctness of the answer according to its completeness and accuracy with a number from 0 to 1 and Return the percentage of correctness of the answers according to Clarity of Thesis Statement: Graders look for a clear and focused thesis statement that presents the main argument or purpose of the essay. The thesis should be identifiable and relevant to the prompt.
                          Organization and Structure: Graders assess how well the essay is organized and structured. They look for logical flow between paragraphs, clear transitions, and a coherent overall structure that supports the thesis.
                          Supporting Evidence: Graders evaluate the quality and relevance of the evidence provided to support the argument. This may include examples, facts, statistics, quotations, or other forms of evidence that are appropriate to the topic.
                          Analysis and Critical Thinking: Graders assess the depth of analysis and critical thinking demonstrated in the essay. They look for the ability to analyze and interpret the evidence, as well as the ability to evaluate different perspectives or arguments.
                          Clarity and Coherence: Graders consider the clarity and coherence of the writing. This includes factors such as sentence structure, grammar, punctuation, vocabulary choice, and overall readability.
                          Engagement with the Prompt: Graders check if the essay directly addresses the prompt or question provided. They look for relevance and depth of engagement with the topic.
                          Originality and Creativity: Depending on the nature of the essay prompt, graders may also consider the originality and creativity of the ideas presented.
                          Conclusion: Graders assess the effectiveness of the essay's conclusion in summarizing key points and reinforcing the thesis.
                          Overall Impression: Graders provide an overall assessment of the essay, considering factors such as coherence, persuasiveness, and overall impact .
                          Correct the question only, do not answer it, do not complete it, and give the evaluation at the beginning of your statement
                          Evaluation : A number between 0 and 1 depending on the correctness and completeness of the answer
                          If Correct, return Evaluation: 1 only.
                          if answer Correct or incorrect or incomplete Provided with it Then Explain why it is correct, explain why it is incomplete, or explain why it is incorrect as (Explanation:).
                          If the STUDENT'S_ANSWER does not exist or is not related to the question or ="" , rate it as 0
                          If the STUDENT'S_ANSWER is the same question, rate it as 0.

                          Just don't write anything else 
                          Just stick to the context
                          ".
                          CONTEXT: {context}
                          STUDENT'S_ANSWER: {question}"""
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain_type_kwargs = {"prompt": PROMPT}
        chain = RetrievalQA.from_chain_type(llm=self.llm,
                                            chain_type="stuff",
                                            retriever=self.vectordb.as_retriever(),
                                            input_key="query",
                                            return_source_documents=True,
                                            chain_type_kwargs=chain_type_kwargs)
        result = chain.invoke(answer)
        result_text = result["result"]
        # Extract evaluation percentage and explanation using regex
        evaluation_regex = r"Evaluation\s*:\s*([\d.]+)"
        explanation_error_regex = r"Explanation(?:\s*\w+)*\s*:\s*(.*)"

        evaluation_match = re.search(evaluation_regex, result_text)
        explanation_error_match = re.search(explanation_error_regex, result_text)
        evaluation = float(evaluation_match.group(1)) if evaluation_match else None
        explanation_error = explanation_error_match.group(1) if explanation_error_match else None

        return evaluation, explanation_error

    def select_random_questions(self, num_questions):
        # Load the CSV file
        try:
            df = pd.read_csv(self.file_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"File '{self.file_path}' not found.")

        # Check if the source column exists in the dataframe
        if self.source_column not in df.columns:
            raise ValueError(f"Column '{self.source_column}' not found in the CSV file.")

        # Extract questions from the specified column
        questions = df[self.source_column].tolist()

        # Randomly select num_questions from the list of questions
        selected_questions = random.sample(questions, min(num_questions, len(questions)))

        return selected_questions

    def evaluate_student_answers(self, student_name, exam_sheet_df,
                                 instructions="There are no specific instructions"):
        # Load the exam sheet from the CSV file

        exam_sheet = exam_sheet_df

        # Create a new DataFrame to store the results
        results_df = pd.DataFrame(
            columns=["Question", "Student Answer", "Evaluation", "Explanation Error"])

        # Loop through each question and answer in the exam sheet
        for index, row in exam_sheet.iterrows():
            question = row["Question"]
            answer = row["Student Answer"]

            # Call answer_corrector method with instructions
            evaluation, explanation_error = self.answer_corrector(question, answer, instructions)

            # Append the results to the DataFrame
            results_df.loc[index] = [question, answer, evaluation, explanation_error]

        # Ensure the folder exists, if not, create it
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)

        # Define the file name with the folder path
        results_file_name = os.path.join(self.folder_path, f"answers_{student_name}.csv")

        # Write the DataFrame to the CSV file
        results_df.to_csv(results_file_name, index=False)

        # Calculate and return the total grade
        total_student_grade = results_df["Evaluation"].sum()
        total_question_grade = num_rows = results_df.shape[0]
        success_rate = (total_student_grade / total_question_grade) * 100
        return success_rate, results_df