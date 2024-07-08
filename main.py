import streamlit as st
from Essay_Exam_Evaluator import Exam
import pandas as pd
import os

# Define the path to the exam CSV file and other constants
EXAM_FILE_PATH = 'machine_learning_questions_answers.csv'
NUM_QUESTIONS = 2
INSTRUCTIONS = "There are no specific instructions"
test_name="lec1_exam_results"
# Initialize the exam instance
exam_instance = Exam(exam_path=EXAM_FILE_PATH, questions_column="Question",folder_path=test_name)


def main():
    st.title("TEST YOUR KNOWLEDGE!")
    st.sidebar.title("Options")

    # Initialize session state variables if not already initialized
    if "exam_started" not in st.session_state:
        st.session_state.exam_started = False

    if "selected_questions" not in st.session_state:
        st.session_state.selected_questions = []

    if not st.session_state.exam_started:
        st.sidebar.text("Enter student information:")
        st.session_state.student_name = st.sidebar.text_input("Student Name")

        # Randomly select questions when starting the exam
        st.session_state.selected_questions = exam_instance.select_random_questions(NUM_QUESTIONS)

    if st.sidebar.button("Start Test"):
        if not st.session_state.student_name:
            st.warning("Please enter student name.")
        else:
            st.session_state.exam_started = True

    if st.session_state.exam_started:
        selected_questions = st.session_state.selected_questions
        st.write("Answer the following questions:")

        if "student_answers" not in st.session_state:
            st.session_state.student_answers = {}

        # Display each question and record answers
        all_answers_given = True  # Flag to check if all questions have been answered
        for index, question in enumerate(selected_questions):
            st.write(question)
            answer_key = f"answer_{index}"
            st.session_state.student_answers[answer_key] = st.text_area(answer_key,
                                                                         st.session_state.student_answers.get(
                                                                             answer_key, ""))
            # Check if any answer is empty
            if not st.session_state.student_answers[answer_key]:
                all_answers_given = False
            # Check if the answer is the same as the question
            elif st.session_state.student_answers[answer_key] == question:
                st.error("You cannot answer the same question. Please provide a different answer.")
                all_answers_given = False

        if st.button("Submit"):
            # Check if all questions have been answered
            if all_answers_given:
                # Convert student answers to DataFrame
                exam_sheet_df = pd.DataFrame({"Question": selected_questions,
                                              "Student Answer": [st.session_state.student_answers[f"answer_{index}"] for
                                                                 index, _ in enumerate(selected_questions)]})

                # Save student answers to CSV file
                exam_sheet_df.to_csv(f"answers_{st.session_state.student_name}.csv", index=False)

                # Evaluate student answers
                success_rate, results_df = exam_instance.evaluate_student_answers(st.session_state.student_name,
                                                                                  exam_sheet_df,
                                                                                  INSTRUCTIONS)
                st.success(f"Test score: {success_rate:.2f}%")

                # Display results DataFrame
                st.write("Results:")
                st.write(results_df)
                st.session_state.exam_started = False

                # Clear student answers from session state
                st.session_state.student_answers = {}
                save_to_excel(test_name, st.session_state.student_name, success_rate)
            else:
                st.warning("Please answer all questions before submitting.")

def save_to_excel(folder_name, student_name, grade):
    # Ensure the folder exists, if not, create it
 
    # Create or load Excel file
    file_path = os.path.join(folder_name, "results.xlsx")
    if os.path.exists(file_path):
        results_df = pd.read_excel(file_path)
    else:
        results_df = pd.DataFrame(columns=["Student Name", "Grade"])

    # Append new row with student name and grade
    new_row = {"Student Name": student_name, "Grade": grade}
    results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)

    # Save DataFrame to Excel
    results_df.to_excel(file_path, index=False)

if __name__ == "__main__":
    main()


#streamlit run main.py