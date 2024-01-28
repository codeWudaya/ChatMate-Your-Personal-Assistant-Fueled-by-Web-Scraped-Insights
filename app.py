import streamlit as st
import pandas as pd
from transformers import pipeline

# Load TAPAS model for table-based question answering
tqa = pipeline(task="table-question-answering", model="google/tapas-base-finetuned-wtq")

# Load table from CSV
table = pd.read_csv("C:\\Users\\udaya\\Desktop\\ChatBotUsing Web Scraping\\Companies.csv").astype(str)

# Streamlit app
def main():
    st.title("ChatMate 🤖")

    # Display the loaded table
    st.write("Scrapped data from wikipedia and converted to Table:")
    st.write(table)

    # User input
    query = st.text_input("Ask a question about the table:")

    # Button to submit the question
    if st.button("Get Answer"):
        if query:
            # Get the answer using TAPAS
            answer = tqa(table=table, query=query)["answer"]
            st.success(f"Answer: {answer}")
        else:
            st.warning("Please enter a question.")

    # Example questions
    st.subheader("Example Questions:")
    example_questions = [
        "Ask any questions realted to table for example 👇👇",
        "Which companies provide healthcare?",
        "What is the revenue of Apple?",
        "Headquarters of Apple and which place?",
        # Add more example questions as needed
    ]

    for example_question in example_questions:
        st.write(f"- {example_question}")

# Run the Streamlit app
if __name__ == "__main__":
    main()
