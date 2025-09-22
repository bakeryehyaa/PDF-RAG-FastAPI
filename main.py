# main.py

import os
from rag import initialize_rag_system, answer_question

def main():
    """
    Main function to run the RAG application.
    """
    # Define the path to your PDF file
    pdf_file_path = "data/data.pdf"
    
    if not os.path.exists(pdf_file_path):
        print(f"Error: The file '{pdf_file_path}' was not found.")
        return

    # Initialize the RAG system
    print("Initializing RAG system...")
    initialize_rag_system(pdf_file_path)
    print("RAG system is ready.")


    while True:
        user_query = input("\nEnter your question (or type 'exit' to quit): ")
        if user_query.lower() == 'exit':
            break

        try:
            answer = answer_question(user_query)
            print("\nAnswer:", answer)
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()