# TalkToDocs: A Multi-Document Study Assistant

TalkToDocs is an AI-powered Streamlit application designed to help you interact with your documents. Upload multiple PDF, DOCX, or TXT files, and then ask questions, generate summaries, create multiple-choice questions, get definitions, compare concepts, or extract information in bullet points.

## Features

-   **Multi-Document Support**: Seamlessly upload and process multiple PDF, DOCX, and TXT files.
-   **AI-Powered Q&A**: Ask natural language questions about your uploaded documents.
-   **Intent-Based Responses**: The application intelligently determines your intent to provide tailored responses:
    -   **Summaries**: Get concise summaries of your documents.
    -   **MCQ Generation**: Generate multiple-choice questions based on the content.
    -   **Definitions**: Extract and define key terms.
    -   **Comparisons**: Compare concepts or entities mentioned in your documents.
    -   **Bullet Points**: Get information extracted into clear, concise bullet points.
-   **Document Statistics**: View statistics about your processed documents, including chunk count, estimated tokens, and character count.
-   **Chat History Management**: Save your chat history as a JSON file for future reference.
-   **Export Last Result**: Export the last AI-generated response as a TXT file.
-   **Clear Documents**: Easily clear all processed documents to start fresh.

## Setup

Follow these steps to get TalkToDocs up and running on your local machine.

### Prerequisites

-   Python 3.8+
-   `pip` (Python package installer)
-   A Groq API Key (for the language model)

### Installation

1.  **Clone the repository (if applicable):**
    ```bash
    git clone https://github.com/nav69eed/TalkToDocs.git
    cd TalkToDocs
    ```
    *(Note: If you received the project as a folder, navigate directly into it.)*

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**
    -   **Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    -   **macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

4.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

### API Key Configuration

TalkToDocs uses the Groq API for its language model. You will need to obtain an API key from the [Groq website](https://groq.com/).

-   During the application runtime, you will be prompted to enter your Groq API key in the sidebar.

## Usage

1.  **Run the Streamlit application:**
    ```bash
    streamlit run main.py
    ```

2.  **Open your web browser:**
    The application will open in your default web browser, usually at `http://localhost:8501`.

3.  **Upload Documents:**
    -   In the sidebar, enter your Groq API key.
    -   Click on "Upload documents (PDF, DOCX, TXT)" and select the files you want to process.
    -   The application will process the documents and display statistics.

4.  **Ask Questions:**
    -   Once documents are processed, use the chat input field at the bottom to ask your questions.
    -   Try different types of queries like:
        -   "Summarize the main points."
        -   "Generate 3 MCQs from the text."
        -   "Define [term]."
        -   "Compare X and Y."
        -   "List the key features."

5.  **Manage Chat and Results:**
    -   Use the buttons at the bottom to "Export Last Result as TXT", "Save Chat History", or "Clear All Documents".

## Technologies Used

-   **Streamlit**: For building the interactive web application.
-   **LangChain**: For orchestrating the LLM interactions, document loading, chunking, and retrieval.
-   **Groq API**: For fast and efficient language model inference.
-   **HuggingFace Embeddings**: For generating document embeddings.
-   **FAISS**: For efficient similarity search and vector storage.
-   **PyPDF2**: For PDF text extraction.
-   **python-docx**: For DOCX text extraction.

## Future Enhancements

-   **Improved Error Handling**: More robust error management and user feedback.
-   **Code Modularization**: Refactoring into separate modules for better organization and maintainability.
-   **Configuration Management**: Externalizing configurable parameters.
-   **Unit Testing**: Implementing tests for core functionalities.
-   **Performance Optimizations**: Further enhancements for large document sets.
-   **Advanced UI/UX**: More sophisticated user interface elements and interactions.
-   **Customizable Prompts**: Allow users to modify prompt templates.