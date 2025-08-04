# Importing required libraries
import streamlit as st
import os
import json
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import PyPDF2
import docx
from datetime import datetime

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "last_result" not in st.session_state:
    st.session_state.last_result = ""
if "uploaded_files_processed" not in st.session_state:
    st.session_state.uploaded_files_processed = []
if "document_stats" not in st.session_state:
    st.session_state.document_stats = {"total_chunks": 0, "total_tokens": 0, "file_details": {}}

# Function to extract text from uploaded file
def extract_text(file):
    try:
        file_extension = os.path.splitext(file.name)[1].lower()
        
        if file_extension == ".pdf":
            reader = PyPDF2.PdfReader(file)
            text = ""
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += f"[Page {i+1}]\n{page_text}\n\n"
                else:
                    text += f"[Page {i+1} - No extractable text]\n\n"
            return text if text.strip() else f"[Warning: No text could be extracted from {file.name}]"
            
        elif file_extension == ".docx":
            doc = docx.Document(file)
            text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
            return text if text.strip() else f"[Warning: No text could be extracted from {file.name}]"
            
        elif file_extension == ".txt":
            text = file.read().decode("utf-8")
            return text if text.strip() else f"[Warning: {file.name} appears to be empty]"
            
        else:
            return f"[Error: Unsupported file type: {file_extension}]"
            
    except Exception as e:
        return f"[Error extracting text from {file.name}: {str(e)}]"


# Function to estimate token count (rough approximation)
def estimate_tokens(text):
    # A very rough estimate: ~4 characters per token for English text
    return len(text) // 4

# Function to chunk text
def chunk_text(text, filename=None):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    # Update document stats if filename is provided
    if filename:
        token_estimate = estimate_tokens(text)
        st.session_state.document_stats["total_chunks"] += len(chunks)
        st.session_state.document_stats["total_tokens"] += token_estimate
        st.session_state.document_stats["file_details"][filename] = {
            "chunks": len(chunks),
            "tokens": token_estimate,
            "characters": len(text)
        }
    
    return chunks

# Function to create vector store
def create_vector_store(texts):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_texts(texts, embeddings)

# Function to determine user intent and select prompt template
def get_prompt_template(intent):
    templates = {
        "summarize": PromptTemplate(input_variables=["context"],template="""
        You are an expert academic assistant. Your goal is to extract the **most important ideas** from the given document. 

        Analyze it **as a whole**. Then, write a clear and concise **summary (3-5 sentences)** that captures the core themes, key arguments, and any major conclusions.

        Avoid repeating minor points or redundant details. Focus on what a reader *must know* to understand the essence of the document.
        make impotant terms bold

        --- Document Start ---
        {context}
        --- Document End ---
        
        Summary:"""),


        "mcq": PromptTemplate(input_variables=["context"],template="""
        You are a subject matter expert and assessment designer.

        Your task is to create **multiple-choice questions** that accurately reflect the core ideas, facts, or concepts from the document provided.

        Each question must:
        - Be **clear**, **unambiguous**, and **answerable** from the content.
        - Include **4 distinct answer options (A–D)**.
        - Clearly **mark the correct option**.
        - Avoid overly trivial or overly complex questions.
        - Focus on **important information**—not minor details.
        - Mix up types of questions: factual, conceptual, interpretative if possible.
        - each option on new line

        Use the following format:

        Q1. [Your question here]  \n
        A. Option A  \n
        B. Option B  \n
        C. Option C  \n
        D. Option D  \n
        **Correct Answer: [A/B/C/D]**

        ---

        Here’s the content to base the questions on:

        --- Document Start ---
        {context}
        --- Document End ---
        
        Now, generate the questions:"""),

        
        "definitions": PromptTemplate(input_variables=["context"],template="""
        Identify and extract important *technical terms*, *concepts*, or *jargon* from the following content.

        For each term, provide:
        - The **term** itself (in bold)
        - A clear and concise **definition** (1–2 lines)
        - An optional **example or context** of how it’s used, if available in the content

        Only include terms that are **relevant**, **non-obvious**, and **central** to understanding the topic.

        Content:
        {context}"""),

        "compare": PromptTemplate(input_variables=["context"],template="""
        You are given the following content and a specific comparison request by the user.

        Your task:
        - Identify the **key elements/entities** that need to be compared.
        - Provide a **clear, structured comparison** using either a table or bullet points.
        - Cover aspects such as: definition, purpose, components, advantages, disadvantages, use-cases, or any other relevant dimensions.
        - Keep the tone **neutral, informative**, and **precise**.

        Only compare what's present in the content and **don’t assume anything not mentioned**.

        Content to analyze:
        {context}"""),
        
        "bullet_points": PromptTemplate(input_variables=["context"],template="""
        Read the following content and extract all relevant information as **clear, concise bullet points**.

        Your task:
        - Focus on **key facts, ideas, or takeaways**.
        - Keep each bullet **short and focused** (1-2 lines max).
        - Use parallel structure if listing multiple related points.
        - Do **not rephrase too much**—preserve original meaning.

        Content:
        {context}"""),


        "default": PromptTemplate(input_variables=["context"],template="""
        You’re chatting with a user who’s asking a question about the following content.

        Your job:
        - Read the document carefully.
        - Answer the question clearly and directly.
        - If the document doesn’t answer it, say so—don’t make stuff up.

        Content:
        {context}""")
    }
    
    # Simple intent detection based on keywords
    intent_lower = intent.lower()
    if "summarize" in intent_lower or "summary" in intent_lower:
        return templates["summarize"]
    elif "mcq" in intent_lower or "question" in intent_lower:
        return templates["mcq"]
    elif "definition" in intent_lower or "terms" in intent_lower:
        return templates["definitions"]
    elif "compare" in intent_lower or "vs" in intent_lower:
        return templates["compare"]
    elif "bullet" in intent_lower or "list" in intent_lower:
        return templates["bullet_points"]
    return templates["default"]

# Function to save chat history to JSON
def save_chat_history():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"chat_history_{timestamp}.json", "w") as f:
        json.dump(st.session_state.messages, f)

# Streamlit app setup
st.title("TalkToDocs: A Multi-Document Study Assistant")
st.write("Upload multiple documents (PDF, DOCX, TXT) and ask anything about them. Get summaries, MCQs, definitions, and more from all your documents at once.")

# Sidebar for API key and file upload
with st.sidebar:
    groq_api_key = st.text_input("Enter your Groq API key", type="password")
    uploaded_files = st.file_uploader("Upload documents (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"], accept_multiple_files=True)
    
    # Process uploaded files
    if uploaded_files and groq_api_key:
        # Check for new files that haven't been processed yet
        new_files = [file for file in uploaded_files if file.name not in st.session_state.uploaded_files_processed]
        
        if new_files:
            with st.spinner(f"Processing {len(new_files)} new document(s)..."):
                all_chunks = []
                
                # Process each new file
                for file in new_files:
                    text = extract_text(file)
                    chunks = chunk_text(text, filename=file.name)
                    all_chunks.extend(chunks)
                    st.session_state.uploaded_files_processed.append(file.name)
                
                # Create or update vector store
                if st.session_state.vector_store is None:
                    st.session_state.vector_store = create_vector_store(all_chunks)
                else:
                    # Add new documents to existing vector store
                    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                    st.session_state.vector_store.add_texts(all_chunks)
                
            st.success(f"{len(new_files)} new document(s) processed! Total documents: {len(st.session_state.uploaded_files_processed)}")
        
        # Display document statistics
        if st.session_state.uploaded_files_processed:
            st.write("### Document Statistics")
            st.write(f"Total documents: {len(st.session_state.uploaded_files_processed)}")
            st.write(f"Total chunks: {st.session_state.document_stats['total_chunks']}")
            st.write(f"Estimated tokens: {st.session_state.document_stats['total_tokens']:,}")
            
            # Show processed files with expandable details
            with st.expander("View processed documents"):
                for file_name in st.session_state.uploaded_files_processed:
                    if file_name in st.session_state.document_stats["file_details"]:
                        stats = st.session_state.document_stats["file_details"][file_name]
                        st.write(f"**{file_name}**")
                        st.write(f"- Chunks: {stats['chunks']}")
                        st.write(f"- Est. tokens: {stats['tokens']:,}")
                        st.write(f"- Characters: {stats['characters']:,}")
                    else:
                        st.write(f"- {file_name}")
                    st.write("---")

# Initialize Groq model if API key is provided
if groq_api_key:
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it")
else:
    st.warning("Please enter your Groq API key to continue.")
    st.stop()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🧑‍🎓" if message["role"] == "user" else "🤖"):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about the document..."):
    # Add user message to session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(prompt)
    
    # Process prompt if vector store exists
    if st.session_state.vector_store:
        with st.chat_message("assistant", avatar="🤖"):
            # Show which documents are being used
            if st.session_state.uploaded_files_processed:
                st.write(f"*Searching across {len(st.session_state.uploaded_files_processed)} document(s):* {', '.join(st.session_state.uploaded_files_processed)}")
            
            # Create RetrievalQA chain with more relevant documents
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=st.session_state.vector_store.as_retriever(search_kwargs={"k": 5}),  # Retrieve more relevant chunks
                chain_type_kwargs={"prompt": get_prompt_template(prompt)}
            )
            
            # Stream response with progress indicator
            response = ""
            message_placeholder = st.empty()
            
            for chunk in qa_chain.stream({"query": prompt}):
                chunk_text = chunk.get("result", "")
                response += chunk_text
                message_placeholder.markdown(response + "▌" if chunk_text else "Thinking...")
            
            # Final response without cursor
            message_placeholder.markdown(response)
            
            # Store response
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.last_result = response
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown("Please upload a document first.")

# Buttons for export, save, and reset
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Export Last Result as TXT"):
        if st.session_state.last_result:
            with open("last_result.txt", "w") as f:
                f.write(st.session_state.last_result)
            st.download_button(
                label="Download TXT",
                data=st.session_state.last_result,
                file_name="last_result.txt",
                mime="text/plain"
            )
with col2:
    if st.button("Save Chat History"):
        save_chat_history()
        st.success("Chat history saved as JSON!")
with col3:
    if st.button("Clear All Documents"):
        st.session_state.vector_store = None
        st.session_state.uploaded_files_processed = []
        st.session_state.document_stats = {"total_chunks": 0, "total_tokens": 0, "file_details": {}}
        st.success("All documents cleared! You can upload new documents now.")