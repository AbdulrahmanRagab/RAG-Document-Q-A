# 📚 RAG-Powered Document Q&A Assistant  

**A Streamlit-based AI chatbot that answers questions using your documents, powered by OpenAI, LangChain, and Hugging Face.**  

 📸 Demo: https://shorturl.at/ant6K

## 🌟 Features  

- **Document Processing**:  
  - Load and chunk PDFs from a directory using `PyPDFDirectoryLoader` and `RecursiveCharacterTextSplitter`.  
- **Vector Embeddings**:  
  - Generate embeddings with **Hugging Face** (`all-MiniLM-L6-v2`) and store them in **ChromaDB**.  
- **Retrieval-Augmented Generation (RAG)**:  
  - Retrieve context-aware answers using LangChain’s `create_retrieval_chain` and OpenAI’s `gpt-4o-mini`.  
- **Interactive UI**:  
  - Adjust model parameters (temperature, max tokens) via Streamlit.  
  - Inspect source documents used for answers with expandable sections.

## 📂 Project Structure  

```plaintext
your-repo/
├── material/          # Folder for PDFs (manually created)
├── chroma_db2/        # Auto-generated vector database
├── app.py             # Main Streamlit application
├── requirements.txt   # Dependencies
├── .env               # API key configuration
└── README.md          # This file
```

## 🚀 Usage  

1. **Add PDFs**:  
   Place your documents in the `material/` folder.  

2. **Run the app**:  
   ```bash
   streamlit run app.py
   ```

3. **Embed documents**:  
   - Click the *"Document Embedding"* button to generate vector embeddings.  

4. **Ask questions**:  
   - Type your question in the input box and press Enter.  
   - The bot will retrieve relevant document snippets and generate an answer.  



## 🔧 Dependencies  

- Python 3.8+  
- Libraries:  
  ```plaintext
  streamlit
  openai
  langchain
  langchain-openai
  chromadb
  python-dotenv
  huggingface-hub
  ```

## 🛠️ Installation  

1. **Clone the repository**:  
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
   ```

2. **Set up a virtual environment (recommended)**:  
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**:  
   ```bash
   pip install -r requirements.txt
   ```

4. **Add your OpenAI API key**:  
   Create a `.env` file in the root directory and add:  
   ```
   OPENAI_API_KEY="your-api-key-here"
   ```


## 📌 Future Improvements  

- [ ] Add support for **FAISS** as an alternative vector store.  
- [ ] Extend file formats (Word, HTML, plain text).  
- [ ] Deploy as a Docker container or Hugging Face Space.  

## 🤝 Contributing  

Pull requests are welcome! For major changes, open an issue first.  
