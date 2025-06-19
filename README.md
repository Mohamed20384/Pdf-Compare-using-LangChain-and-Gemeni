# 📄 Arabic PDF Comparison Tool

A modern Streamlit web application that enables **automated comparison between two Arabic PDF documents**. Leveraging the power of **Google Gemini** and **LangChain’s RAG architecture**, this tool performs both **semantic comparisons** and **numerical similarity analysis** using **TF-IDF cosine similarity**.

The app provides users with insights into:
- Shared and differing content,
- A human-readable comparison in Arabic,
- And a measurable similarity score — all with a simple interface and RTL support.

---

## 🚀 Key Features

✅ Upload and preview two Arabic PDF documents  
🧠 Generate AI-based comparison using **Gemini 2.5 (flash)** with **retrieval-augmented generation (RAG)**  
📊 Compute text similarity score using **TF-IDF & cosine similarity**  
📤 Download AI comparison results and extracted document texts  
🎨 Full support for **Arabic language** and **RTL formatting**  
🔁 Choose between **detailed** and **summary** comparison styles  
⚡ Fast performance with caching and chunked embedding strategy

---

## 🧱 Tech Stack Overview

| Technology | Role |
|------------|------|
| [Streamlit](https://streamlit.io/) | Frontend + backend interface |
| [Google Generative AI](https://ai.google.dev/) | Natural language comparison engine (Gemini) |
| [LangChain](https://www.langchain.com/) | Chunking, retrieval, embedding pipeline |
| [GoogleGenerativeAIEmbeddings](https://github.com/langchain-ai/langchain-google-genai) | Embedding model for document chunks |
| [PyPDF2](https://pypi.org/project/PyPDF2/) | Extracting selectable text from PDF pages |
| [scikit-learn](https://scikit-learn.org/) | TF-IDF and cosine similarity computation |
| [python-dotenv](https://pypi.org/project/python-dotenv/) | Environment variable management |

---

## 📦 Installation & Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/arabic-pdf-comparator.git
cd arabic-pdf-comparator

.
├── app.py                  # Main Streamlit application
├── requirements.txt        # All Python dependencies
├── .env                    # Your Google API key (not committed)
└── README.md               # Project documentation
