# 📄 Arabic PDF Comparison Tool

A Streamlit-powered app that compares **two Arabic PDF documents** using both **AI-based natural language comparison (Gemini with RAG)** and **numeric similarity (TF-IDF cosine similarity)**. The tool highlights **similarities, differences**, and computes a **percentage match** with optional download support.

---

## 🚀 Features

- ✅ Upload **two Arabic PDF files**
- 🧠 Compare content using **Google Gemini** with **LangChain RAG**
- 📊 Calculate **similarity score** using **TF-IDF + cosine similarity**
- 📄 View extracted text previews
- 🧾 Download comparison results and raw extracted text
- 🎨 Right-to-left (RTL) Arabic support
- 🔁 Choose between **detailed** and **summary** comparisons

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| [Streamlit](https://streamlit.io/) | Web app framework |
| [Google Generative AI (Gemini)](https://ai.google.dev/) | Natural language document comparison |
| [LangChain](https://www.langchain.com/) | Text chunking and retrieval |
| [PyPDF2](https://pypi.org/project/PyPDF2/) | PDF text extraction |
| [scikit-learn](https://scikit-learn.org/) | TF-IDF vectorization and cosine similarity |
| [dotenv](https://pypi.org/project/python-dotenv/) | Secure API key loading |

---

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/arabic-pdf-comparator.git
   cd arabic-pdf-comparator

