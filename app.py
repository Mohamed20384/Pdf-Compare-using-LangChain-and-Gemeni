import os
import io
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
import google.generativeai as genai
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Configuration
class Config:
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    TOP_K_CHUNKS = 5
    EMBEDDING_BATCH_SIZE = 10
    MAX_PREVIEW_CHARS = 5000
    COMPARISON_TYPES = {
        "detailed": "Detailed Comparison",
        "summary": "Summary Comparison"
    }

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GENAI_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Cached resources
@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)

@st.cache_resource
def get_embeddings():
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

llm = get_llm()
embeddings = get_embeddings()

@st.cache_data
def extract_text_cached(file_bytes):
    reader = PdfReader(io.BytesIO(file_bytes))
    return "\n".join([page.extract_text() or "" for page in reader.pages])

def validate_pdf_text(text, filename):
    if not text.strip():
        raise ValueError(f"PDF {filename} appears to be empty or unreadable")
    return text

@st.cache_data
def get_top_k_chunks(text, k=Config.TOP_K_CHUNKS):
    splitter = CharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE, 
        chunk_overlap=Config.CHUNK_OVERLAP
    )
    chunks = splitter.split_text(text)
    if not chunks:
        return []
    
    vectors = []
    for i in range(0, len(chunks), Config.EMBEDDING_BATCH_SIZE):
        vectors.extend(embeddings.embed_documents(chunks[i:i+Config.EMBEDDING_BATCH_SIZE]))
    
    query_vec = embeddings.embed_query("ما هي أهم محتويات المستند؟")
    similarities = cosine_similarity([query_vec], vectors)[0]
    top_k_indices = similarities.argsort()[-k:][::-1]
    return [chunks[i] for i in top_k_indices]

def compare_docs_with_rag(text1, text2, comparison_type="detailed"):
    chunks1 = get_top_k_chunks(text1)
    chunks2 = get_top_k_chunks(text2)

    txt1 = "\n\n".join(chunks1)
    txt2 = "\n\n".join(chunks2)

    templates = {
        "detailed": """قارن بين محتوى المستندين التاليين باللغة العربية:
        
        المستند الأول:
        {txt1}
        
        المستند الثاني:
        {txt2}
        
        - ما أوجه التشابه بالتفصيل؟
        - ما أوجه الاختلاف؟
        - ما هي النقاط الرئيسية في كل مستند؟
        """,
        "summary": """قدم ملخصًا مقارنًا بين المستندين التاليين باللغة العربية:
        
        المستند الأول:
        {txt1}
        
        المستند الثاني:
        {txt2}
        
        في فقرتين:
        1. الملخص المشترك
        2. الفروقات الرئيسية
        """
    }
    
    prompt = templates.get(comparison_type, templates["detailed"]).format(
        txt1=txt1, txt2=txt2
    )
    
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Connection Error : {str(e)}"

def numeric_similarity(text1, text2):
    tfidf = TfidfVectorizer().fit_transform([text1, text2])
    sim_score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return sim_score

def show_file_preview(text, title):
    with st.expander(f"Preview: {title}"):
        st.text_area(
            "Extracted Text", 
            value=text[:Config.MAX_PREVIEW_CHARS] + ("..." if len(text) > Config.MAX_PREVIEW_CHARS else ""), 
            height=200,
            disabled=True
        )

# Streamlit UI
st.set_page_config(page_title="Arabic PDF Comparison", layout="wide")
st.markdown("""
    <style>
    .reportview-container .main .block-container {
        direction: ltr;
        text-align: left;
    }
    .rtl-output {
        direction: rtl;
        text-align: right;
        color: white;
        padding: 1rem;
        margin-top: 1rem;
        white-space: pre-wrap;
        line-height: 1.8;
    }
    .similarity-high {
        color: #4CAF50;
        font-weight: bold;
    }
    .similarity-medium {
        color: #FFC107;
        font-weight: bold;
    }
    .similarity-low {
        color: #F44336;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Title with Gemini logo
st.markdown("""
            <div style="display: flex; align-items: center; gap: 15px;">
                <h1 style="margin: 0;">Arabic PDF Comparison using Gemeni + RAG</h1>
                <img src="https://www.boundaryml.com/gemini.png" alt="Gemini Logo" width="100">
            </div>
        """, unsafe_allow_html=True)

# File upload
col1, col2 = st.columns(2)
with col1:
    pdf1 = st.file_uploader("Upload First PDF", type="pdf", key="pdf1")
with col2:
    pdf2 = st.file_uploader("Upload Second PDF", type="pdf", key="pdf2")

# Only show comparison options if both files are uploaded
if pdf1 and pdf2:
    comparison_type = st.radio(
        "Comparison Type",
        options=list(Config.COMPARISON_TYPES.keys()),
        format_func=lambda x: Config.COMPARISON_TYPES[x],
        horizontal=True
    )
    
    analyze_button = st.button("Analyze")
    
    if analyze_button:  # This is now properly scoped
        try:
            with st.spinner("Reading and analyzing files..."):
                text1 = validate_pdf_text(extract_text_cached(pdf1.read()), pdf1.name)
                text2 = validate_pdf_text(extract_text_cached(pdf2.read()), pdf2.name)
            
            # Show previews
            col1, col2 = st.columns(2)
            with col1:
                show_file_preview(text1, pdf1.name)
            with col2:
                show_file_preview(text2, pdf2.name)
            
            progress_bar = st.progress(0)
            with st.spinner("Comparing with Gemini..."):
                comparison = compare_docs_with_rag(text1, text2, comparison_type)
                progress_bar.progress(50)
            
            with st.spinner("Calculating similarity score..."):
                score = numeric_similarity(text1, text2)
                progress_bar.progress(100)
            
            st.success("Analysis complete!")
            
            # Display similarity score with color coding
            st.subheader("Similarity Score")
            similarity_class = "similarity-high" if score > 0.7 else "similarity-medium" if score > 0.4 else "similarity-low"
            st.markdown(f"""
                <div style="font-size: 1.2rem;">
                    Similarity: <span class="{similarity_class}">{score * 100:.2f}%</span><br>
                    Dissimilarity: <span class="{similarity_class}">{(1 - score) * 100:.2f}%</span>
                </div>
            """, unsafe_allow_html=True)
            
            st.subheader(f"Comparison ({Config.COMPARISON_TYPES[comparison_type]})")
            st.markdown(f"""<div class='rtl-output'><pre>{comparison}</pre></div>""", unsafe_allow_html=True)
            
            # Download options
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "Download Comparison Result", 
                    data=comparison, 
                    file_name=f"comparison_{pdf1.name.split('.')[0]}_vs_{pdf2.name.split('.')[0]}.txt"
                )
            with col2:
                st.download_button(
                    "Download Extracted Texts", 
                    data=f"Document 1:\n{text1}\n\nDocument 2:\n{text2}",
                    file_name=f"extracted_texts_{pdf1.name.split('.')[0]}_and_{pdf2.name.split('.')[0]}.txt"
                )
        
        except ValueError as ve:
            st.error(str(ve))
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")