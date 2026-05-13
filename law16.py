from pathlib import Path

code = r'''import os
import re
import uuid
import time
import hashlib

import fitz  # PyMuPDF
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ================= إعداد الصفحة =================
st.set_page_config(
    page_title="⚖️ فقه الذكاء: باحث قانوني",
    layout="wide"
)

st.title("⚖️ law_hnsw مساعدك الذكي للاستشارات القانونية")

# جعل الواجهة باتجاه عربي
st.markdown("""
<style>
  .main .block-container {
      direction: rtl;
      text-align: right;
  }
  .stTextArea textarea {
      direction: rtl;
      text-align: right;
  }
  .stMarkdown, .stSuccess, .stWarning, .stInfo {
      direction: rtl;
      text-align: right;
  }
</style>
""", unsafe_allow_html=True)


# ================= إعداد ChromaDB =================
DB_PATH = os.path.abspath("./legal_chroma_db50122")
COLLECTION_NAME = "legal_docs"
os.makedirs(DB_PATH, exist_ok=True)

@st.cache_resource
def load_chroma():
    client = chromadb.PersistentClient(path=DB_PATH)
    coll = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=None
    )
    return client, coll

chroma_client, collection = load_chroma()


def get_chroma_index_info(db_path: str = DB_PATH):
    total_size_bytes = 0

    for root, dirs, files in os.walk(db_path):
        for f in files:
            fp = os.path.join(root, f)
            try:
                total_size_bytes += os.path.getsize(fp)
            except OSError:
                pass

    size_mb = round(total_size_bytes / (1024 * 1024), 2)

    try:
        vector_count = collection.count()
    except Exception:
        vector_count = None

    return vector_count, size_mb


# ================= تنظيف النصوص القانونية العربية =================
def normalize_arabic_text(text: str) -> str:
    if not text:
        return ""

    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = text.replace("ى", "ي").replace("ؤ", "و").replace("ئ", "ي")
    text = re.sub(r"[ـ]+", "", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def clean_long_token_sequence(seq: str) -> str:
    tokens = re.split(r'\s*[،,]\s*|\s+', seq)
    tokens = [t for t in tokens if t.strip()]

    seen = []
    for t in tokens:
        if t not in seen:
            seen.append(t)

    return "، ".join(seen[:6])


def remove_repeated_phrases(text: str, max_repeat: int = 2) -> str:
    if not text:
        return ""

    words = text.split()
    cleaned = []
    repeat_count = 0

    for i, word in enumerate(words):
        if i > 0 and word == words[i - 1]:
            repeat_count += 1
            if repeat_count < max_repeat:
                cleaned.append(word)
        else:
            repeat_count = 0
            cleaned.append(word)

    text = " ".join(cleaned)
    text = re.sub(r'\b([\u0600-\u06FF]+)\b(?:\s*[،,]\s*\1\b)+', r'\1', text)

    return text.strip()


def remove_noisy_sequences(text: str) -> str:
    if not text:
        return ""

    text = re.sub(
        r'((?:\b[\u0600-\u06FF]{2,}\b\s*[،,]?\s*){6,})',
        lambda m: clean_long_token_sequence(m.group(0)),
        text
    )

    return text.strip()


def remove_page_noise(text: str) -> str:
    if not text:
        return ""

    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        line = line.strip()

        if not line:
            continue

        if re.fullmatch(r"\d+", line):
            continue

        if len(line) <= 2:
            continue

        if any(x in line for x in [
            "اللايحة الادارية",
            "اللائحة الادارية",
            "وزارة",
            "الاكاديمية",
            "أكاديمية",
            "كلية",
            "صفحة"
        ]) and len(line) < 80:
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def clean_legal_text(text: str) -> str:
    if not text:
        return ""

    text = remove_page_noise(text)
    text = normalize_arabic_text(text)
    text = re.sub(r'[^\w\s\u0600-\u06FF:،\-\.\(\)/\n]', ' ', text)
    text = remove_repeated_phrases(text)
    text = remove_noisy_sequences(text)
    text = re.sub(r'\s*،\s*', '، ', text)
    text = re.sub(r'\s*-\s*', ' - ', text)
    text = re.sub(r'[ ]{2,}', ' ', text)

    return text.strip()


# ================= إعداد Text Splitter =================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=120,
    separators=[r"مادة\s*\d+", r"الفصل", r"الباب", "\n\n", "\n", " "],
    is_separator_regex=True
)


# ================= تحميل نموذج E5 للتضمين =================
@st.cache_resource
def load_sbert():
    return SentenceTransformer("intfloat/multilingual-e5-small")


model_embed = load_sbert()


def embed_texts(texts, is_query: bool = False):
    prefix = "query: " if is_query else "passage: "

    enc = model_embed.encode(
        [prefix + t for t in texts],
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return enc.tolist()


# ================= إجابة بدون LLM ثقيل =================
def generate_retrieval_answer(sorted_docs, question, retrieval_time):
    if not sorted_docs:
        return "لم يتم العثور على نصوص قانونية ذات صلة بالسؤال."

    best_doc = sorted_docs[0]["doc"]
    source = sorted_docs[0]["source"]

    match = re.search(r"(?:المادة|مادة)\s*(\d+)", best_doc)
    article_no = match.group(1) if match else "غير محددة"

    answer = f"""
بناءً على أقرب نص قانوني مسترجع من قاعدة المعرفة، فإن الإجابة مرتبطة بالمادة ({article_no}) من المصدر: {source}.

النص القانوني الأقرب:
{best_doc}

زمن الاسترجاع: {retrieval_time} ثانية.
"""

    return answer.strip()


# ================= استخراج المواد القانونية من PDF =================
def extract_articles_from_pdf(file):
    articles = []
    current_article = ""
    article_pattern = re.compile(r"^(?:المادة|مادة|Article|article)\s*\d+", re.IGNORECASE)
    seen_articles = set()

    try:
        file_bytes = file.read()

        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            for page in doc:
                text = page.get_text("text")
                text = remove_page_noise(text)

                lines = text.split("\n")

                for line in lines:
                    line = line.strip()

                    if not line:
                        continue

                    line = clean_legal_text(line)

                    if article_pattern.match(line):
