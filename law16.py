import os
import re
import uuid
import time
import hashlib

import fitz  # PyMuPDF
import streamlit as st
import chromadb
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ================= إعداد الصفحة =================
st.set_page_config(
    page_title="⚖️ فقه الذكاء: باحث قانوني + مولّد احترافي",
    layout="wide"
)
st.title("⚖️ law_hnsw مساعدك الذكي للاستشارات القانونية")

# جعل الواجهة باتجاه عربي (يمين → يسار)
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

chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=None
)

def get_chroma_index_info(db_path: str = DB_PATH):
    """إرجاع عدد المتجهات وحجم قاعدة البيانات."""
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
        normalize_embeddings=True
    )
    return enc.tolist()


# ================= تحميل نموذج التوليد =================
@st.cache_resource
def load_llm():
    try:
        model_name = "aubmindlab/aragpt2-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return tokenizer, model
    except Exception as e:
        st.error(f"حدث خطأ في تحميل نموذج التوليد: {str(e)}")
        return None, None

tokenizer, llm_model = load_llm()


# ================= توليد الإجابة =================
def generate_short_answer(context, question, references, max_new_tokens=150):
    if not tokenizer or not llm_model:
        return "النموذج غير متاح حاليًا، يرجى التحقق من الإعدادات."

    try:
        start_time = time.time()

        prompt = f"""
السؤال: {question}

النصوص القانونية ذات الصلة:
{context}

المطلوب: أعطني إجابة قانونية مختصرة ومباشرة لا تتجاوز 4 أسطر،
مع ذكر رقم المادة واسم القانون أو اللائحة عند الاقتضاء.
"""

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="max_length"
        )

        outputs = llm_model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id
        )

        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if "المطلوب:" in answer:
            answer = answer.split("المطلوب:")[-1].strip()

        elapsed_time = round(time.time() - start_time, 2)

        formatted_refs = []
        for ref in references:
            match = re.search(r"(?:المادة|مادة)\s*(\d+)", ref["doc"])
            article_no = match.group(1) if match else "غير محددة"
            source_name = ref["source"].replace(".pdf", "")
            formatted_refs.append(f"المادة ({article_no}) – {source_name}")

        refs_text = ""
        if formatted_refs:
            refs_text = "\n\n📖 المراجع:\n" + "\n".join(f"- {r}" for r in formatted_refs)

        return f"{answer.strip()}\n\n⏳ زمن استجابة النموذج: {elapsed_time} ثانية{refs_text}"

    except Exception as e:
        st.error(f"خطأ في توليد الإجابة: {str(e)}")
        return "تعذر توليد الإجابة، يرجى إعادة المحاولة."


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
                        if current_article and len(current_article) > 50:
                            current_article = clean_legal_text(current_article)
                            article_hash = hashlib.md5(current_article.encode("utf-8")).hexdigest()

                            if article_hash not in seen_articles:
                                articles.append(current_article)
                                seen_articles.add(article_hash)

                        current_article = line
                    else:
                        current_article += " " + line

            if current_article and len(current_article) > 50:
                current_article = clean_legal_text(current_article)
                article_hash = hashlib.md5(current_article.encode("utf-8")).hexdigest()

                if article_hash not in seen_articles:
                    articles.append(current_article)
                    seen_articles.add(article_hash)

        return articles

    except Exception as e:
        st.sidebar.error(f"خطأ في معالجة الملف: {str(e)}")
        return []


# ================= تخزين التضمينات في Chroma =================
def embed_and_store(articles, source_name=""):
    if not articles:
        return 0

    total_added = 0
    try:
        chunks = []

        for article in articles:
            article = clean_legal_text(article)
            article_chunks = text_splitter.split_text(article)

            for chunk in article_chunks:
                chunk = clean_legal_text(chunk)
                if len(chunk) > 30:
                    chunks.append(chunk)

        if not chunks:
            return 0

        embeddings = embed_texts(chunks, is_query=False)
        ids = [str(uuid.uuid4()) for _ in chunks]

        collection.add(
            documents=chunks,
            embeddings=embeddings,
            metadatas=[{"source": source_name}] * len(chunks),
            ids=ids
        )

        total_added = len(chunks)

    except Exception as e:
        st.sidebar.error(f"خطأ في تخزين البيانات في Chroma: {str(e)}")

    return total_added


# ================= واجهة تحميل الملفات =================
st.sidebar.header("📤 تحميل ملفات PDF قانونية")
uploaded_files = st.sidebar.file_uploader(
    "اختر ملفات PDF",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    if st.sidebar.button("➕ إضافة جميع الملفات إلى قاعدة المعرفة"):
        with st.spinner("🚀 جاري معالجة الملفات..."):
            total_all = 0
            for file in uploaded_files:
                st.sidebar.write(f"📄 معالجة: {file.name}")
                articles = extract_articles_from_pdf(file)
                count = embed_and_store(articles, source_name=file.name)

                if count > 0:
                    st.sidebar.success(f"✅ تمت إضافة {count} قطعة من {file.name}")
                    total_all += count

            if total_all > 0:
                st.sidebar.success(f"✅ تمت إضافة {total_all} قطعة من {len(uploaded_files)} ملف")


# ================= إدارة قاعدة البيانات =================
st.sidebar.header("⚙️ إدارة قاعدة البيانات / الفهرس")

if st.sidebar.button("🔁 HNSW: إعادة بناء الفهرس"):
    try:
        try:
            chroma_client.delete_collection(name=COLLECTION_NAME)
        except Exception:
            pass

        collection = chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=None
        )
        st.sidebar.success("✅ تمت إعادة بناء فهرس HNSW. يرجى إعادة إضافة الملفات.")
    except Exception as e:
        st.sidebar.error(f"خطأ في إعادة بناء الفهرس: {str(e)}")

if st.sidebar.button("🔍 عرض معلومات الفهرس"):
    vector_count, size_mb = get_chroma_index_info()
    st.sidebar.info(
        f"📊 عدد المتجهات المخزنة (نقاط في فهرس HNSW): **{vector_count if vector_count is not None else 'غير متاح'}**\n\n"
        f"💾 حجم قاعدة البيانات (تقريبًا): **{size_mb} MB**"
    )

try:
    doc_count = collection.count()
    st.sidebar.info(f"ℹ️ قاعدة البيانات تحتوي حاليًا على **{doc_count}** وثيقة/متجه")
except Exception:
    st.sidebar.info("ℹ️ قاعدة البيانات غير متاحة حاليًا.")


# ================= حفظ زمن الاسترجاع السابق =================
if "last_retrieval_time" not in st.session_state:
    st.session_state["last_retrieval_time"] = None


# ================= البحث والتوليد =================
st.markdown("---")
query = st.text_area("🧠 اكتب سؤالك القانوني هنا:", height=100, key="query_input")

if query:
    with st.spinner("🔍 جاري البحث والتوليد..."):
        try:
            retrieval_start = time.time()
            query_embedding = embed_texts([query], is_query=True)

            results = collection.query(
                query_embeddings=query_embedding,
                n_results=5,
                include=["documents", "metadatas", "distances"]
            )

            retrieval_time = round(time.time() - retrieval_start, 3)

            prev_time = st.session_state.get("last_retrieval_time")
            st.session_state["last_retrieval_time"] = retrieval_time

            if results["documents"] and results["documents"][0]:
                unique_docs = {}

                for i, doc in enumerate(results["documents"][0]):
                    doc_hash = hashlib.md5(doc.encode()).hexdigest()
                    if doc_hash not in unique_docs:
                        unique_docs[doc_hash] = {
                            "doc": doc,
                            "source": results["metadatas"][0][i]["source"],
                            "distance": results["distances"][0][i]
                        }

                sorted_docs = sorted(unique_docs.values(), key=lambda x: x["distance"])
                context_texts = "\n\n".join(d["doc"] for d in sorted_docs)

                st.markdown("### 📄 المواد القانونية ذات الصلة:")
                for i, doc in enumerate(sorted_docs):
                    similarity_score = round(1 - doc["distance"], 2)
                    with st.expander(
                        f"🔹 القطعة #{i+1} (درجة التشابه: {similarity_score}) | المصدر: {doc['source']}"
                    ):
                        st.write(doc["doc"])

                if prev_time is not None:
                    st.info(
                        f"⏱️ زمن الاسترجاع الحالي من Chroma (HNSW + Cosine): **{retrieval_time} ثانية** "
                        f"(الزمن السابق: {prev_time} ثانية)"
                    )
                else:
                    st.info(f"⏱️ زمن الاسترجاع من Chroma (HNSW + Cosine): **{retrieval_time} ثانية**")

                answer = generate_short_answer(context_texts, query, sorted_docs)

                st.markdown("### 📝 الإجابة القانونية المُولدة:")

                answer_html = answer.replace("\n", "<br>")
                st.markdown(
                    f"""
                    <div style="direction: rtl; text-align: right; background-color: #f0fff4;
                                border-radius: 8px; padding: 10px; border: 1px solid #d4ecd8;">
                        {answer_html}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.warning("⚠️ لم يتم العثور على مواد قانونية ذات صلة بالسؤال.")

        except Exception as e:
            st.error(f"حدث خطأ أثناء البحث أو التوليد: {str(e)}")
else:
    st.info("ℹ️ اكتب سؤالًا قانونيًا للحصول على إجابة مستندة إلى التشريعات.")