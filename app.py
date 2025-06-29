import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import fitz  # PyMuPDF
import docx
from io import BytesIO
from transformers import pipeline

# Load models: embedding và summarizer
@st.cache_resource
def load_models():
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    summarizer = pipeline("summarization", model="t5-base", tokenizer="t5-base")
    return embed_model, summarizer

embed_model, summarizer = load_models()

st.set_page_config(layout="wide", page_title="JD & CV Matcher with Summary")
st.title("🤖 Match & Summary: JD ↔ CV PDF / LinkedIn Profile")

# Upload JD file or input manually
jd_file = st.file_uploader("📄 Upload JD (.pdf or .docx)", type=["pdf", "docx"])
if jd_file:
    if jd_file.name.endswith(".pdf"):
        jd_text = "\n".join(page.get_text() for page in fitz.open(stream=jd_file.read(), filetype="pdf"))
    else:
        jd_text = "\n".join([para.text for para in docx.Document(BytesIO(jd_file.read())).paragraphs])
else:
    jd_text = st.text_area("📝 Or paste Job Description manually", height=200)

# Upload multiple CV PDFs
uploaded_cvs = st.file_uploader("📁 Upload multiple CV PDF files", type="pdf", accept_multiple_files=True)

# Paste LinkedIn profiles
linkedin_input = st.text_area(
    "🧑‍💼 Or paste multiple LinkedIn profiles (separate by empty line)",
    height=200
)

def extract_text_from_pdf(f):
    return "\n".join(page.get_text() for page in fitz.open(stream=f.read(), filetype="pdf"))

def summarize_text(text):
    short = text[:1000]
    try:
        return summarizer("summarize: " + short, max_length=50, min_length=10, do_sample=False)[0]['summary_text']
    except:
        return "⚠️ Summary failed"

if st.button("🔍 Run Match & Summarize"):
    if not jd_text.strip():
        st.warning("⚠️ Please provide a Job Description.")
    else:
        jd_vec = embed_model.encode([jd_text])[0]
        results = []

        # Process CV PDFs
        for f in uploaded_cvs:
            txt = extract_text_from_pdf(f)
            score = cosine_similarity([embed_model.encode([txt])[0]], [jd_vec])[0][0]
            summary = summarize_text(txt)
            results.append({"Nguồn": f"📄 {f.name}", "Match %": round(score * 100, 2), "Brief Summary": summary})

        # Process LinkedIn texts
        profiles = [p.strip() for p in linkedin_input.split("\n\n") if p.strip()]
        for i, prof in enumerate(profiles):
            score = cosine_similarity([embed_model.encode([prof])[0]], [jd_vec])[0][0]
            summary = summarize_text(prof)
            results.append({"Nguồn": f"🧑 Profile #{i+1}", "Match %": round(score * 100, 2), "Brief Summary": summary})

        st.markdown("### 📊 Matching Results with Brief Summary")
        st.dataframe(results, use_container_width=True)
