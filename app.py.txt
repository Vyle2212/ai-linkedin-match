import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import fitz  # PyMuPDF for PDF text extraction

# Load embedding model once
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

st.set_page_config(page_title="📄 Multi-CV Matcher", layout="wide")
st.title("🤖 Match nhiều CV PDF với 1 JD")

# Upload multiple CV PDFs
uploaded_files = st.file_uploader("📁 Chọn nhiều file PDF CV", type="pdf", accept_multiple_files=True)

# Enter Job Description
jd_text = st.text_area("📝 Nhập Job Description", height=200)

def extract_text_from_pdf(uploaded_file):
    text = ""
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

if st.button("🔍 Check Match"):
    if not uploaded_files or not jd_text.strip():
        st.warning("⚠️ Vui lòng upload ít nhất 1 CV PDF và nhập JD.")
    else:
        jd_vec = model.encode([jd_text])[0]
        results = []
        for file in uploaded_files:
            try:
                cv_text = extract_text_from_pdf(file)
                cv_vec = model.encode([cv_text])[0]
                score = cosine_similarity([cv_vec], [jd_vec])[0][0]
                results.append({"Tên file": file.name, "Match %": round(score * 100, 2)})
            except Exception:
                results.append({"Tên file": file.name, "Match %": "❌ Error đọc PDF"})
        st.markdown("### 📊 Kết quả so sánh:")
        st.dataframe(results, use_container_width=True)
