import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import fitz  # PyMuPDF
from io import BytesIO

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()
st.set_page_config(layout="wide", page_title="JD & CV Matcher")

st.title("🤖 Match JD file – CV PDF – LinkedIn Profile")
st.markdown("Upload JD (PDF or Word), CVs (PDF), hoặc paste LinkedIn profiles.")

# Upload JD file
jd_file = st.file_uploader("📄 Upload JD file (.pdf or .docx)", type=["pdf", "docx"])
if jd_file:
    if jd_file.name.endswith(".pdf"):
        with fitz.open(stream=jd_file.read(), filetype="pdf") as doc:
            jd_text = "\n".join(page.get_text() for page in doc)
    else:  # docx
        import docx
        jd_text = "\n".join(para.text for para in docx.Document(BytesIO(jd_file.read())).paragraphs)
else:
    jd_text = st.text_area("📝 Hoặc nhập JD thủ công", height=200)

# Upload multiple CV PDFs
uploaded_cvs = st.file_uploader("📁 Upload CV PDF (có thể nhiều)", type="pdf", accept_multiple_files=True)

# Paste LinkedIn profiles
linkedin_input = st.text_area(
    "🧑‍💼 Hoặc paste nhiều LinkedIn profiles (cách nhau bằng dòng trống)",
    height=200
)

def extract_text_from_pdf(f):
    txt = ""
    with fitz.open(stream=f.read(), filetype="pdf") as doc:
        txt = "".join(page.get_text() for page in doc)
    return txt

if st.button("🔍 Check Match"):
    if not jd_text.strip():
        st.warning("⚠️ Vui lòng cung cấp JD.")
    else:
        jd_vec = model.encode([jd_text])[0]
        results = []

        # Match CV PDFs
        for f in uploaded_cvs:
            try:
                cv_txt = extract_text_from_pdf(f)
                score = cosine_similarity([model.encode([cv_txt])[0]], [jd_vec])[0][0]
                results.append({"Nguồn": f"📄 {f.name}", "Match %": round(score*100,2)})
            except:
                results.append({"Nguồn": f"📄 {f.name}", "Match %": "Error"})

        # Match pasted profiles
        profiles = [p.strip() for p in linkedin_input.split("\n\n") if p.strip()]
        for idx, prof in enumerate(profiles):
            score = cosine_similarity([model.encode([prof])[0]], [jd_vec])[0][0]
            results.append({"Nguồn": f"🧑 Profile #{idx+1}", "Match %": round(score*100,2)})

        st.markdown("### 📊 Kết quả match:")
        st.dataframe(results, use_container_width=True)
