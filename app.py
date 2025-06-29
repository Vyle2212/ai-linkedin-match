st.markdown("## 📄 Upload JD file (PDF hoặc Word)")
jd_file = st.file_uploader("Upload Job Description (.pdf or .docx)", type=["pdf", "docx"])

def read_jd_file(file):
    if file.name.endswith(".pdf"):
        with fitz.open(stream=file.read(), filetype="pdf") as doc:
            return "\n".join([page.get_text() for page in doc])
    elif file.name.endswith(".docx"):
        from docx import Document
        document = Document(file)
        return "\n".join([para.text for para in document.paragraphs])
    else:
        return ""

# Nếu có file thì ưu tiên đọc JD từ file
jd_text = ""
if jd_file:
    jd_text = read_jd_file(jd_file)
else:
    jd_text = st.text_area("📝 Hoặc nhập Job Description tại đây", height=200)
