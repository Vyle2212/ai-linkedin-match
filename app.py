import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache(allow_output_mutation=True)
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

st.title("🤖 AI Matching: LinkedIn Profile vs Job Description")
st.write("Điền hồ sơ LinkedIn và JD để xem mức độ phù hợp (%)")

profile_input = st.text_area("🧑 LinkedIn Profile", height=200)
jd_input = st.text_area("📄 Job Description", height=200)

if st.button("🔍 Check Match"):
    if profile_input and jd_input:
        p_vec = model.encode([profile_input])[0]
        j_vec = model.encode([jd_input])[0]
        score = cosine_similarity([p_vec], [j_vec])[0][0]
        st.success(f"✅ Match Score: **{round(score*100,2)}%**")
    else:
        st.warning("⚠️ Vui lòng điền cả hai trường dữ liệu.")
