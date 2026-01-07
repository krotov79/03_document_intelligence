import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Document Intelligence", layout="centered")

st.title("📄 Document Intelligence")
st.write("Text classification + Named Entity Recognition")

text = st.text_area(
    "Enter text",
    height=150,
    placeholder="EU rejects German call to boycott British lamb...",
)

if st.button("Analyze") and text.strip():
    with st.spinner("Analyzing..."):
        # ---- Classification ----
        cls_resp = requests.post(
            f"{API_URL}/classify",
            json={"text": text},
            timeout=10,
        )
        cls_data = cls_resp.json()

        # ---- NER ----
        ner_resp = requests.post(
            f"{API_URL}/ner",
            json={"text": text},
            timeout=10,
        )
        ner_data = ner_resp.json()

    st.subheader("📌 Classification")
    st.write(
        f"**Label:** {cls_data['label']}  \n"
        f"**Confidence:** {cls_data['confidence']:.2f}"
    )

    st.subheader("🔍 Named Entities")
    if ner_data["entities"]:
        for ent in ner_data["entities"]:
            st.markdown(
                f"- **{ent['text']}** → `{ent['label']}` "
                f"(score: {ent['score']:.2f})"
            )
    else:
        st.write("No entities found.")
