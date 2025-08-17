import streamlit as st, pdfplumber, io

st.title("PDF sanity check")
f = st.file_uploader("Upload a PDF", type=["pdf"])
if f:
    with pdfplumber.open(io.BytesIO(f.read())) as pdf:
        st.write(f"Pages detected: {len(pdf.pages)}")
        st.text((pdf.pages[0].extract_text() or "")[:800])