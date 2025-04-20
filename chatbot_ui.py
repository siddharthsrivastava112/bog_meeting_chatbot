import streamlit as st
from rag_query_handler import TextRAGHandler

# App configuration
st.set_page_config(
    page_title="MNNIT PG Ordinance Assistant",
    page_icon="📚",
    layout="centered"
)

@st.cache_resource
def load_rag_handler():
    return TextRAGHandler()

# Initialize RAG handler
rag_handler = load_rag_handler()

# UI Components
st.title("MNNIT Allahabad Postgraduate Ordinance Assistant")
st.markdown("""
Ask questions about:
- PhD/M.Tech admission criteria
- Course duration requirements
- Academic regulations
""")

# Query input
user_query = st.text_area(
    "Enter your question about MNNIT's PG Ordinance:",
    placeholder="E.g. What is the minimum CGPA required for PhD admission?",
    height=150
)

# Search button
if st.button("Search Ordinance", type="primary"):
    if not user_query.strip():
        st.warning("Please enter a valid question")
    else:
        with st.spinner("Searching official documents..."):
            response = rag_handler.handle_input(user_query)
            
            st.subheader("Official Answer")
            if response.startswith("Error:") or response.startswith("System error:"):
                st.error(response)
            else:
                st.markdown(response)
                
            st.markdown("---")
            st.caption("Disclaimer: Responses are generated from official MNNIT documents")

# Sidebar
with st.sidebar:
    st.markdown("### About")
    st.markdown("""
    This assistant answers questions using MNNIT Allahabad's:
    - Postgraduate Ordinance documents
    - Official academic regulations
    """)
    st.markdown("---")
    st.markdown("**Note:** For official confirmation, always refer to the original ordinance documents.")