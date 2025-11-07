"""
IHRA Chatbot - Advanced Version with File Upload
Supports TXT, PDF, and DOCX files
Fixed for Streamlit Cloud deployment
"""

import streamlit as st
import os
import openai
from typing import List, Dict
import chromadb
import PyPDF2
from docx import Document as DocxDocument
import tempfile

# OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Page config
st.set_page_config(
    page_title="IHRA Assistant Pro",
    page_icon="🏥",
    layout="wide"
)

class AdvancedIHRAChatbot:
    def __init__(self):
        """Initialize chatbot with vector database"""
        # Use ephemeral client for Streamlit Cloud (in-memory)
        self.client = chromadb.EphemeralClient()
        
        self.collection = self.client.get_or_create_collection(
            name="ihra_documents_advanced",
            metadata={"hnsw:space": "cosine"}
        )
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            return f"Error reading PDF: {str(e)}"
    
    def extract_text_from_docx(self, docx_file) -> str:
        """Extract text from DOCX file"""
        try:
            doc = DocxDocument(docx_file)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text
        except Exception as e:
            return f"Error reading DOCX: {str(e)}"
    
    def load_document(self, uploaded_file) -> str:
        """Load document from uploaded file"""
        file_type = uploaded_file.name.split('.')[-1].lower()
        
        if file_type == 'txt':
            return uploaded_file.getvalue().decode('utf-8')
        elif file_type == 'pdf':
            return self.extract_text_from_pdf(uploaded_file)
        elif file_type == 'docx':
            return self.extract_text_from_docx(uploaded_file)
        else:
            return "Unsupported file type"
    
    def reset_database(self):
        """Clear the vector database"""
        try:
            self.client.delete_collection("ihra_documents_advanced")
            self.collection = self.client.get_or_create_collection(
                name="ihra_documents_advanced",
                metadata={"hnsw:space": "cosine"}
            )
            return True
        except Exception as e:
            st.error(f"Error resetting database: {str(e)}")
            return False
    
    def _chunk_document(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
        """Split document into chunks"""
        chunks = []
        lines = text.split('\n')
        current_chunk = ""
        chunk_id = 0
        
        for line in lines:
            if len(current_chunk) + len(line) < chunk_size:
                current_chunk += line + "\n"
            else:
                if current_chunk.strip():
                    chunks.append({
                        "id": f"chunk_{chunk_id}",
                        "text": current_chunk.strip(),
                        "metadata": {"chunk_id": chunk_id}
                    })
                    chunk_id += 1
                
                words = current_chunk.split()
                overlap_text = " ".join(words[-overlap:]) if len(words) > overlap else ""
                current_chunk = overlap_text + " " + line + "\n"
        
        if current_chunk.strip():
            chunks.append({
                "id": f"chunk_{chunk_id}",
                "text": current_chunk.strip(),
                "metadata": {"chunk_id": chunk_id}
            })
        
        return chunks
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding from OpenAI"""
        try:
            response = openai.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            st.error(f"Embedding error: {str(e)}")
            raise
    
    def store_document(self, document_text: str, progress_callback=None):
        """Store document in vector database"""
        chunks = self._chunk_document(document_text)
        
        total_chunks = len(chunks)
        for i, chunk in enumerate(chunks):
            embedding = self._get_embedding(chunk["text"])
            
            self.collection.add(
                embeddings=[embedding],
                documents=[chunk["text"]],
                ids=[chunk["id"]],
                metadatas=[chunk["metadata"]]
            )
            
            if progress_callback:
                progress_callback(i + 1, total_chunks)
        
        return len(chunks)
    
    def _retrieve_relevant_chunks(self, query: str, top_k: int = 5) -> List[str]:
        """Retrieve relevant chunks"""
        if self.collection.count() == 0:
            return []
        
        query_embedding = self._get_embedding(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count())
        )
        
        return results['documents'][0] if results['documents'] else []
    
    def chat(self, user_query: str, conversation_history: List[Dict] = None, 
             model: str = "gpt-4o-mini", temperature: float = 0.7) -> str:
        """Chat with RAG"""
        if self.collection.count() == 0:
            return "⚠️ No documents loaded. Please upload a document first."
        
        relevant_chunks = self._retrieve_relevant_chunks(user_query, top_k=5)
        
        if not relevant_chunks:
            return "⚠️ Could not find relevant information in the document."
        
        context = "\n\n".join([f"[Context {i+1}]:\n{chunk}" 
                               for i, chunk in enumerate(relevant_chunks)])
        
        messages = [
            {
                "role": "system",
                "content": """You are an intelligent assistant for the Islamabad Healthcare Regulatory Authority (IHRA). 

Provide accurate, helpful, and professional responses based on the provided context.
If information is not in the context, clearly state that.
Use bullet points and formatting for clarity."""
            }
        ]
        
        if conversation_history:
            messages.extend(conversation_history[-10:])  # Last 10 messages
        
        messages.append({
            "role": "user",
            "content": f"""Context from documents:
{context}

Question: {user_query}

Please answer based on the context provided."""
        })
        
        try:
            response = openai.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=1500
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            return f"❌ Error: {str(e)}"


# Initialize session state
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = AdvancedIHRAChatbot()

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'document_loaded' not in st.session_state:
    st.session_state.document_loaded = False

if 'chunk_count' not in st.session_state:
    st.session_state.chunk_count = 0


# Sidebar
with st.sidebar:
    st.title("🏥 IHRA Assistant Pro")
    st.markdown("---")
    
    # API Key warning
    if not os.getenv("OPENAI_API_KEY"):
        st.error("⚠️ OPENAI_API_KEY not found in secrets!")
        st.info("Add it in Streamlit Cloud: Settings → Secrets")
    
    # Document upload section
    st.subheader("📄 Upload Document")
    
    uploaded_file = st.file_uploader(
        "Choose a file (TXT, PDF, DOCX)",
        type=['txt', 'pdf', 'docx'],
        help="Upload your IHRA document"
    )
    
    if uploaded_file:
        if st.button("🚀 Process Document", use_container_width=True):
            with st.spinner("📖 Reading document..."):
                doc_text = st.session_state.chatbot.load_document(uploaded_file)
            
            if doc_text and not doc_text.startswith("Error"):
                # Reset database
                st.session_state.chatbot.reset_database()
                
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(current, total):
                    progress = current / total
                    progress_bar.progress(progress)
                    status_text.text(f"Processing chunks: {current}/{total}")
                
                try:
                    # Store document
                    chunk_count = st.session_state.chatbot.store_document(
                        doc_text,
                        update_progress
                    )
                    
                    st.session_state.document_loaded = True
                    st.session_state.chunk_count = chunk_count
                    st.session_state.messages = []  # Clear chat
                    
                    progress_bar.empty()
                    status_text.empty()
                    st.success(f"✅ Loaded {chunk_count} chunks!")
                except Exception as e:
                    st.error(f"❌ Error processing document: {str(e)}")
            else:
                st.error(f"❌ {doc_text}")
    
    # Document status
    st.markdown("---")
    st.subheader("📊 Status")
    
    if st.session_state.document_loaded:
        st.success(f"✅ Document loaded")
        st.info(f"📦 {st.session_state.chunk_count} chunks")
        st.metric("Total Chunks", st.session_state.chunk_count)
    else:
        st.warning("⚠️ No document loaded")
    
    # Settings
    st.markdown("---")
    st.subheader("⚙️ Settings")
    
    model_choice = st.selectbox(
        "Model",
        ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
        help="GPT-4 is more accurate but slower and costlier"
    )
    
    temperature = st.slider(
        "Creativity",
        0.0, 1.0, 0.7,
        help="Lower = more focused, Higher = more creative"
    )
    
    # Actions
    st.markdown("---")
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    if st.button("🔄 Reset Database", use_container_width=True):
        if st.session_state.chatbot.reset_database():
            st.session_state.document_loaded = False
            st.session_state.chunk_count = 0
            st.session_state.messages = []
            st.success("✅ Database reset!")
            st.rerun()


# Main interface
st.title("💬 IHRA Healthcare Assistant Pro")

if not st.session_state.document_loaded:
    st.info("👈 Please upload a document in the sidebar to get started!")
    
    st.markdown("""
    ### 📚 Supported File Types
    - **TXT**: Plain text files
    - **PDF**: PDF documents
    - **DOCX**: Microsoft Word documents
    
    ### 🎯 Features
    - **RAG Technology**: Retrieval Augmented Generation for accurate answers
    - **Vector Database**: Fast semantic search with ChromaDB
    - **Multiple Models**: Choose between GPT-4 and GPT-3.5
    - **Conversation Memory**: Context-aware responses
    
    ### ⚙️ Deployment Notes
    - Uses in-memory storage (data resets on app restart)
    - Optimized for Streamlit Cloud
    """)
else:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask your question..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                response = st.session_state.chatbot.chat(
                    prompt,
                    st.session_state.messages[:-1],
                    model=model_choice,
                    temperature=temperature
                )
            st.markdown(response)
        # Add assistant message
        st.session_state.messages.append({"role": "assistant", "content": response})

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <small>
    🏥 IHRA - Islamabad Healthcare Regulatory Authority<br>
    📞 051-9199-902 | 📧 info@ihra.gov.pk
    </small>
</div>
""", unsafe_allow_html=True)