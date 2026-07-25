import sys
sys.stdout.reconfigure(encoding='utf-8')
import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()

from utils.audio_processor import process_input, cleanup_downloads_folder
from core.transcriber import transcribe_all, correct_transcript, transcribe_chunk_groq
from core.summarizer import summarize, generate_title, generate_mindmap
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question, build_pdf_rag_chain
from core.pdf_extractor import extract_pdf_text
from core.vector_store import build_pdf_vector_store, get_pdf_retriever

st.set_page_config(page_title="AI Audio Transcriber", page_icon="🎙️", layout="centered")

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF4B4B;
        text-align: center;
        font-weight: bold;
    }
    .sub-header {
        text-align: center;
        color: #888888;
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🎙️ AI Audio Transcriber</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Fast & accurate transcription powered by Whisper & Groq API</p>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🎥 Video Transcriber", "📄 PDF Q&A"])

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Audio Source")
        input_type = st.radio("Select input type:", ("YouTube URL", "Audio/Video File"))

        source = None
        if input_type == "YouTube URL":
            source = st.text_input("Enter YouTube Video URL:", placeholder="https://youtu.be/...")
        else:
            uploaded_file = st.file_uploader("Upload an audio or video file", type=["mp3", "wav", "m4a", "ogg", "mp4", "mkv", "webm", "mov"])
            if uploaded_file is not None:
                # Save the uploaded file temporarily
                temp_dir = "downloads"
                os.makedirs(temp_dir, exist_ok=True)
                source = os.path.join(temp_dir, uploaded_file.name)
                with open(source, "wb") as f:
                    f.write(uploaded_file.getbuffer())

    with col2:
        st.subheader("2. Settings")
        language = st.selectbox(
            "Transcription Language", 
            ("English", "Hindi", "Marathi", "Hinglish")
        )

    # Initialize session state variables
    if "processed" not in st.session_state:
        st.session_state.processed = False
        st.session_state.title = ""
        st.session_state.summary = ""
        st.session_state.action_items = ""
        st.session_state.decisions = ""
        st.session_state.questions = ""
        st.session_state.transcript = ""
        st.session_state.mindmap = ""
        st.session_state.rag_chain = None
        st.session_state.chat_history = []
        st.session_state.engine_log = []
        st.session_state.chunks = []
        st.session_state.re_transcribed = False

    st.markdown("---")

    if st.button("🚀 Start Transcription", use_container_width=True):
        if source:
            with st.spinner("Processing audio... This might take a while depending on your hardware."):
                try:
                    # Process input (download YouTube or just get file chunks)
                    st.info("🎵 Splitting audio into chunks...")
                    chunks = process_input(source)
                    st.session_state.chunks = chunks
                    st.session_state.re_transcribed = False
                    
                    # Transcribe
                    st.info(f"⚡ Transcribing chunks in {language}...")
                    transcript, engine_log = transcribe_all(chunks, language=language)
                    
                    if language.lower() in ["marathi", "hindi", "hinglish"]:
                        with st.spinner("🔧 Correcting transcript errors..."):
                            transcript = correct_transcript(transcript, language=language)
                    
                    st.info("🧠 Generating title, summary, and extracting key points...")
                    
                    try:
                        title = generate_title(transcript)
                    except Exception as e:
                        title = "Untitled Transcript"
                        st.warning(f"Failed to generate title: {e}")
                        
                    try:
                        summary = summarize(transcript)
                    except Exception as e:
                        summary = "Summary could not be generated."
                        st.warning(f"Failed to generate summary: {e}")
                        
                    try:
                        action_items = extract_action_items(transcript)
                    except Exception as e:
                        action_items = "Action items could not be generated."
                        st.warning(f"Failed to extract action items: {e}")
                        
                    try:
                        decisions = extract_key_decisions(transcript)
                    except Exception as e:
                        decisions = "Key decisions could not be generated."
                        st.warning(f"Failed to extract decisions: {e}")
                        
                    try:
                        questions = extract_questions(transcript)
                    except Exception as e:
                        questions = "Questions could not be generated."
                        st.warning(f"Failed to extract questions: {e}")
                    
                    with st.spinner("🧠 Generating mind map..."):
                        try:
                            mindmap = generate_mindmap(transcript)
                        except Exception as e:
                            mindmap = ""
                            st.warning(f"Failed to generate mind map: {e}")
                    
                    # Store in session state
                    st.session_state.title = title
                    st.session_state.summary = summary
                    st.session_state.action_items = action_items
                    st.session_state.decisions = decisions
                    st.session_state.questions = questions
                    st.session_state.mindmap = mindmap
                    st.session_state.transcript = transcript
                    st.session_state.engine_log = engine_log
                    st.session_state.rag_chain = build_rag_chain(transcript)
                    st.session_state.chat_history = []  # Clear previous chat history
                    st.session_state.processed = True
                    
                    # Clean up downloaded files to free up space
                    cleanup_downloads_folder()
                    
                    st.success("✅ Analysis complete!")
                except Exception as e:
                    st.error(f"❌ An error occurred: {e}")
        else:
            st.warning("⚠️ Please provide an input source (URL or File).")

    if st.session_state.processed:
        st.markdown("---")
        st.markdown(f"## 📌 {st.session_state.title}")
        
        st.markdown("### 📋 SUMMARY")
        st.write(st.session_state.summary)
        
        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
        
        # 1. Action Items
        with st.container(border=True):
            st.markdown("## ✅ ACTION ITEMS")
            if st.session_state.action_items and st.session_state.action_items.strip():
                st.write(st.session_state.action_items)
            else:
                st.markdown("<p style='color: #888888; font-style: italic;'>No action items found.</p>", unsafe_allow_html=True)
                
        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
        
        # 2. Key Decisions
        with st.container(border=True):
            st.markdown("## 🔑 KEY DECISIONS")
            if st.session_state.decisions and st.session_state.decisions.strip():
                st.write(st.session_state.decisions)
            else:
                st.markdown("<p style='color: #888888; font-style: italic;'>No key decisions found.</p>", unsafe_allow_html=True)
                
        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
        
        # 3. Open Questions
        with st.container(border=True):
            st.markdown("## ❓ OPEN QUESTIONS")
            if st.session_state.questions and st.session_state.questions.strip():
                st.write(st.session_state.questions)
            else:
                st.markdown("<p style='color: #888888; font-style: italic;'>No open questions found.</p>", unsafe_allow_html=True)
        
        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
        
        # Mind Map
        st.markdown("## 🧠 Mind Map")
        if st.session_state.mindmap:
            mermaid_html = f"""
            <pre class="mermaid">
{st.session_state.mindmap}
            </pre>
            <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
            <script>
                mermaid.initialize({{startOnLoad: true}});
            </script>
            """
            st.components.v1.html(mermaid_html, height=550)
        else:
            st.markdown("<p style='color: #888888; font-style: italic;'>No mind map generated.</p>", unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 📝 TRANSCRIPT")
        st.text_area("", st.session_state.transcript, height=300, label_visibility="collapsed")
        if st.session_state.get("re_transcribed", False):
            st.caption("✅ Re-transcribed using Groq API")
        
        with st.expander("🔍 Transcription engine breakdown"):
            log = st.session_state.get("engine_log", [])
            total_chunks = len(log)
            local_count = sum(1 for entry in log if entry.get("engine") == "local")
            groq_count = sum(1 for entry in log if entry.get("engine") == "groq")
            avg_confidence = sum(entry.get("confidence", 0.0) for entry in log) / total_chunks if total_chunks > 0 else 0.0
            
            st.write(f"**Total Chunks:** {total_chunks}")
            st.write(f"**Local Whisper Chunks:** {local_count}")
            st.write(f"**Groq API Chunks:** {groq_count}")
            st.write(f"**Average Confidence (logprob):** {avg_confidence:.4f}")
            
        if st.button("👎 Not satisfied? Re-transcribe with Groq", use_container_width=True):
            if "chunks" in st.session_state and st.session_state.chunks:
                with st.spinner("Re-transcribing all chunks using Groq API..."):
                    try:
                        groq_transcript = ""
                        engine_log = []
                        for chunk in st.session_state.chunks:
                            text = transcribe_chunk_groq(chunk, language=language)
                            groq_transcript += text + " "
                            engine_log.append({
                                "engine": "groq",
                                "confidence": 0.0
                            })
                        
                        transcript = groq_transcript.strip()
                        if language.lower() in ["marathi", "hindi", "hinglish"]:
                            transcript = correct_transcript(transcript, language=language)
                        
                        st.session_state.transcript = transcript
                        st.session_state.engine_log = engine_log
                        st.session_state.re_transcribed = True
                        
                        # Re-run downstream analyses on the new transcript safely
                        try: st.session_state.title = generate_title(transcript)
                        except: st.session_state.title = "Untitled Transcript"
                        
                        try: st.session_state.summary = summarize(transcript)
                        except: st.session_state.summary = "Summary could not be generated."
                        
                        try: st.session_state.action_items = extract_action_items(transcript)
                        except: st.session_state.action_items = "Action items could not be generated."
                        
                        try: st.session_state.decisions = extract_key_decisions(transcript)
                        except: st.session_state.decisions = "Key decisions could not be generated."
                        
                        try: st.session_state.questions = extract_questions(transcript)
                        except: st.session_state.questions = "Questions could not be generated."
                        
                        try: st.session_state.mindmap = generate_mindmap(transcript)
                        except: st.session_state.mindmap = ""
                        st.session_state.rag_chain = build_rag_chain(transcript)
                        st.session_state.chat_history = []
                        
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Re-transcription failed: {e}")
            else:
                st.error("❌ Audio chunks are not available in session state.")
        
        # Combine all text into one string for the download button
        full_export = f"📌 TITLE: {st.session_state.title}\n\n📋 SUMMARY\n------------------------------------------------------------\n{st.session_state.summary}\n\n✅ ACTION ITEMS\n------------------------------------------------------------\n{st.session_state.action_items}\n\n🔑 KEY DECISIONS\n------------------------------------------------------------\n{st.session_state.decisions}\n\n❓ OPEN QUESTIONS\n------------------------------------------------------------\n{st.session_state.questions}\n\n📝 TRANSCRIPT\n------------------------------------------------------------\n{st.session_state.transcript}"
        
        # Provide download button for the full report
        st.download_button(
            label="💾 Download Full Report",
            data=full_export,
            file_name="transcript_report.txt",
            mime="text/plain",
            use_container_width=True
        )

        st.markdown("---")
        st.markdown("### 💬 Chat with your Meeting")
        
        answer_language = st.selectbox("Answer language:", ["Auto-detect (match my question)", "English", "Marathi", "Hindi"], key="video_answer_lang")
        
        # Text input for the question
        question = st.text_input("Ask a question about the transcript:", key="rag_question")
        
        if st.button("Ask Assistant", use_container_width=True):
            if question:
                with st.spinner("Thinking..."):
                    try:
                        if st.session_state.rag_chain is None:
                            st.session_state.rag_chain = build_rag_chain(st.session_state.transcript)
                        answer = ask_question(st.session_state.rag_chain, question, answer_language)
                        st.session_state.chat_history.append((question, answer))
                    except Exception as e:
                        st.error(f"❌ Failed to get answer: {e}")
            else:
                st.warning("Please enter a question.")
                
        # Display chat history
        if st.session_state.chat_history:
            st.markdown("#### Conversation History:")
            for q, a in reversed(st.session_state.chat_history):
                with st.chat_message("user"):
                    st.write(q)
                with st.chat_message("assistant"):
                    st.write(a)

with tab2:
    st.subheader("📄 PDF Q&A")
    st.markdown("Upload a PDF document to parse, index, and ask questions about its content.")

    # Initialize PDF session state variables
    if "pdf_processed" not in st.session_state:
        st.session_state.pdf_processed = False
        st.session_state.pdf_retriever = None
        st.session_state.pdf_chat_history = []
        st.session_state.pdf_doc_id = ""
        st.session_state.pdf_rag_chain = None

    uploaded_pdf = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_pdf is not None:
        # Check if this is a new document to reset the state
        doc_id = uploaded_pdf.name
        if st.session_state.pdf_doc_id != doc_id:
            st.session_state.pdf_processed = False
            st.session_state.pdf_retriever = None
            st.session_state.pdf_chat_history = []
            st.session_state.pdf_doc_id = doc_id
            st.session_state.pdf_rag_chain = None

        if not st.session_state.pdf_processed:
            if st.button("📥 Process PDF Document", use_container_width=True):
                with st.spinner("Extracting text and building vector database..."):
                    try:
                        # Save the uploaded file temporarily
                        temp_dir = "downloads"
                        os.makedirs(temp_dir, exist_ok=True)
                        temp_pdf_path = os.path.join(temp_dir, doc_id)
                        with open(temp_pdf_path, "wb") as f:
                            f.write(uploaded_pdf.getbuffer())

                        # Extract text
                        st.info("Reading PDF pages...")
                        pdf_text = extract_pdf_text(temp_pdf_path)
                        
                        st.write(f"**Debug Extracted Text Preview:**\n\n`{pdf_text[:200]}...`")

                        if not pdf_text.strip():
                            st.error("No text could be extracted from the PDF.")
                        else:
                            st.info("Ingesting text into vector store...")
                            st.write(f"**Debug:** Extracted {len(pdf_text)} characters from PDF.")
                            
                            vector_store = build_pdf_vector_store(pdf_text, doc_id)
                            st.write("**Debug:** `build_pdf_vector_store` completed without error.")
                            
                            try:
                                col_name = vector_store._collection.name
                                chunk_count = vector_store._collection.count()
                                st.write(f"**Debug:** Collection Name: `{col_name}`, Chunks created/stored: {chunk_count}")
                            except Exception as dbg_e:
                                st.write(f"**Debug Error:** Could not read collection info: {dbg_e}")

                            st.session_state.pdf_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
                            st.session_state.pdf_rag_chain = build_pdf_rag_chain(st.session_state.pdf_retriever)
                            st.session_state.pdf_processed = True
                            
                            # Clean up downloaded files to free up space
                            cleanup_downloads_folder()
                            
                            st.success("✅ PDF processed and indexed successfully!")
                    except Exception as e:
                        st.error(f"❌ Failed to process PDF: {e}")

    # Display chat interface if PDF is processed
    if st.session_state.pdf_processed:
        st.markdown("---")
        st.markdown("### 💬 Chat with your PDF Document")
        
        pdf_answer_language = st.selectbox("Answer language:", ["Auto-detect (match my question)", "English", "Marathi", "Hindi"], key="pdf_answer_lang")
        
        # Text input for the question
        pdf_question = st.text_input("Ask a question about the PDF:", key="pdf_rag_question")
        
        if st.button("Ask PDF Assistant", use_container_width=True):
            if pdf_question:
                with st.spinner("Thinking..."):
                    try:
                        if st.session_state.pdf_rag_chain is None:
                            st.session_state.pdf_rag_chain = build_pdf_rag_chain(st.session_state.pdf_retriever)
                        
                        retriever = st.session_state.pdf_retriever
                        try:
                            col_name = retriever.vectorstore._collection.name
                            total_chunks = retriever.vectorstore._collection.count()
                            retrieved_docs = retriever.invoke(pdf_question)
                            st.info(f"**Debug Retrieval:** Querying collection `{col_name}`. Retrieved {len(retrieved_docs)} chunks for this question.")
                            
                            if len(retrieved_docs) == 0:
                                st.error(f"**Debug:** No matching content found in vector store — {total_chunks} chunks exist in this collection.")
                        except Exception as dbg_e:
                            st.write(f"**Debug Error during retrieval tracking:** {dbg_e}")

                        answer = ask_question(st.session_state.pdf_rag_chain, pdf_question, pdf_answer_language)
                        st.session_state.pdf_chat_history.append((pdf_question, answer))
                    except Exception as e:
                        st.error(f"❌ Failed to get answer: {e}")
            else:
                st.warning("Please enter a question.")
                
        # Display PDF chat history
        if st.session_state.pdf_chat_history:
            st.markdown("#### Conversation History:")
            for q, a in reversed(st.session_state.pdf_chat_history):
                with st.chat_message("user"):
                    st.write(q)
                with st.chat_message("assistant"):
                    st.write(a)
