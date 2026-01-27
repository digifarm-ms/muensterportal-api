"""Streamlit application for Münster4You Wiki RAG system."""

import time

import streamlit as st

from rag.config import config
from rag.generation import RAGGenerator
from rag.retrieval import WikiRetriever

# Page configuration
st.set_page_config(
    page_title="Münster4You - Wiki Assistent", page_icon="🏛️", layout="wide"
)


@st.cache_resource
def load_retriever():
    """Load and cache the retriever."""
    return WikiRetriever()


@st.cache_resource
def load_generator():
    """Load and cache the generator."""
    return RAGGenerator()


def main():
    # Title and description
    st.title("🏛️ Münster4You - Wiki Assistent")
    st.markdown(
        "Stelle Fragen über Münster und erhalte Antworten basierend auf dem Münster4You-Wiki."
    )

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Einstellungen")

        top_k = st.slider(
            "Anzahl Dokumente",
            min_value=1,
            max_value=10,
            value=config.default_top_k,
            help="Wie viele relevante Dokumente sollen abgerufen werden?",
        )

        temperature = st.slider(
            "Temperatur",
            min_value=0.0,
            max_value=1.0,
            value=config.default_temperature,
            step=0.1,
            help="Höhere Werte = kreativere Antworten, niedrigere Werte = präzisere Antworten",
        )

        st.divider()

        st.subheader("ℹ️ Über diese App")
        st.markdown(
            """
            Diese App nutzt **Retrieval-Augmented Generation (RAG)**, um Fragen über Münster zu beantworten:

            1. 🔍 **Suche**: Findet relevante Wiki-Seiten
            2. 🤖 **Generierung**: Erstellt Antwort mit Qwen3 30B
            3. 📚 **Quellen**: Zeigt verwendete Dokumente

            **Modelle:**
            - Embeddings: mixedbread-ai (Deutsch)
            - Generation: Qwen3 30B (via Ollama)
            """
        )

        st.divider()

        # Show stats
        if st.button("📊 Statistiken anzeigen"):
            try:
                retriever = load_retriever()
                st.metric("Anzahl Dokumente", len(retriever.df))
                st.metric("Embedding Dimension", retriever.doc_embeddings.shape[1])
            except Exception as e:
                st.error(f"Fehler beim Laden der Statistiken: {e}")

    # Main content area
    st.divider()

    # Question input
    question = st.text_input(
        "❓ Stelle eine Frage über Münster:",
        placeholder="z.B. Wo kann ich in Münster frisches Gemüse kaufen?",
        key="question_input",
    )

    # Search button
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        search_button = st.button("🔍 Suchen", type="primary", use_container_width=True)
    with col2:
        if st.button("🗑️ Löschen", use_container_width=True):
            st.rerun()

    # Process query
    if search_button and question:
        try:
            # Load models
            with st.spinner("Lade Modelle..."):
                retriever = load_retriever()
                generator = load_generator()

            # Retrieve documents
            with st.spinner("Suche relevante Dokumente..."):
                start_time = time.time()
                results = retriever.retrieve(question, top_k=top_k)
                retrieval_time = time.time() - start_time

            if not results:
                st.warning(
                    "⚠️ Keine relevanten Dokumente gefunden. Versuche eine andere Frage."
                )
                return

            # Display retrieved documents
            st.subheader(f"📚 Gefundene Dokumente ({len(results)})")

            # Create columns for document cards
            for i, result in enumerate(results, 1):
                with st.expander(
                    f"**{i}. {result.page_title}** - Relevanz: {result.similarity_score:.1%}",
                    expanded=(i == 1),  # Expand first result
                ):
                    st.markdown(f"**Ähnlichkeit:** {result.similarity_score:.3f}")
                    st.markdown(f"**Länge:** {result.page_len} Zeichen")
                    st.divider()

                    # Show content preview
                    preview_length = 500
                    content_preview = result.content_text[:preview_length]
                    if len(result.content_text) > preview_length:
                        content_preview += "..."

                    st.markdown(content_preview)

            st.divider()

            # Generate answer
            st.subheader("💡 Antwort")

            with st.spinner("Generiere Antwort..."):
                start_time = time.time()
                answer = generator.generate(question, results, temperature=temperature)
                generation_time = time.time() - start_time

            # Display answer in a nice box
            st.markdown(
                f"""
                <div style="background-color: #8f939c; padding: 20px; border-radius: 10px; border-left: 5px solid #1f77b4;">
                {answer}
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Performance metrics
            st.divider()
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("⏱️ Abrufzeit", f"{retrieval_time:.2f}s")
            with col2:
                st.metric("⏱️ Generierungszeit", f"{generation_time:.2f}s")
            with col3:
                st.metric("⏱️ Gesamtzeit", f"{retrieval_time + generation_time:.2f}s")

        except Exception as e:
            st.error(f"❌ Fehler bei der Verarbeitung: {str(e)}")
            st.exception(e)

    elif search_button and not question:
        st.warning("⚠️ Bitte gib eine Frage ein.")

    # Example questions
    st.divider()
    st.subheader("💡 Beispiel-Fragen")

    example_questions = [
        "Wo kann ich in Münster frisches Gemüse kaufen?",
        "Was kann man in der Freizeit in Münster machen?",
        "Wo finde ich Hofläden in Handorf?",
        "Welche Veranstaltungen gibt es in Münster?",
        "Was gibt es Interessantes über die Geschichte von Münster?",
    ]

    cols = st.columns(2)
    for i, example in enumerate(example_questions):
        with cols[i % 2]:
            if st.button(f"📝 {example}", key=f"example_{i}", use_container_width=True):
                st.session_state.question_input = example
                st.rerun()


if __name__ == "__main__":
    main()
