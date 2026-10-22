
import streamlit as st

from src.ask import ask_question

st.set_page_config(
    page_title="Intelligent Document Agent",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Intelligent Document Agent")
st.write(
    "Haz preguntas sobre tus documentos PDF y obtén respuestas "
    "basadas en su contenido, con referencias a las fuentes."
)

with st.sidebar:
    st.header("Configuración")
    top_k = st.slider(
        "Fragmentos relevantes que recuperar",
        min_value=1,
        max_value=10,
        value=5,
    )
    st.caption("Las respuestas se generan con Gemini y RAG.")

question = st.text_area(
    "¿Qué quieres saber de tus documentos?",
    placeholder="Ej.: ¿Qué es RAG y cómo funciona?",
    height=100,
)

if st.button("Consultar documentos", type="primary"):
    if not question.strip():
        st.warning("Escribe una pregunta antes de continuar.")
    else:
        try:
            with st.spinner("Buscando información y generando respuesta..."):
                answer, sources = ask_question(
                    question.strip(),
                    top_k=top_k,
                )

            st.subheader("Respuesta")
            st.markdown(answer)

            st.subheader("Fuentes consultadas")

            if sources:
                for i, source in enumerate(sources, start=1):
                    with st.expander(
                        f"{i}. {source['source']} — página {source['page']}"
                        f" (similitud: {source['similarity']:.3f})"
                    ):
                        st.write(source["text"])
            else:
                st.info("No se han encontrado fragmentos relevantes.")

        except FileNotFoundError:
            st.error(
                "No se encuentra el índice. Ejecuta primero el indexador "
                "de documentos PDF."
            )
        except Exception as exc:
            st.error(f"No se pudo procesar la pregunta: {exc}")

st.divider()
st.caption("Proyecto de portfolio · Python · RAG · Gemini · SQLite")
