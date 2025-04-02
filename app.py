import gradio as gr
from document_loader import load_text_from_file
from vector_store import create_vectorstore, get_retriever
from query_engine import get_answer

retriever = None

def upload_file(file):
    global retriever
    text = load_text_from_file(file.name)
    vectorstore = create_vectorstore(text)
    retriever = get_retriever(vectorstore)
    return "✅ Document uploaded and indexed!"

def chat_with_doc(query, history):
    if retriever is None:
        return "❌ Please upload a document first."
    return get_answer(retriever, query)

with gr.Blocks() as demo:
    gr.Markdown("# 📄 Chat with Your Document")

    with gr.Row():
        file_input = gr.File(label="Upload PDF, DOCX, or TXT")
        upload_status = gr.Textbox(label="Status", interactive=False)

    file_input.change(fn=upload_file, inputs=file_input, outputs=upload_status)

    chatbot = gr.ChatInterface(fn=chat_with_doc)

demo.launch()
