from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from pydantic import PrivateAttr
from typing import Optional, List
import torch

class LocalCausalLLM(LLM):
    model_id: str = "mistralai/Mistral-7B-Instruct-v0.1"
    _tokenizer: any = PrivateAttr()
    _model: any = PrivateAttr()
    _pipeline: any = PrivateAttr()

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, model_id="mistralai/Mistral-7B-Instruct-v0.1", **kwargs):
        super().__init__(model_id=model_id, **kwargs)
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
        )
        self._pipeline = pipeline(
            "text-generation",
            model=self._model,
            tokenizer=self._tokenizer,
         
        )

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        result = self._pipeline(
            prompt,
            max_new_tokens=256,
            do_sample=False,
            temperature=0.3
        )
        return result[0]["generated_text"]

    @property
    def _llm_type(self) -> str:
        return "mistralai/Mistral-7B-Instruct-v0.1"

def format_prompt(query, context):
    return f"""You are a helpful assistant answering questions based on a document.

Context:
{context}

Question:
{query}

Answer:"""

def get_answer(retriever, query):
    llm = LocalCausalLLM()

    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = format_prompt(query, context)

    return llm(prompt)
