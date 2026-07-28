import os
import logging
import requests
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.llms import LLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from llama_cpp import Llama, llama_supports_gpu_offload

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ── GPU Detection ──
_gpu_available = False

def detect_gpu():
    global _gpu_available
    if not llama_supports_gpu_offload():
        log.info("GPU offload not available (llama-cpp-python compiled without CUDA)")
        return
    try:
        import subprocess
        result = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout.strip():
            _gpu_available = True
            log.info(f"GPU detected: {result.stdout.strip().split(chr(10))[0]}")
        else:
            log.info("nvidia-smi failed — no NVIDIA GPU or driver issue")
    except Exception as e:
        log.info(f"GPU detection skipped: {e}")

def get_gpu_layers() -> int:
    return -1 if _gpu_available else 0


# --- 1. Custom LangChain Wrappers for llama.cpp ---

class LlamaCppEmbeddings(Embeddings):
    """Custom embedding wrapper using llama-cpp-python."""
    def __init__(self, model_path: str):
        self.client = Llama(model_path=model_path, embedding=True, verbose=False, n_gpu_layers=get_gpu_layers())

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # llama-cpp accepts a list or single string depending on version, 
        # embedding creation typically takes a string or list.
        return [self.client.create_embedding(text)["data"][0]["embedding"] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.client.create_embedding(text)["data"][0]["embedding"]


class LlamaCppLLM(LLM):
    """Custom LLM wrapper using llama-cpp-python for text generation."""
    model_path: str
    temperature: float = 0.1
    max_tokens: int = 512
    client: Llama = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client = Llama(
            model_path=self.model_path, 
            n_ctx=2048, 
            verbose=False,
            n_gpu_layers=get_gpu_layers()
        )

    def _call(self, prompt: str, stop: list[str] | None = None, **kwargs) -> str:
        response = self.client(
            prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop=stop or []
        )
        return response["choices"][0]["text"]

    @property
    def _llm_type(self) -> str:
        return "llama_cpp_custom"


# --- 2. Main RAG Pipeline Script ---

DATA_PATH = "./data"
PERSIST_DIRECTORY = "./chroma_db"

EMBEDDING_MODEL_PATH = "./models/all-MiniLM-L6-v2-ggml-model-f16.gguf"
EMBEDDING_MODEL_INFO = {
    "repo_id": "second-state/All-MiniLM-L6-v2-Embedding-GGUF",
    "filename": "all-MiniLM-L6-v2-ggml-model-f16.gguf",
}
LLM_MODEL_PATH = "./models/qwen2.5-1.5b-instruct-q4_k_m.gguf"


def ensure_embedding_model() -> bool:
    if os.path.exists(EMBEDDING_MODEL_PATH):
        return True
    log.info("Embedding model not found, downloading...")
    os.makedirs(os.path.dirname(EMBEDDING_MODEL_PATH), exist_ok=True)
    url = f"https://huggingface.co/{EMBEDDING_MODEL_INFO['repo_id']}/resolve/main/{EMBEDDING_MODEL_INFO['filename']}"
    tmp = EMBEDDING_MODEL_PATH + ".tmp"
    try:
        resp = requests.get(url, stream=True, timeout=30)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = int(downloaded / total * 100)
                    log.info(f"Downloading embedding model... {pct}%")
        os.replace(tmp, EMBEDDING_MODEL_PATH)
        log.info(f"Embedding model downloaded ({downloaded} bytes)")
        return True
    except Exception as e:
        log.error(f"Failed to download embedding model: {e}")
        if os.path.exists(tmp):
            os.remove(tmp)
        return False


def main():
    detect_gpu()
    print("--- 1. Initializing llama.cpp Models ---")
    if not ensure_embedding_model():
        print("ERROR: Could not download embedding model. Exiting.")
        return
    embeddings = LlamaCppEmbeddings(model_path=EMBEDDING_MODEL_PATH)
    llm = LlamaCppLLM(model_path=LLM_MODEL_PATH)

    print("--- 2. Loading and Chunking Documents ---")
    pdf_files = [f for f in os.listdir(DATA_PATH) if f.endswith(".pdf")]
    if pdf_files:
        raw_documents = []
        for pdf_file in pdf_files:
            loader = PyPDFLoader(os.path.join(DATA_PATH, pdf_file))
            raw_documents.extend(loader.load())
    else:
        loader = DirectoryLoader(DATA_PATH, glob="*.txt", loader_cls=TextLoader)
        raw_documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    docs = text_splitter.split_documents(raw_documents)

    print("--- 3. Creating Vector Store ---")
    if not docs:
        print("No documents found in data/. Place .txt or .pdf files and try again.")
        return
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    print("--- 4. Setting up RAG Prompt & Chain ---")
    system_prompt = (
        "<|system|>\n"
        "You are a tutor helping you student pass the AWS Certified Generative AI Developer - Professional (AIP-C01) Exam. "
        "Use the retrieved context to answer the question. "
        "If you don't know, say you don't know.\n"
        "Context:\n{context}<|end|>\n"
        "<|user|>\n{input}<|end|>\n"
        "<|assistant|>"
    )
    
    prompt = ChatPromptTemplate.from_template(system_prompt)

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    print("--- 5. Executing Query ---")
    query = "What should I focus on for my upcoming exam?"
    print(f"Query: {query}\n")

    response = rag_chain.invoke({"input": query})

    print("=== Answer ===")
    print(response["answer"])

if __name__ == "__main__":
    main()