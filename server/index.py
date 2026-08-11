"""Vector index: document loading, chunking, ChromaDB index build/ingestion."""
import os
import shutil
import logging

from llama_cpp import Llama
from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from . import state
from .config import (
    log,
    DATA_PATH,
    PERSIST_DIRECTORY,
    EMBEDDING_MODEL_FILENAME,
    EMBEDDING_MODEL_PATH,
    INDEX_VERSION,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    RETRIEVAL_K,
)
from .gpu import get_gpu_layers
from .embeddings import LlamaCppEmbeddings, ensure_embedding_model
from .domains import load_domains, get_domain_path
from .llm import get_current_model_path
from .settings import save_settings


def ensure_embedding_index_matches():
    """Wipe the persisted Chroma index if it was built with a different embedding model or index version."""
    if os.path.exists(PERSIST_DIRECTORY):
        expected = f"{EMBEDDING_MODEL_FILENAME}|{INDEX_VERSION}"
        actual = state.embedding_model_marker
        if actual != expected:
            log.info(f"Index format changed ({actual or 'unknown'} -> {expected}); rebuilding index")
            for entry in os.listdir(PERSIST_DIRECTORY):
                entry_path = os.path.join(PERSIST_DIRECTORY, entry)
                if os.path.isdir(entry_path):
                    shutil.rmtree(entry_path)
                else:
                    os.remove(entry_path)


def mark_embedding_index_matches():
    state.embedding_model_marker = f"{EMBEDDING_MODEL_FILENAME}|{INDEX_VERSION}"
    save_settings()


def load_documents_from_data_dir() -> list:
    """Load all supported documents (.pdf + .txt) from the data dir."""
    raw_documents = []
    if not os.path.isdir(DATA_PATH):
        return raw_documents
    for f in sorted(os.listdir(DATA_PATH)):
        fpath = os.path.join(DATA_PATH, f)
        if not os.path.isfile(fpath):
            continue
        try:
            if f.lower().endswith(".pdf"):
                raw_documents.extend(PyPDFLoader(fpath).load())
            elif f.lower().endswith(".txt"):
                raw_documents.extend(TextLoader(fpath, encoding="utf-8").load())
        except Exception as e:
            log.warning(f"Skipping {f}: {e}")
    return raw_documents


def indexed_source_files(vector_store) -> set:
    """Return the set of source filenames already present in the index."""
    try:
        data = vector_store.get(include=["metadatas"])
        metadatas = data.get("metadatas") or []
        return {os.path.basename(m.get("source", "")) for m in metadatas if m and m.get("source")}
    except Exception as e:
        log.warning(f"Could not read indexed sources: {e}")
        return set()


def build_resources():
    if os.path.isdir(PERSIST_DIRECTORY):
        try:
            shutil.rmtree(PERSIST_DIRECTORY)
        except Exception:
            pass

    ensure_embedding_index_matches()

    if not ensure_embedding_model():
        log.error("Embedding model unavailable. RAG features disabled.")
        state.embeddings_instance = None
        model_path = get_current_model_path()
        if model_path:
            log.info(f"Loading chat model: {model_path}")
            state.llm_instance = Llama(model_path=model_path, n_ctx=4096, verbose=False, n_gpu_layers=get_gpu_layers())
        return

    log.info("Initializing embedding model")
    state.embeddings_instance = LlamaCppEmbeddings(model_path=EMBEDDING_MODEL_PATH)

    model_path = get_current_model_path()
    if model_path:
        log.info(f"Loading chat model: {model_path}")
        state.llm_instance = Llama(model_path=model_path, n_ctx=4096, verbose=False, n_gpu_layers=get_gpu_layers())
    else:
        log.warning("No chat model found. Use Settings to download one.")

    log.info("Loading and Chunking Documents from all domains")
    raw_documents = []
    domains = load_domains()
    for domain in domains:
        dpath = get_domain_path(domain)
        if not os.path.isdir(dpath):
            continue
        pdf_files = [f for f in os.listdir(dpath) if f.endswith(".pdf")]
        if pdf_files:
            for pdf_file in pdf_files:
                loader = PyPDFLoader(os.path.join(dpath, pdf_file))
                for doc in loader.load():
                    doc.metadata["domain"] = domain
                    raw_documents.append(doc)
        else:
            loader = DirectoryLoader(dpath, glob="*.txt", loader_cls=TextLoader)
            for doc in loader.load():
                doc.metadata["domain"] = domain
                raw_documents.append(doc)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    docs = text_splitter.split_documents(raw_documents)
    log.info(f"Loaded {len(raw_documents)} source pages -> {len(docs)} chunks across {len(domains)} domain(s): {domains}")

    if docs:
        log.info("Opening Vector Store")
        state.vector_store = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=state.embeddings_instance,
            collection_metadata={"hnsw:space": "cosine"},
        )
        existing = indexed_source_files(state.vector_store)
        new_docs = [d for d in docs if os.path.basename(d.metadata.get("source", "")) not in existing]
        if new_docs:
            log.info(f"Adding {len(new_docs)} new chunks (skipping {len(docs) - len(new_docs)} already indexed)")
            state.vector_store.add_documents(new_docs)
        else:
            log.info("All documents already indexed")
        state.retriever = state.vector_store.as_retriever(search_kwargs={"k": RETRIEVAL_K})
    else:
        log.info("No documents found — vector store will be created on first ingestion")
    mark_embedding_index_matches()


def run_ingestion(file_paths: list[str], domain: str = "General"):
    ensure_embedding_index_matches()
    embeddings = state.embeddings_instance or LlamaCppEmbeddings(model_path=EMBEDDING_MODEL_PATH)

    state.vector_store = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )

    domain_dir = get_domain_path(domain)
    os.makedirs(domain_dir, exist_ok=True)

    total = len(file_paths)
    state.ingestion_progress["status"] = "running"
    state.ingestion_progress["total"] = total
    state.ingestion_progress["domain"] = domain

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    for i, file_path in enumerate(file_paths):
        filename = os.path.basename(file_path)
        state.ingestion_progress["current"] = i + 1
        state.ingestion_progress["current_file"] = filename
        state.ingestion_progress["message"] = f"Loading {filename}..."

        try:
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            else:
                loader = TextLoader(file_path, encoding="utf-8")
            raw_docs = loader.load()

            docs = text_splitter.split_documents(raw_docs)

            for doc in docs:
                doc.metadata["domain"] = domain

            try:
                state.vector_store._collection.delete(where={"source": file_path})
            except Exception:
                pass

            state.ingestion_progress["message"] = f"Indexing {filename} ({len(docs)} chunks)..."
            state.vector_store.add_documents(docs)

            dest = os.path.join(domain_dir, filename)
            shutil.move(file_path, dest)

            state.ingestion_progress["message"] = f"Processed {filename} ({len(docs)} chunks)"
        except Exception as e:
            log.warning(f"Error processing {filename}: {e}")
            state.ingestion_progress["message"] = f"Error: {filename} - {e}"

    state.retriever = state.vector_store.as_retriever(search_kwargs={"k": RETRIEVAL_K})
    mark_embedding_index_matches()
    state.ingestion_progress["status"] = "completed"
    state.ingestion_progress["message"] = f"Ingested {total} file(s) into '{domain}'"
