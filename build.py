"""
Build distributable bundles of the Local RAG application using PyInstaller.

Usage:
    python build.py                      # build both CPU and CUDA variants
    python build.py --variant cpu        # CPU-only build
    python build.py --variant cuda       # CUDA-enabled build only

Output:
    dist/LocalRAG-CPU/   - CPU-only bundle
    dist/LocalRAG-CUDA/  - CUDA-enabled bundle
"""
import argparse
import os
import shutil
import subprocess
import sys


LLAMA_CPP_CUDA_WHEEL = (
    "https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.34-cu125/"
    "llama_cpp_python-0.3.34-py3-none-win_amd64.whl"
)


def dir_size(path):
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total += os.path.getsize(fp)
    return total


def build_pyinstaller(name, root, dist_dir, build_dir):
    """Run PyInstaller with the standard shared arguments."""
    static_src = os.path.join(root, "static")
    embedding_src = os.path.join(root, "models", "all-MiniLM-L6-v2-ggml-model-f16.gguf")
    icon_src = os.path.join(root, "app_icon", "icon_preview.ico")
    entry_point = os.path.join(root, "gui.py")

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--clean",
        "--noconfirm",
        "--name", name,
        "--onedir",
        "--icon", icon_src,
        "--add-data", f"{static_src}{os.pathsep}static",
        "--add-data", f"{embedding_src}{os.pathsep}models",
        "--exclude", "torch",
        "--exclude", "sentence_transformers",
        "--exclude", "sklearn",
        "--exclude", "langgraph",
        "--exclude", "langgraph_checkpoint",
        "--exclude", "langgraph_prebuilt",
        "--exclude", "langgraph_sdk",
        "--exclude", "langchain_classic",
        "--exclude", "langchain",
        "--exclude", "transformers",
        "--exclude", "tokenizers",
        "--collect-all", "chromadb",
        "--collect-all", "llama_cpp",
        "--hidden-import", "uvicorn.logging",
        "--hidden-import", "uvicorn.loops.auto",
        "--hidden-import", "uvicorn.protocols.http.auto",
        "--hidden-import", "ddgs",
        "--hidden-import", "langchain_community.document_loaders",
        "--hidden-import", "langchain_community.document_loaders.pdf",
        "--hidden-import", "langchain_community.vectorstores",
        "--hidden-import", "langchain_community.vectorstores.chroma",
        "--hidden-import", "langchain_text_splitters",
        "--hidden-import", "langchain_core.embeddings",
        "--noconsole",
        entry_point,
    ]

    print(f"Running PyInstaller for {name}...")
    subprocess.check_call(cmd, cwd=root)

    bundle = os.path.join(dist_dir, name, f"{name}.exe")
    size = dir_size(os.path.join(dist_dir, name)) / 1e6
    print(f"\n{name} build complete!")
    print(f"Bundle: {bundle}")
    print(f"Size: {size:.1f} MB\n")
    return bundle, size


def main():
    parser = argparse.ArgumentParser(description="Build Local RAG distributable bundles")
    parser.add_argument(
        "--variant",
        choices=["cpu", "cuda", "all"],
        default="all",
        help="Which variant(s) to build (default: all)",
    )
    args = parser.parse_args()

    root = os.path.abspath(os.path.dirname(__file__))
    dist_dir = os.path.join(root, "dist")
    build_dir = os.path.join(root, "build")

    embedding_src = os.path.join(root, "models", "all-MiniLM-L6-v2-ggml-model-f16.gguf")
    if not os.path.exists(embedding_src):
        print(f"ERROR: Embedding model not found at {embedding_src}")
        print("Place the GGUF embedding model in models/ before building.")
        sys.exit(1)

    # Clean previous build artifacts
    for d in [build_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)

    # ---------- CPU variant ----------
    if args.variant in ("cpu", "all"):
        # Ensure CPU version of llama-cpp-python is installed
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "--force-reinstall", "--no-cache-dir",
            "llama-cpp-python==0.3.34",
        ])
        build_pyinstaller("LocalRAG-CPU", root, dist_dir, build_dir)

    # ---------- CUDA variant ----------
    if args.variant in ("cuda", "all"):
        print("Installing CUDA-enabled llama-cpp-python...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "--force-reinstall", "--no-cache-dir",
            LLAMA_CPP_CUDA_WHEEL,
        ])
        build_pyinstaller("LocalRAG-CUDA", root, dist_dir, build_dir)

    print("All builds complete!")


if __name__ == "__main__":
    main()