"""Download SBERT model to the local HuggingFace cache.

Run:
    UV_CACHE_DIR=.uv-cache uv run python scripts/download_sbert.py

If you need a proxy, set:
    export https_proxy=http://127.0.0.1:7897 http_proxy=http://127.0.0.1:7897 all_proxy=socks5://127.0.0.1:7897
"""

from sentence_transformers import SentenceTransformer


def main() -> None:
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)
    emb = model.encode(["test"], normalize_embeddings=True)
    print(f"Downloaded and cached {model_name}, sample embedding dim={emb.shape[1]}")


if __name__ == "__main__":
    main()
