config = {
    "embeddings": {
        "default": "text-embedding-v3",
        "legal": "text-embedding-v3",
        "fund": "text-embedding-v3",
        "batch_size": 10,
        "max_workers": 5,
        "rpm": 30
    },
    "pdf": {
        "text": "fitz", # "pdfminer", "pymupdf"
        "table": "pdfplumber", # "camelot"
        "formula": "baidu" # "mathpix"
    },
    "splitter": {
        "chunk_size": 1000,
        "chunk_overlap": 300
    },
    "retrieval": {
        "legal": {
            "k": 5,
            "threshold": 0.7
        },
        "fund": {
            "k": 5,
            "threshold": 0.7
        },
    },
    "llm": {
        "model": "qwen-turbo"
    }
}
