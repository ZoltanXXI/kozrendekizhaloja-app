#!/usr/bin/env python3
import os
import json
import math
import time
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from urllib.parse import urlparse

def chunks_from_text(text, max_chars=100000):
    text = text.replace("\r", "\n")
    parts = []
    cur = ""
    for paragraph in text.split("\n\n"):
        if not paragraph.strip():
            continue
        if len(cur) + len(paragraph) + 2 <= max_chars:
            cur = (cur + "\n\n" + paragraph).strip()
        else:
            if cur:
                parts.append(cur.strip())
            if len(paragraph) <= max_chars:
                cur = paragraph.strip()
            else:
                for i in range(0, len(paragraph), max_chars):
                    parts.append(paragraph[i:i+max_chars].strip())
                cur = ""
    if cur:
        parts.append(cur.strip())
    return parts

def get_openai_embeddings(client, texts, model="text-embedding-3-small"):
    out = []
    batch = 32
    for i in range(0, len(texts), batch):
        chunk = texts[i:i+batch]
        resp = client.embeddings.create(model=model, input=chunk)
        for d in resp.data:
            out.append(np.array(d.embedding, dtype='float32'))
        time.sleep(0.1)
    return out

def get_local_embeddings(model, texts):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model)
    embs = m.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return [np.array(e, dtype='float32') for e in embs]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out-index", default="recipes.index")
    parser.add_argument("--out-meta", default="recipes_meta.jsonl")
    parser.add_argument("--openai", action="store_true")
    parser.add_argument("--batch-chars", type=int, default=1000)
    parser.add_argument("--emb-model", default="text-embedding-3-small")
    parser.add_argument("--local-model", default="all-MiniLM-L6-v2")
    args = parser.parse_args()

    df = pd.read_csv(args.csv, dtype=str, keep_default_na=False)
    records = []
    for _, r in df.iterrows():
        url = (r.get("url") or "").strip()
        title = (r.get("title") or "").strip()
        ingredients = r.get("ingredients") or ""
        instructions = r.get("instructions") or ""
        if instructions and instructions.startswith('[') and instructions.endswith(']'):
            try:
                instr_par = json.loads(instructions)
                instructions = "\n".join(instr_par) if isinstance(instr_par, list) else str(instr_par)
            except Exception:
                pass
        text = ""
        if ingredients:
            text += "ALAPANYAGOK:\n" + (ingredients if isinstance(ingredients, str) else str(ingredients)) + "\n\n"
        if instructions:
            text += "ELKÉSZÍTÉS:\n" + instructions
        if not text.strip():
            continue
        rec_id = url or f"row_{_}"
        records.append({"id": rec_id, "url": url, "title": title, "text": text})

    meta_lines = []
    docs = []
    for rec in records:
        chunks = chunks_from_text(rec["text"], max_chars=args.batch_chars)
        for i, c in enumerate(chunks):
            doc_id = f"{rec['id']}::chunk::{i}"
            meta = {"id": doc_id, "source_id": rec["id"], "url": rec["url"], "title": rec["title"], "text": c}
            meta_lines.append(meta)
            docs.append(c)

    if not docs:
        print("No documents found.")
        return

    use_openai = args.openai and os.getenv("OPENAI_API_KEY")
    if use_openai:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        emb_list = get_openai_embeddings(client, docs, model=args.emb_model)
    else:
        emb_list = get_local_embeddings(args.local_model, docs)

    import faiss
    dim = len(emb_list[0])
    xb = np.stack(emb_list, axis=0)
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(xb)
    index.add(xb)
    faiss.write_index(index, args.out_index)

    with open(args.out_meta, "w", encoding="utf-8") as fh:
        for m in meta_lines:
            fh.write(json.dumps(m, ensure_ascii=False) + "\n")

    print("Wrote index:", args.out_index)
    print("Wrote meta:", args.out_meta)
    print("Docs:", len(meta_lines))

if __name__ == "__main__":
    main()
