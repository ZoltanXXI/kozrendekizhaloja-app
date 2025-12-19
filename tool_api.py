from fastapi import FastAPI, HTTPException, Request, Query, Header
from pydantic import BaseModel
import os
import pathlib
import hashlib
import datetime
import json
import logging
from typing import Optional, List

DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))
API_KEY = os.getenv("INTERNAL_API_KEY", "changeme")
AUDIT_LOG = os.path.abspath(os.path.join(os.path.dirname(__file__), "logs", "tool_api_audit.log"))
os.makedirs(os.path.dirname(AUDIT_LOG), exist_ok=True)
logging.basicConfig(filename=AUDIT_LOG, level=logging.INFO, format="%(message)s")

app = FastAPI()

def _safe_resolve(rel_path: str):
    p = pathlib.Path(rel_path)
    if p.is_absolute():
        candidate = p
    else:
        candidate = pathlib.Path(DATA_ROOT) / p
    try:
        resolved = candidate.resolve(strict=False)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid path")
    if not str(resolved).startswith(DATA_ROOT):
        raise HTTPException(status_code=403, detail="Access denied")
    if not resolved.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return resolved

def _audit(entry: dict):
    entry["ts"] = datetime.datetime.utcnow().isoformat() + "Z"
    logging.info(json.dumps(entry, ensure_ascii=False))

class ListFilesRequest(BaseModel):
    prefix: Optional[str] = ""
    max_items: Optional[int] = 500

@app.get("/tool/get_file")
async def get_file(path: str = Query(...), start: int = Query(0), max_chars: int = Query(8000), x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    resolved = _safe_resolve(path)
    try:
        with open(resolved, "r", encoding="utf-8", errors="replace") as fh:
            content = fh.read()
    except Exception:
        try:
            with open(resolved, "rb") as fh:
                data = fh.read()
            content = data.decode("utf-8", errors="replace")
        except Exception:
            raise HTTPException(status_code=500, detail="Could not read file")
    total = len(content)
    if start < 0:
        start = 0
    excerpt = content[start:start+max_chars]
    h = hashlib.sha256(content.encode("utf-8")).hexdigest()
    out = {
        "path": str(resolved),
        "requested_start": int(start),
        "returned_chars": len(excerpt),
        "total_chars": total,
        "sha256": h,
        "content": excerpt
    }
    _audit({"action": "get_file", "path": str(resolved), "start": int(start), "max_chars": int(max_chars)})
    return out

@app.post("/tool/list_files")
async def list_files(req: ListFilesRequest, x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    prefix = req.prefix or ""
    items = []
    base = pathlib.Path(DATA_ROOT)
    target = (base / prefix) if prefix else base
    try:
        resolved = target.resolve(strict=False)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid prefix")
    if not str(resolved).startswith(DATA_ROOT):
        raise HTTPException(status_code=403, detail="Access denied")
    for root, dirs, files in os.walk(resolved):
        for f in files:
            full = pathlib.Path(root) / f
            rel = str(full.relative_to(base))
            items.append(rel)
            if len(items) >= req.max_items:
                break
        if len(items) >= req.max_items:
            break
    _audit({"action": "list_files", "prefix": prefix, "returned": len(items)})
    return {"files": items}

@app.get("/tool/health")
async def health(x_api_key: Optional[str] = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return {"ok": True, "data_root": DATA_ROOT}
