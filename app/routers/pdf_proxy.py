import os
import tempfile
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from app.core.pdf_compressor import compress_pdf

router = APIRouter()


@router.post("/v1/documents/compress")
async def compress_pdf_upload(file: UploadFile = File(...)):
    """
    Accepts a PDF upload, runs the map-reduce compression pipeline,
    and returns the compressed markdown summary plus token stats.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are supported.")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name

        print(f"[PDF-PROXY] Compressing '{file.filename}'...")
        result = await compress_pdf(tmp_path)

        saved_pct = result["compression_ratio"]
        header = (
            f"🗜️ **PDF Compression** — `{file.filename}` "
            f"{result['original_tokens']:,} → {result['compressed_tokens']:,} tokens "
            f"(**{saved_pct}%** saved)\n\n---\n\n"
        )

        return JSONResponse({
            "filename": file.filename,
            "original_tokens": result["original_tokens"],
            "compressed_tokens": result["compressed_tokens"],
            "compression_ratio": saved_pct,
            "chat_ready_text": header + result["summary"],
            "summary": result["summary"]
        })

    except Exception as e:
        print(f"[PDF-PROXY] Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)