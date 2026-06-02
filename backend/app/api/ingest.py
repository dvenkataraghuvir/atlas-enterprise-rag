from fastapi import APIRouter, UploadFile, File

router = APIRouter()

@router.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    # TODO: parse and embed document
    return {"filename": file.filename, "status": "pending"}
