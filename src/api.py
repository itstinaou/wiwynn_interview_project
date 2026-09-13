from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi import Depends
import secrets
import shutil

from .pipeline import run_image

app = FastAPI(title="Animal Segmentation Metrology API")
security = HTTPBasic()

def auth(credentials: HTTPBasicCredentials = Depends(security)):
    ok_user = secrets.compare_digest(credentials.username, "demo")
    ok_pass = secrets.compare_digest(credentials.password, "demo123")
    if not (ok_user and ok_pass):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return credentials.username

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/measure")
async def measure(file: UploadFile = File(...), _: str = Depends(auth)):
    input_dir = Path("data/api_inputs")
    input_dir.mkdir(parents=True, exist_ok=True)
    path = input_dir / file.filename
    with path.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    rows = run_image(path)
    return {"image": file.filename, "results": rows}
