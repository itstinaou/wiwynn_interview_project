from pathlib import Path
import secrets

import pandas as pd
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from .pipeline import run_csv

app = FastAPI(title="Animal Segmentation Metrology API")
security = HTTPBasic()

TEST_CSV = Path("data/test_data.csv")
RESULTS_CSV = Path("outputs/results.csv")


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
async def measure(_: str = Depends(auth)):
    if not TEST_CSV.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Test CSV not found: {TEST_CSV}"
        )

    try:
        run_csv(TEST_CSV)

        if not RESULTS_CSV.exists():
            raise HTTPException(
                status_code=500,
                detail="Measurement failed: results.csv was not generated."
            )

        results = pd.read_csv(RESULTS_CSV).to_dict(orient="records")

        return {
            "input": str(TEST_CSV),
            "results": results
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
