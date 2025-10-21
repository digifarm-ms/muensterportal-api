from fastapi import FastAPI, Query
from typing import Optional

app = FastAPI(title="Muenster4You API", version="0.1.0")

@app.get("/")
async def root():
    return {"message": "Hello from muenster4you!"}

@app.get("/search")
async def search(q: Optional[str] = Query(None, description="Search query string")):
    if not q:
        return {"message": "Please provide a search query", "query": None}

    return {
        "message": f"Search results for: {q}",
        "query": q,
        "results": [
            f"Mock result 1 for '{q}'",
            f"Mock result 2 for '{q}'",
            f"Mock result 3 for '{q}'"
        ]
    }

def main() -> None:
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
