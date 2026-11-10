"""Start the local review/demo API; one model instance per worker."""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("healthpal_ai.app:app", host="127.0.0.1", port=8001, workers=1)
