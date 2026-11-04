"""FastAPI application: REST API + static dashboard on localhost."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend import config as C
from backend.api.routes import router

app = FastAPI(title="Smart Grid Fault Recovery (CNN + DQN + Llama 3.2)", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(127\.0\.0\.1|localhost)(:\d+)?",   # any local port (see run.py --port)
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)
app.mount("/results", StaticFiles(directory=C.RESULTS_DIR), name="results")
app.mount("/", StaticFiles(directory=C.FRONTEND_DIR, html=True), name="frontend")
