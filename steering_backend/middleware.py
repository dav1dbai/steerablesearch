from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import List

def add_cors_middleware(app: FastAPI, origins: List[str] = None):
    """
    Add CORS middleware to FastAPI application
    """
    if origins is None:
        origins = ["*"]  # Allow all origins for easier development
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    return app