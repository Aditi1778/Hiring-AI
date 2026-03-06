import logging
import os
import shutil
import sys
import traceback
from contextlib import asynccontextmanager
from datetime import datetime

import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from core.config import settings
from core.database import db_client
from core.security import get_current_user_id
from services.ai_matcher import process_resume_matching
from services.parser import parser_tool

load_dotenv()
# ---------------------------------------------------
# --- 1. PRODUCTION LOGGING CONFIGURATION ---
# ---------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("OXHIRE")


# ------------------------------------------------------------------
# Lifespan (Database & Agent Initialization)
# ------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    # [STARTUP LOGIC]
    logger.info("OxHireAI Server is starting up...")

    # check if api key valid or not
    if not settings.GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY is missing!")

    await db_client.connect_to_mongo()
    logger.info("🚀 OxHIRE AI Agents Initialized with Database")

    yield  # above code runs on startup and below on shutdown

    # [SHUTDOWN LOGIC]
    logger.info("OxHireAI Server is shutting down...")
    await db_client.close_mongo_connection()

    logger.info("✅ Database Connection Closed")
    # Yahan DB connections close karne ka kaam hota hai
    logger.info("Cleaning up resources...")


app = FastAPI(title="OxHireAI - Advanced Recruitment Engine", lifespan=lifespan)


# ------------------------------------------------------------------
# 🛡️ CORS Middleware Configuration
# ------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------
# GLOBAL EXCEPTION HANDLER ------------------
# ------------------------------------------------------------------


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Captures any unhandled error, logs the full traceback,
    and returns a clean response to the client.
    """
    error_msg = "".join(traceback.format_exception(None, exc, exc.__traceback__))
    logger.error(f"Unhandled Exception at {request.url.path}: \n{error_msg}")

    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "An internal server error occurred. Our team has been notified.",
            "detail": str(exc) if app.debug else "Internal Server Error",
        },
    )


# ------------------------------------------------------------------
# RESUME ANALYZER ENDPOINT ----------
# ------------------------------------------------------------------


@app.post("/analyze-resume")
async def analyze_resume(
    jd_text: str = Form(...),
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user_id),
):
    """
    This endpoint:
    1. Saves the uploaded resume temporarily.
    2. Calls ResumeParserTool to create a LlamaIndex.
    3. Triggers the CrewAI Agents for matching.
    """
    # 1. Temporary file saving (for processing)
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # 2. Step 1: Parsing (LlamaIndex + Sherpa/LlamaParse)
        # Background: This creates the "Smart Index" with layout context
        index = parser_tool.get_smart_index(temp_path)

        if not index:
            return {"error": "Failed to parse the document layout."}

        # 3. Step 2: Agentic Matching (CrewAI Agents)
        # Background: Agent 1 extracts info, Agent 2 matches with JD
        query_engine = index.as_query_engine()
        final_result = process_resume_matching(query_engine, jd_text)

        # Step 3: Ready data in mongoDB
        resume_document = {
            "user": user_id,
            "filename": file.filename,
            "jd_text": jd_text,
            "analysis_date": final_result["analysis_date"],
            "status": final_result["status"],
            "results": final_result["results"],  # Match Score and Reasoning into this
        }

        # Step 4: Insert into DB
        await db_client.db.analysis_results.insert_one(resume_document)

        return {
            "status": "success",
            "message": "Analysis saved to MongoDB",
            "data": final_result,
        }

    except Exception as e:
        return {"error": f"Internal Server Error: {str(e)}"}

    finally:
        # Clean up: Delete temp file after processing
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ------------------------------------------------------------------
# GET HISTORY OF ALL ANALYSIS ENDPOINT ----------
# ------------------------------------------------------------------


@app.get("/get-history")
async def get_history():
    """
    Fetch all previous resume analysis records from MongoDB.
    Returns a list of analysis results sorted by date (latest first).
    """
    try:
        # Initialize an empty list to store our records
        history_list = []

        # Access the collection and find all documents
        # Sort by 'analysis_date' in descending order (-1) ---> latest results on the top
        cursor = db_client.db.analysis_results.find({}).sort("analysis_date", -1)

        # Iterate through the cursor asynchronously
        async for document in cursor:
            # Convert MongoDB ObjectId to string for JSON compatibility
            document["_id"] = str(document["_id"])
            history_list.append(document)

        # Return structured response to the client
        return {
            "status": "success",
            "total_records": len(history_list),
            "data": history_list,
        }

    except Exception as e:
        # Log the error and return a failure message
        logger.error(f"⚔️Database error while fetching history: {str(e)}")
        return {
            "status": "error",
            "message": "Could not retrieve history from the database.",
        }


# ------------------------------------------------------------------
# STATUS AND HEALTH CHECK ---------
# ------------------------------------------------------------------


@app.get("/")
async def root():
    return {"message": "OXHIRE AI", "version": "1.0.0", "status": "running"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    # Use the dynamic port
    uvicorn.run(app, host="0.0.0.0", port=settings.PORT, reload=True)
