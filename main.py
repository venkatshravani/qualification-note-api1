import os
import io
import sys
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import openai

# 1. Enforce UTF-8 on stdout/stderr (good)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# 2. Load env vars
load_dotenv()

openai.api_type    = "azure"
openai.api_key     = os.getenv("AZURE_OPENAI_API_KEY")
openai.api_base    = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_version = os.getenv("AZURE_OPENAI_API_VERSION")
deployment_name    = os.getenv("AZURE_DEPLOYMENT_NAME")

# 3. Create FastAPI app with explicit docs/openapi URLs and root_path if needed
# If your app is served at root URL (no subpath), root_path can stay empty ""
app = FastAPI(
    title="Qualification Note API",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    root_path=""  # Change this if your API is served under a prefix, e.g. "/api"
)

# 4. CORS middleware (keep as is, but consider restricting allow_origins in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, set specific origins here
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 5. Pydantic model
class OpportunityRequest(BaseModel):
    email_content: str
    attachment_text: str | None = None

# 6. System prompt
SYSTEM_PROMPT = """
System Instruction:

You are an assistant generating structured, opportunity-specific Qualification Notes.

Format:

- Introductory Sentence

- Customer Overview

- Customer Details (bullets)

- Opportunity Overview

- Key Highlights (bullets)

- Timeline (bullets if available)

Start with:

Dear Raj and Anthony,

Sections:

1. Customer Overview

2. Opportunity Summary (3-4 bullets from input)

3. Engagement Scope (bullets)

4. Timeline

5. Key Highlights

6. Qualification Scoring

Use bold headers, formal tone, and bullet points. Do not hallucinate.
"""

# 7. API route
@app.post("/generate-qualification-note/")
async def generate_note(req: OpportunityRequest):
    combined = req.email_content
    if req.attachment_text:
        combined += f"\n\nAttachment Content:\n{req.attachment_text}"

    messages = [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": combined},
    ]

    try:
        resp = openai.ChatCompletion.create(
            engine=deployment_name,
            messages=messages,
            temperature=0.3,
            max_tokens=1500
        )
        note = resp.choices[0].message["content"]
        return JSONResponse(content={"qualification_note": note})

    except openai.error.OpenAIError:
        logging.exception("OpenAI API error")
        raise HTTPException(status_code=502, detail="Failed to generate qualification note.")

    except Exception:
        logging.exception("Unexpected error")
        raise HTTPException(status_code=500, detail="Internal server error.")

 