import sys
import io
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import AzureOpenAI
import logging

# === Set UTF-8 encoding explicitly ===
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
os.environ["PYTHONIOENCODING"] = "utf-8"

# === FastAPI app setup ===
app = FastAPI()

# === CORS Middleware ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Request schema ===
class QualificationRequest(BaseModel):
    email_body: str
    attachment_text: str

# === Azure OpenAI Client Setup ===
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-02-15-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4")

# === Route to Generate Qualification Note ===
@app.post("/generate-qualification-note")
def generate_qualification_note(request: QualificationRequest):
    try:
        # === Enhanced Prompt ===
        prompt = f"""
System Instruction:

You are an assistant dedicated to generating structured, opportunity-specific Qualification Notes for bid teams. Your role is to extract details from the opportunity input provided and format the content into a well-defined output note. Follow the structure and tone precisely. Strictly do not hallucinate. Follow all the instructions mentioned.

Important Note:
-> Do not create or infer any information that is not clearly stated in the input, unless explicitly permitted.
-> Do not replace, rename, or hallucinate the customer name. Use the exact name provided in the input.

Format the output in the following structure:

Dear Raj and Anthony,

Our team has identified an opportunity from potential client [Customer Name]

[Opportunity Name or Title], [Opportunity ID]

sharing the details as below for your reference:

**Customer Overview**  
(See logic below)

**Customer Details**
- Bullet points if provided

**Opportunity Overview**
- Nature of engagement  
- Client’s goal  
- Expected deliverables  
- Urgency, timing, or business context

**Engagement Scope**
- Stated or implied scope  
- Tools, platforms, stages (if mentioned)

**Timeline**
- Extracted timeline/milestones (if available)

**Key Highlights**

Business Drivers:
- Reasons for engagement

Technology Preferences:
- Named tools/platforms

Existing Tools or Processes:
- Current state

**Qualification Scoring**
Clarity: (1-line justification)  
Value: (1-line justification)  
Urgency: (1-line justification)

Avoid generic statements. Use precise, client-facing tone. Do not assume or hallucinate. Extract only from input.

EMAIL BODY:
{request.email_body}

ATTACHMENT TEXT:
{request.attachment_text}
"""

        # === OpenAI Call ===
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are an assistant that strictly generates structured qualification notes based only on the input below. Never hallucinate or assume. Follow formatting rules exactly."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3
        )

        qualification_note = response.choices[0].message.content.strip()
        qualification_note = qualification_note.encode("utf-8", "ignore").decode("utf-8")

        return {"qualification_note": qualification_note}

    except Exception as e:
        logging.exception("Error generating qualification note")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


