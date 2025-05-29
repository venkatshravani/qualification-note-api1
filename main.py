import os
import openai
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import sys
import io

# Load environment variables
load_dotenv()

# Ensure UTF-8 encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
os.environ["PYTHONIOENCODING"] = "utf-8"

# Configure Azure OpenAI
openai.api_type = "azure"
openai.api_key = os.getenv("AZURE_OPENAI_KEY")
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_version = "2024-02-15-preview"

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic schema
class OpportunityRequest(BaseModel):
    email_content: str

# API route
@app.post("/generate-qualification-note/")
async def generate_note(request: OpportunityRequest):
    prompt = """
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
""".strip()

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": request.email_content}
    ]

    try:
        response = openai.ChatCompletion.create(
            engine="gpt-4",  # Use your deployed model name
            messages=messages,
            temperature=0.3,
            max_tokens=1000
        )
        return {"qualification_note": response.choices[0].message["content"]}
    except Exception as e:
        return {"error": str(e)}
