from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
import openai
import uvicorn
import logging

# Azure OpenAI configuration
openai.api_type = "azure"
openai.api_base = "https://qualification01.openai.azure.com/"
openai.api_version = "2024-12-01-preview"
import os

api_key = os.getenv("AZURE_OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("AZURE_OPENAI_API_KEY environment variable is not set!")


# Setup logging
logging.basicConfig(level=logging.INFO)

# FastAPI app
app = FastAPI()

# Root GET endpoint to verify server is running
@app.get("/")
async def read_root():
    return {"message": "FastAPI server is running"}

# Request schema with validation
class EmailText(BaseModel):
    email_text: str

    @validator('email_text')
    def must_not_be_empty(cls, v):
        if not v or v.strip() == "":
            raise ValueError('email_text cannot be empty')
        return v

@app.post("/generate-qualification-note")
async def generate_qualification_note(request: EmailText):
    logging.info(f"Received email_text (first 50 chars): {request.email_text[:50]}...")
    
    prompt = f"""
    Dear Team,

    Please find below the Qualification Note prepared based on the provided opportunity content.

    ---

    **1. Customer Overview**  
    [Provide a brief overview of the customer.]

    **2. Opportunity Summary**  
    Summarize the opportunity in 3-4 bullet points.

    **3. Engagement Scope**  
    Mention key areas or deliverables in this engagement.

    **4. Timeline**  
    If mentioned, extract expected start/end or duration details.

    **5. Key Highlights**  
    - Business drivers  
    - Technology preferences  
    - Existing tools or processes

    **6. Qualification Scoring**  
    Provide a score between 1-10 based on clarity, value, fit, and urgency.

    ---

    Here is the extracted content:
    \"\"\"{request.email_text}\"\"\"

    Please generate the above Qualification Note in the same format.
    """

    try:
        response = openai.ChatCompletion.create(
            engine="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant skilled in business documentation."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )

        qualification_note = response['choices'][0]['message']['content']
        logging.info("OpenAI API call successful, returning qualification note.")
        return {"qualification_note": qualification_note}

    except openai.error.OpenAIError as e:
        logging.error(f"OpenAI API error: {e}")
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {e}")

    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Run locally
if __name__ == "__main__":
   uvicorn.run(app, host="0.0.0.0", port=8000) 

