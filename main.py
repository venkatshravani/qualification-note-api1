from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
import logging
import os
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Azure OpenAI credentials from environment
api_key = os.getenv("AZURE_OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("AZURE_OPENAI_API_KEY environment variable is not set!")

client = AzureOpenAI(
    api_key=api_key,
    api_version="2024-12-01-preview",
    azure_endpoint="https://qualification01.openai.azure.com/"
)

# FastAPI app
app = FastAPI()

@app.get("/")
async def root():
    return {"message": "FastAPI Qualification Note API is running."}

# Pydantic model
class EmailText(BaseModel):
    email_text: str

    @validator('email_text')
    def not_empty(cls, v):
        if not v.strip():
            raise ValueError("email_text cannot be empty.")
        return v

@app.post("/generate-qualification-note")
async def generate_note(request: EmailText):
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
        response = client.chat.completions.create(
            model="gpt-4",  # match your Azure OpenAI deployment name
            messages=[
                {"role": "system", "content": "You are a helpful assistant skilled in writing qualification notes."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        llm_generated_text = response.choices[0].message.content

        # Add custom salutation and context before the generated note
        header = (
            "**Dear Raj and Anthony,**  \n\n"
            "Our team has identified an opportunity from the potential client **Edwards Lifesciences**. "
            "The scope involves **Assessment + Migration from UiPath to Microsoft Power Automate**. "
            "I am sharing the details below for your reference:\n\n"
        )

        response_text = header + llm_generated_text

        return {"qualification_note": response.choices[0].message.content}

    except Exception as e:
        logging.error(f"OpenAI API error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")
