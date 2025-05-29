from fastapi import FastAPI, HTTPException, Form
from dotenv import load_dotenv
from typing import Optional
import openai
import os
import logging
 
app = FastAPI()
load_dotenv()
 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
 
openai.api_type = "azure"
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_version = os.getenv("AZURE_OPENAI_API_VERSION")
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT_NAME")
 
logger.info(f"Loaded deployment: {AZURE_DEPLOYMENT}")
 
@app.get("/")
def root():
    return {"message": "FastAPI is running! Welcome to qualification-note!"}
 
def create_prompt(email_body: str, attachment_text: Optional[str] = None) -> str:
    attachment_section = ""
    if attachment_text:
        attachment_section = f"\n\n**Attachment Content:**\n\"\"\"\n{attachment_text.strip()}\n\"\"\""
    prompt = f"""
System Instruction:
 
You are an assistant dedicated to generating structured, opportunity-specific Qualification Notes for bid teams. Your role is to extract details from the opportunity input provided and format the content into a well-defined output note. Follow the structure and tone precisely. Strictly do not hallucinate. Follow all the instructions mentioned.
 
Important Note:
 
-> Do not create or infer any information that is not clearly stated in the input, unless explicitly permitted.
 
-> Do not replace, rename, or hallucinate the customer name. Use the exact name provided in the input.
 
Format the output in the following structure:
 
- Introductory Sentence: [Brief sentence stating that this is an identified opportunity and you're sharing the details]
 
- Customer Overview
 
- Customer Details (as bullet points)
 
- Opportunity Overview
 
- Key Highlights (as bullet points)
 
- Timeline (as bullet point, if available)
 
Salutation:
 
Always start the output with:
 
"Dear Raj and Anthony,"
 
Then include:
 
Our team has identified an opportunity from potential client [Customer Name]
 
[Opportunity Name or Title], [Opportunity ID] - these are from the input provided. Opportunity Id would be something of Alphanumeric from input.
 
sharing the details as below for your reference:
 
Sections to Generate:
 
1. Customer Overview
 
You are allowed to search only for the exact [Customer Name] provided.
 
Provide a brief background (3–5 lines) specific to this customer only if found from credible sources.
 
If customer information cannot be reliably found, do not hallucinate. Instead write:
 
“Publicly available information on [Customer Name] is limited. The following overview is based on opportunity-specific context.”
 
If found, mention the company’s size, revenue, or digital initiatives only if relevant to the opportunity.
 
Avoid generic descriptions (e.g., “a leading global provider…”).
 
2. Opportunity Summary
 
Summarize from the input only in 3–4 bullet points, using only input information without interpretation or external assumptions:
 
- Nature of engagement (e.g., automation assessment, cloud migration)
 
- Client’s goal
 
- Expected deliverables
 
- Urgency, timing, or business context mentioned
 
3. Engagement Scope
 
Extract clearly stated or implied scope items only.
 
Mention tools, platforms, transformation stages, or CoE elements only if stated in the input.
 
Use short paragraphs or bullet points.
 
Do not invent or assume any deliverables.
 
4. Timeline
 
Extract dates, deadlines, or implementation phases (e.g., “Discovery in June,” “Go-Live in Q4”).
 
5. Key Highlights
 
Break into sub-sections:
 
Business Drivers:
 
List reasons for the engagement as per input (e.g., cost optimization, license rationalization)
 
Technology Preferences:
 
Mention any named platforms (e.g., UiPath, Power Automate) explicitly stated
 
Existing Tools or Processes:
 
Capture current bot count, maturity, governance, or legacy tools if provided
 
Use bullet format only. Be precise.
 
6. Qualification Scoring
 
Provide a 1-line justification for each. Be conservative.
 
Clarity: (Is the problem statement clear?)
 
Value: (Is there strong potential business value?)
 
Urgency: (Is there a timeline or compelling pressure?)
 
Formatting Rules:
 
- Use bold headers for each section
 
- Follow a formal, precise, client-facing tone
 
- Do not deviate from structure or add any commentary
 
- Use a professional tone, concise language, and bullet points where applicable
 
- Avoid excessive explanation. Focus on factual, high-level points suitable for internal opportunity briefs.
 
 
**Email Content:**
\"\"\"
{email_body.strip()}
\"\"\"
{attachment_section}
"""
    return prompt
 
@app.post("/generate-qualification-note")
async def generate_qualification_note(
    email_body: str = Form(...),
    attachment_text: Optional[str] = Form(None)
):
    try:
        prompt = create_prompt(email_body, attachment_text)
 
        logger.info("Sending prompt to Azure OpenAI...")
 
        response = openai.ChatCompletion.create(
            engine=AZURE_DEPLOYMENT,
            messages=[
                {"role": "system", "content": "You are a helpful assistant skilled in business documentation."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1500
        )
 
        result = response["choices"][0]["message"]["content"]
        logger.info("Response generated successfully.")
        return {"qualification_note": result}
 
    except Exception as e:
        logger.exception("Error while generating qualification note.")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)