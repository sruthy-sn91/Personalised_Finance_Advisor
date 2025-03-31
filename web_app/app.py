from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os
from src.main import PersonalFinanceAdvisor
from src.voice.voice_assistant import VoiceAssistant
from pydantic import BaseModel
from typing import List

app = FastAPI()

# Ensure directories exist (for FAISS and PDFs)
os.makedirs("./data/vector_store", exist_ok=True)
os.makedirs("./data", exist_ok=True)

# Mount static files (CSS, images, etc.)
app.mount("/static", StaticFiles(directory="web_app/static"), name="static")
templates = Jinja2Templates(directory="web_app/templates")

advisor = PersonalFinanceAdvisor()
voice_assistant = VoiceAssistant()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Personal Finance Advisor!"}

@app.get("/interface")
def interface(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Ask a Finance or Investment Question
@app.post("/ask")
def ask_question(
    user_query: str = Form(...),
    risk_tolerance: str = Form(None)
):
    user_profile = {"risk_tolerance": risk_tolerance} if risk_tolerance else {}
    answer = advisor.ask_finance_question(user_query, user_profile)
    return {"answer": answer}

# Upload Company Financial Statements (multiple PDFs)
@app.post("/upload_pdf")
async def upload_pdf(files: list[UploadFile] = File(...)):
    messages = []
    for file in files:
        contents = await file.read()
        pdf_path = f"./data/{file.filename}"
        with open(pdf_path, "wb") as f:
            f.write(contents)
        advisor.ingest_pdf_and_index(pdf_path)
        messages.append(f"{file.filename} successfully ingested.")
    return {"message": " ; ".join(messages)}

# Compare Company Statements
@app.post("/compare")
def compare_statements(compare_query: str = Form(...)):
    answer = advisor.compare_statements(compare_query)
    return {"answer": answer}

# Buy/Sell Recommendations
@app.post("/recommend")
def recommend(
    symbol: str = Form(...),
    risk_tolerance: str = Form(None)
):
    user_profile = {"risk_tolerance": risk_tolerance} if risk_tolerance else {}
    recommendation = advisor.get_buy_sell_recommendation(symbol, user_profile)
    return {"recommendation": recommendation}

# RLHF Feedback
@app.post("/feedback")
def feedback(
    user_query: str = Form(...),
    model_response: str = Form(...),
    user_feedback: str = Form(...)
):
    advisor.record_feedback_for_rlhf(user_query, model_response, user_feedback)
    return {"message": "Feedback recorded. Thank you!"}

# Set Personal Financial Goals
@app.post("/goals")
def set_goals(goals: str = Form(...)):
    result = advisor.set_personal_goals(goals)
    return {"message": result}

# -------------------------------
# Updated Scenario Analysis Endpoint Using JSON
# -------------------------------

class PortfolioItem(BaseModel):
    symbol: str
    shares: float

class ScenarioParams(BaseModel):
    recession: bool
    tech_down: float

class ScenarioPayload(BaseModel):
    portfolio: List[PortfolioItem]
    scenario_params: ScenarioParams

@app.post("/scenario")
async def scenario(payload: ScenarioPayload):
    # FastAPI automatically parses the incoming JSON into payload
    result = advisor.run_scenario(payload.portfolio, payload.scenario_params)
    # Return the result directly (it should include "portfolioImpact")
    return result

# Risk Assessment
@app.post("/risk_assessment")
def risk_assessment(risk_tolerance: str = Form(...)):
    user_profile = {"risk_tolerance": risk_tolerance}
    result = advisor.assess_risk(user_profile)
    return {"risk_assessment": result}

# Macroeconomic Brief
@app.get("/macro")
def macro_brief():
    brief = advisor.get_macro_brief()
    return {"brief": brief}

# Set Price Alerts
@app.post("/alerts")
def set_alert(
    symbol: str = Form(...),
    threshold: float = Form(...)
):
    message = advisor.set_alert(symbol, threshold)
    return {"message": message}

# Set User Preferences with separate fields
@app.post("/preferences")
def set_preferences(
    risk_aversion: str = Form(...),
    favorite_sectors: str = Form(...)
):
    sectors = [s.strip() for s in favorite_sectors.split(",") if s.strip()]
    result = advisor.set_preferences(f"risk_aversion:{risk_aversion}, favorite_sectors:{','.join(sectors)}")
    return {"message": result}

# Voice Assistant
@app.post("/listen")
def voice_listen():
    try:
        query = voice_assistant.listen()
        if query:
            answer = advisor.ask_finance_question(query)
            voice_assistant.speak(answer)
            return {"answer": answer}
        return {"answer": "No query detected."}
    except Exception as e:
        return {"answer": f"Error with voice assistant: {str(e)}"}
