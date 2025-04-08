from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
from threading import Thread
import os

from src.main import PersonalFinanceAdvisor
from src.voice.voice_assistant import VoiceAssistant

app = FastAPI()

# Ensure necessary directories exist
os.makedirs("./data/vector_store", exist_ok=True)
os.makedirs("./data", exist_ok=True)

# Static and Templates
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

@app.post("/ask")
def ask_question(user_query: str = Form(...), risk_tolerance: str = Form(None)):
    user_profile = {"risk_tolerance": risk_tolerance} if risk_tolerance else {}
    answer = advisor.ask_finance_question(user_query, user_profile)
    return {"answer": answer}

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

@app.post("/compare")
def compare_statements(compare_query: str = Form(...)):
    answer = advisor.compare_statements(compare_query)
    return {"answer": answer}

@app.post("/recommend")
def recommend(symbol: str = Form(...), risk_tolerance: str = Form(None)):
    user_profile = {"risk_tolerance": risk_tolerance} if risk_tolerance else {}
    recommendation = advisor.get_buy_sell_recommendation(symbol, user_profile)
    return {"recommendation": recommendation}

@app.post("/feedback")
def feedback(user_query: str = Form(...), model_response: str = Form(...), user_feedback: str = Form(...)):
    advisor.record_feedback_for_rlhf(user_query, model_response, user_feedback)
    return {"message": "Feedback recorded. Thank you!"}

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
    return advisor.run_scenario(payload.portfolio, payload.scenario_params)

@app.post("/finetune")
def trigger_finetune():
    def run():
        advisor.lora_finetuner.finetune("datasets/lora_training_data.json")
    Thread(target=run).start()
    return {"message": "LoRA fine-tuning started in background."}

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

@app.get("/macro")
def macro_brief():
    try:
        brief = advisor.get_macro_brief()
        return {"brief": brief}
    except Exception as e:
        return {"error": str(e)}
