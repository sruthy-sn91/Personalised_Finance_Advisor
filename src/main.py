import os
import requests
from bs4 import BeautifulSoup

from src.ingestion.pdf_ingestion import PDFIngestion
from src.rag.vector_store import VectorStore
from src.rag.retriever import Retriever
from src.services.sentiment_analysis import SentimentAggregator
from src.services.market_data import MarketDataService
from src.services.recommendation_engine import RecommendationEngine
from src.services.risk_assessment import RiskAssessment
from src.services.scenario_analysis import ScenarioAnalysis
from src.services.compliance import ComplianceChecker
from src.llm.finetune_lora import LoraFineTuner
from src.llm.rlhf_trainer import RLHFTrainer
from src.llm.inference import LLMInference
import os

class PersonalFinanceAdvisor:
    def __init__(self):
        self.pdf_ingester = PDFIngestion()
        self.vector_store = VectorStore()
        self.retriever = Retriever(self.vector_store)
        self.llm_inference = LLMInference()

        self.sentiment_aggregator = SentimentAggregator()
        self.market_data = MarketDataService()
        self.recommendation_engine = RecommendationEngine()
        self.risk_assessment = RiskAssessment()
        self.scenario_analysis = ScenarioAnalysis(self.market_data, self.sentiment_aggregator)
        self.compliance_checker = ComplianceChecker()
        self.lora_finetuner = LoraFineTuner()
        self.rlhf_trainer = RLHFTrainer()

        self.vector_store.load_or_create()

    def ask_finance_question(self, user_query, user_profile=None):
        context = self.retriever.get_relevant_docs(user_query)
        answer = self.llm_inference.generate_answer(user_query, context, user_profile=user_profile)
        return self.compliance_checker.add_disclaimer(answer)

    def ingest_pdf_and_index(self, pdf_path):
        text = self.pdf_ingester.extract_text_from_pdf(pdf_path)
        self.vector_store.add_document(text, doc_id=pdf_path)
        self.vector_store.save()

    def get_buy_sell_recommendation(self, symbol, user_profile=None):
        market_info = self.market_data.get_stock_data(symbol)
        sentiment = self.sentiment_aggregator.get_sentiment_for_ticker(symbol)
        return self.recommendation_engine.get_recommendation(symbol, market_info, sentiment, user_profile)

    def record_feedback_for_rlhf(self, user_query, model_response, user_feedback):
        self.rlhf_trainer.record_feedback(user_query, model_response, user_feedback)

    def run_scenario(self, portfolio, scenario_params):
        return self.scenario_analysis.run_scenario(portfolio, scenario_params)

    def compare_statements(self, compare_query: str):
        context = self.retriever.get_relevant_docs(compare_query)
        answer = self.llm_inference.generate_answer(compare_query, context)
        return self.compliance_checker.add_disclaimer(answer)

    def finetune_model_with_lora(self, training_dataset_path):
        self.lora_finetuner.finetune(training_dataset_path)


    
    def get_nse_index_performance(self):
        try:
            url = "https://www.google.com/finance/quote/NIFTY_50:INDEXNSE"
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, "html.parser")
            price_tag = soup.find("div", class_="YMlKec fxKbKc")
            change_tag = soup.find("div", class_="JwB6zf")
            if price_tag:
                price = price_tag.text.strip()
                change = change_tag.text.strip() if change_tag else ""
                return f"{price} {change}"
        except Exception as e:
            print(f"[ERROR] NSE fetch failed: {e}")
        return "Data unavailable"

    def get_bse_index_performance(self):
        try:
            url = "https://www.google.com/finance/quote/SENSEX:INDEXBOM"
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, "html.parser")
            price_tag = soup.find("div", class_="YMlKec fxKbKc")
            change_tag = soup.find("div", class_="JwB6zf")
            if price_tag:
                price = price_tag.text.strip()
                change = change_tag.text.strip() if change_tag else ""
                return f"{price} {change}"
        except Exception as e:
            print(f"[ERROR] BSE fetch failed: {e}")
        return "Data unavailable"

    def get_macro_brief(self):
        try:
            ex_response = requests.get("https://api.exchangerate-api.com/v4/latest/USD")
            ex_data = ex_response.json()
            usd_inr = float(ex_data["rates"]["INR"])
        except Exception as e:
            print("[ERROR] USD to INR:", e)
            usd_inr = None

        try:
            gold_api_key = os.getenv("GOLDAPI_API_KEY")
            headers = {"x-access-token": gold_api_key, "Content-Type": "application/json"}
            gold_response = requests.get("https://www.goldapi.io/api/XAU/USD", headers=headers)
            gold_data = gold_response.json()
            gold_usd = float(gold_data["price"])
            gold_rate = gold_usd * usd_inr * (10 / 31.1035) if usd_inr else None
        except Exception as e:
            print("[ERROR] Gold rate:", e)
            gold_rate = None

        try:
            # ✅ Silver API call (similar to gold)
            gold_api_key = os.getenv("GOLDAPI_API_KEY")
            headers = {"x-access-token": gold_api_key, "Content-Type": "application/json"}
            silver_response = requests.get("https://www.goldapi.io/api/XAG/USD", headers=headers)
            silver_data = silver_response.json()
            silver_usd = float(silver_data["price"])
            silver_rate = silver_usd * usd_inr * (10 / 31.1035) if usd_inr else None
        except Exception as e:
            print("[ERROR] Silver rate:", e)
            silver_rate = None

        try:
            oil_api_key = os.getenv("OIL_API_KEY")
            headers = {"Authorization": f"Token {oil_api_key}"}
            oil_response = requests.get("https://api.oilpriceapi.com/v1/prices/latest", headers=headers)
            oil_data = oil_response.json()
            crude_oil = float(oil_data["data"]["price"])
        except Exception as e:
            print("[ERROR] Crude oil rate:", e)
            crude_oil = None

        brief = "Today's Macro Brief:\n"
        brief += f"- USD to INR: {usd_inr:.2f}\n" if usd_inr else "- USD to INR: Unavailable\n"
        brief += f"- Gold: ₹{gold_rate:.2f}/10g\n" if gold_rate else "- Gold: Unavailable\n"
        brief += f"- Silver: ₹{silver_rate:.2f}/10g\n" if silver_rate else "- Silver: Unavailable\n"
        brief += f"- Crude Oil: ${crude_oil:.2f}/barrel\n" if crude_oil else "- Crude Oil: Unavailable\n"
        brief += f"- NIFTY 50: {self.get_nse_index_performance()}\n"
        brief += f"- SENSEX: {self.get_bse_index_performance()}\n"

        return brief


