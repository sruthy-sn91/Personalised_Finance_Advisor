from src.ingestion.pdf_ingestion import PDFIngestion
from src.rag.vector_store import VectorStore
from src.rag.retriever import Retriever
from src.llm.inference import LLMInference
from src.services.sentiment_analysis import SentimentAggregator
from src.services.market_data import MarketDataService
from src.services.recommendation_engine import RecommendationEngine
from src.services.risk_assessment import RiskAssessment
from src.services.scenario_analysis import ScenarioAnalysis
from src.services.compliance import ComplianceChecker
from src.llm.finetune_lora import LoraFineTuner
from src.llm.rlhf_trainer import RLHFTrainer

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
        # Pass market_data and sentiment_aggregator to ScenarioAnalysis
        self.scenario_analysis = ScenarioAnalysis(self.market_data, self.sentiment_aggregator)
        self.compliance_checker = ComplianceChecker()
        self.lora_finetuner = LoraFineTuner()
        self.rlhf_trainer = RLHFTrainer()

        # Ensure the vector store is ready (create or load existing)
        self.vector_store.load_or_create()

        # Placeholders for user data
        self.user_goals = ""
        self.user_preferences = {}
        self.alerts = []

    def ask_finance_question(self, user_query, user_profile=None):
        # Retrieve relevant docs from vector store
        context = self.retriever.get_relevant_docs(user_query)
        # Generate answer using LLM with context
        answer = self.llm_inference.generate_answer(user_query, context, user_profile=user_profile)
        # Add disclaimers for compliance
        answer = self.compliance_checker.add_disclaimer(answer)
        return answer

    def ingest_pdf_and_index(self, pdf_path):
        # Extract text from PDF
        text = self.pdf_ingester.extract_text_from_pdf(pdf_path)
        # Add to vector store
        self.vector_store.add_document(text, doc_id=pdf_path)
        self.vector_store.save()

    def get_buy_sell_recommendation(self, symbol, user_profile=None):
        # Retrieve market data
        market_info = self.market_data.get_stock_data(symbol)
        # Get sentiment from news/social
        sentiment = self.sentiment_aggregator.get_sentiment_for_ticker(symbol)
        # Generate recommendation
        recommendation = self.recommendation_engine.get_recommendation(symbol, market_info, sentiment, user_profile)
        return recommendation

    def record_feedback_for_rlhf(self, user_query, model_response, user_feedback):
        self.rlhf_trainer.record_feedback(user_query, model_response, user_feedback)

    def run_scenario(self, portfolio, scenario_params):
        return self.scenario_analysis.run_scenario(portfolio, scenario_params)

    def assess_risk(self, user_profile):
        return self.risk_assessment.assess(user_profile)

    def set_personal_goals(self, goals: str):
        self.user_goals = goals
        return f"Personal goals updated: {goals}"

    def compare_statements(self, compare_query: str):
        # For now, we feed the query to the LLM with retrieved context.
        context = self.retriever.get_relevant_docs(compare_query)
        answer = self.llm_inference.generate_answer(compare_query, context)
        answer = self.compliance_checker.add_disclaimer(answer)
        return answer

    def get_macro_brief(self):
        brief = (
            "Today's Macro Brief:\n"
            "- Interest rates stable at 4.5%\n"
            "- Inflation expected to decrease by 0.1%\n"
            "- USD strengthening against EUR\n"
        )
        return brief

    def set_alert(self, symbol: str, threshold: float):
        self.alerts.append({"symbol": symbol, "threshold": threshold})
        return f"Alert set for {symbol} at price {threshold}"

    def set_preferences(self, preferences: str):
        prefs_dict = {}
        for item in preferences.split(","):
            if ":" in item:
                k, v = item.split(":")
                prefs_dict[k.strip()] = v.strip()
        self.user_preferences.update(prefs_dict)
        return f"Preferences updated: {prefs_dict}"

    def finetune_model_with_lora(self, training_dataset_path):
        self.lora_finetuner.finetune(training_dataset_path)

    def retrain_with_feedback(self):
        self.rlhf_trainer.retrain_with_feedback()
