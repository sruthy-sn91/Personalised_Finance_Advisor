# src/services/scenario_analysis.py

from src.services.market_data import MarketDataService
from src.services.sentiment_analysis import SentimentAggregator

class ScenarioAnalysis:
    def __init__(self, market_data_service: MarketDataService, sentiment_aggregator: SentimentAggregator):
        """
        market_data_service: instance for fetching current prices
        sentiment_aggregator: instance for performing sentiment analysis
        """
        self.market_data_service = market_data_service
        self.sentiment_aggregator = sentiment_aggregator

    def run_scenario(self, portfolio, scenario_params):
        """
        portfolio: a list of Pydantic PortfolioItem objects with attributes .symbol and .shares
        scenario_params: a Pydantic ScenarioParams object with attributes .recession (bool) and .tech_down (float)
        
        Returns a dict with key "portfolioImpact", which is a list of scenario analysis results.
        """
        results = []
        print("DEBUG: Received portfolio:", portfolio)
        print("DEBUG: Received scenario_params:", scenario_params)

        # Define a list of tech stock symbols that are subject to tech_down adjustment
        tech_symbols = ["INFY", "TCS", "WIPRO", "GOOGL", "MSFT", "AAPL", "META", "NFLX", "NVDA"]

        for stock in portfolio:
            # Since 'stock' is a Pydantic object, use dot notation:
            symbol = stock.symbol.upper()
            shares = stock.shares

            # 1. Get current market price (fallback to 100 if not available)
            try:
                data = self.market_data_service.get_stock_data(symbol)
                current_price = data.get("current_price", 100)
            except Exception as e:
                print(f"DEBUG: Error fetching market data for {symbol}: {e}")
                current_price = 100

            # 2. Apply scenario adjustments
            scenario_adjusted_price = current_price
            # If recession is True, reduce price by 10%
            if scenario_params.recession:
                scenario_adjusted_price *= 0.90

            # If tech downturn is provided and symbol is in tech list, adjust further
            if symbol in tech_symbols and scenario_params.tech_down > 0:
                scenario_adjusted_price *= (1 - scenario_params.tech_down)

            # 3. Get sentiment (use fallback "Neutral" if none returned)
            sentiment = self.sentiment_aggregator.get_sentiment_for_ticker(symbol) or "Neutral"

            # 4. For now, set probable end-of-month price equal to scenario_adjusted_price
            probable_end_of_month_price = scenario_adjusted_price

            # 5. Calculate percentage change from current price
            change_percent = ((scenario_adjusted_price - current_price) / current_price) * 100

            results.append({
                "symbol": symbol,
                "shares": shares,
                "currentPrice": round(current_price, 2),
                "scenarioAdjustedPrice": round(scenario_adjusted_price, 2),
                "sentiment": sentiment,
                "probableEndOfMonthPrice": round(probable_end_of_month_price, 2),
                "changePercent": round(change_percent, 2)
            })

        print("DEBUG: Returning results:", results)
        return {"portfolioImpact": results}
