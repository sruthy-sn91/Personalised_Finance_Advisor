class RecommendationEngine:
    def get_recommendation(self, symbol, market_info, sentiment, user_profile=None):
        price = market_info["current_price"]
        # Example logic
        if sentiment == "Positive" and price < 105:
            return f"Recommendation: BUY {symbol} based on positive sentiment and current price at {price}."
        else:
            return f"Recommendation: SELL or HOLD {symbol}, sentiment = {sentiment}, price = {price}."
