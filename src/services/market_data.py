import yfinance as yf

class MarketDataService:
    def get_stock_data(self, symbol):
        try:
            # If the symbol does not contain a period and is likely for an Indian company,
            # append ".NS" (for the NSE). You can extend this logic for other markets as needed.
            if symbol.isalpha() and "." not in symbol:
                symbol = symbol.upper() + ".NS"
            
            stock = yf.Ticker(symbol)
            data = stock.info
            
            if data is None or "regularMarketPrice" not in data:
                raise ValueError(f"No market data found for symbol: {symbol}")
            
            # Create a simplified dictionary that includes the keys expected by the recommendation engine.
            result = {
                "current_price": data.get("regularMarketPrice"),
                "day_high": data.get("regularMarketDayHigh"),
                "day_low": data.get("regularMarketDayLow"),
                "volume": data.get("regularMarketVolume")
            }
            return result
        except Exception as e:
            raise Exception(f"Error fetching market data for {symbol}: {e}")
