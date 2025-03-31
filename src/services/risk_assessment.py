class RiskAssessment:
    def assess(self, user_profile):
        # user_profile might have net_worth, risk_tolerance, investment_horizon, etc.
        risk_level = user_profile.get("risk_tolerance", "moderate")
        if risk_level == "high":
            return "Aggressive portfolio recommended."
        elif risk_level == "low":
            return "Conservative portfolio recommended."
        else:
            return "Balanced portfolio recommended."
