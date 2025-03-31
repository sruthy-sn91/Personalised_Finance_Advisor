class ComplianceChecker:
    def add_disclaimer(self, answer):
        disclaimer = (
            "\n\nDisclaimer: This information is provided for educational purposes "
            "and does not constitute financial advice. Always consult a licensed professional."
        )
        return answer + disclaimer
