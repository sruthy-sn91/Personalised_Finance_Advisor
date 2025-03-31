# Personalized Finance Advisor

Personal Finance Advisor is a comprehensive, AI-powered tool designed to help users navigate complex financial decisions. By integrating advanced language model inference, parameter-efficient fine-tuning (PEFT), and reinforcement learning from human feedback (RLHF), this project delivers personalized financial advice and insights. It also includes capabilities for ingesting and analyzing financial documents, retrieving market data, and even interacting via voice commands.

This project demonstrates the integration of modern AI techniques and financial data analysis to create an end-to-end financial advisory system. Key functionalities include:

** LLM Inference: Use a state-of-the-art language model (via the Groq API) to answer finance-related questions.

** PEFT (LoRA Fine-Tuning): Fine-tune the language model in a parameter-efficient manner to adapt to the financial domain.

** RLHF: Record and use user feedback to iteratively improve the model.

** PDF Ingestion and RAG: Extract text from financial PDFs and create a searchable document repository using FAISS.

** Market Data & Sentiment Analysis: Retrieve real-time market data and perform sentiment analysis to support investment recommendations.

** Recommendation Engine: Provide actionable buy/sell/hold advice based on market data and sentiment.

**  Assessment & Scenario Analysis: Evaluate user risk profiles and simulate various market scenarios.

** Voice Assistant: Enable hands-free interaction with speech recognition and text-to-speech.

## Features

** Ask a Finance Question: Submit natural language questions and get answers powered by a fine-tuned language model.

** Upload Financial Statements: Ingest PDFs containing company financial statements and index their content for retrieval.

** Compare Company Statements: Compare data across multiple financial reports.

** Buy/Sell Recommendations: Receive tailored investment recommendations based on market data and sentiment.

** Scenario Analysis: Simulate market conditions (like a recession or tech downturn) to forecast portfolio impacts.

** Risk Assessment: Get personalized portfolio recommendations based on your risk tolerance.

** Voice Assistant: Interact using voice commands.

** Feedback Loop: Provide feedback to help refine and improve the model.

** Set Alerts & Preferences: Configure price alerts, financial goals, and user preferences.

## Technologies Used

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-brightgreen.svg)](https://fastapi.tiangolo.com/)
[![Uvicorn](https://img.shields.io/badge/Uvicorn-0.18+-blue.svg)](https://www.uvicorn.org/)
[![Hugging Face Transformers](https://img.shields.io/badge/HuggingFace-Transformers-orange.svg)](https://huggingface.co/transformers/)
[![FAISS](https://img.shields.io/badge/FAISS-Facebook-red.svg)](https://github.com/facebookresearch/faiss)
[![PyPDF2](https://img.shields.io/badge/PyPDF2-3.0-blueviolet.svg)](https://pythonhosted.org/PyPDF2/)
[![yfinance](https://img.shields.io/badge/yfinance-0.2.55-green.svg)](https://pypi.org/project/yfinance/)
[![SpeechRecognition](https://img.shields.io/badge/SpeechRecognition-3.8.1-yellow.svg)](https://pypi.org/project/SpeechRecognition/)
[![pyttsx3](https://img.shields.io/badge/pyttsx3-2.90-orange.svg)](https://pypi.org/project/pyttsx3/)
[![Jinja2](https://img.shields.io/badge/Jinja2-3.1.2-blue.svg)](https://palletsprojects.com/p/jinja/)
[![HTML5](https://img.shields.io/badge/HTML5-E34F26.svg?logo=html5&logoColor=white)](https://developer.mozilla.org/en-US/docs/Web/HTML)
[![CSS3](https://img.shields.io/badge/CSS3-1572B6.svg?logo=css3&logoColor=white)](https://developer.mozilla.org/en-US/docs/Web/CSS)
[![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E.svg?logo=javascript&logoColor=black)](https://developer.mozilla.org/en-US/docs/Web/JavaScript)


Python
FastAPI – Web framework for building API endpoints.
Uvicorn – ASGI server for running FastAPI applications.
Hugging Face Transformers & PEFT (LoRA) – For language model inference and efficient fine-tuning.
FAISS – For storing and searching document embeddings.
PyPDF2 – For PDF text extraction.
yfinance – For fetching market data.
SpeechRecognition & pyttsx3 – For voice interaction.
Jinja2, HTML, CSS, JavaScript – For front-end interface and styling.

## Project Structure

<img width="211" alt="Screenshot 2025-03-31 at 11 58 00 PM" src="https://github.com/user-attachments/assets/27db8e3e-ea60-4b10-ac06-bcff50b22087" />


## How to Run

1. Clone the repository

   git clone https://github.com/yourusername/PersonalFinanceAdvisor.git
   
   cd PersonalFinanceAdvisor

3.  Create a virtual environment and activate it:

    python -m venv venv

    source venv/bin/activate   # Windows: venv\Scripts\activate

    conda create -n virtenv -y
      
5. Install the dependencies

   pip install -r requirements.txt

6.  Configure environment variables

    Create a .env file in the root directory and add your API keys and configuration settings (e.g., GROQ_API_KEY, NEWS_API_KEY, etc.).
   
7.  Start the FastAPI server by running:

    python run.py
    
9.  Access the interface 

    http://127.0.0.1/8000/interface

## Functionalities

** Ask a Finance Question:

Submit your financial or investment queries via the form and receive AI-generated responses with compliance disclaimers.

** Upload PDFs:

Upload financial statements (PDFs) to ingest and index the data for analysis.

** Compare Company Statements:

Compare data across multiple financial documents by entering a comparison query.

** Buy/Sell Recommendations:

Enter a stock symbol (and optionally, your risk tolerance) to get personalized investment recommendations.

** Scenario Analysis:

Simulate various market conditions (e.g., recession, tech downturn) on your portfolio and see the projected impact.

** Risk Assessment:

Get an evaluation of your risk tolerance and receive a recommended portfolio strategy.

** Voice Assistant:

Use the voice assistant to ask questions and get audible responses.

** Feedback & Model Improvement:

Provide feedback on the AI responses to help improve the model through RLHF.

** Set Alerts & Preferences:

Configure price alerts and update your personal financial goals and preferences.

** Macro Brief:

Access a quick summary of macroeconomic conditions.

## Contributing

Contributions are welcome! Please fork this repository and submit pull requests. Ensure that you include tests and follow the project’s coding style guidelines.
