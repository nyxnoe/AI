AI Decision Simulator
The AI Decision Simulator is an intelligent web application designed to help users analyze strategic scenarios by providing risk assessments, outcome probabilities, and actionable insights. Developed as a B.Tech CSE project at BRCM COET, it combines local rule-based logic with generative AI to offer robust decision support.

Features
Multi-Engine Analysis: Supports three distinct processing modes for decision simulation:

Anthropic (Claude): High-quality AI analysis using the Claude API.

Ollama: Integration with local LLMs (e.g., Llama 3, Mistral) for private processing.

Rule Engine: A synchronous, rule-based fallback that ensures the system is always operational without external dependencies.

Intelligent Routing: Automatically routes requests based on engine availability.

Data Visualization: Uses Chart.js to render Radar and Bar charts to visualize feasibility, ROI, risk, and efficiency.

History Tracking: Persists simulation results in a local SQLite database for future review.

Robust Input Validation: Sanitizes and validates user input (domain, budget, risk, time horizon) to ensure reliable model inference.

Setup & Installation
Prerequisites
Python 3.x

Ollama (Optional: for local LLM support)

Installation
Clone the repository and navigate to the directory.

Install the required dependencies:

Bash
pip install flask requests python-dotenv
Configure your environment variables by creating a .env file to store your ANTHROPIC_API_KEY.

Run the application:

Bash
python app.py
Project Structure
app.py: The main Flask backend. Orchestrates input processing, engine selection, and API interactions.

decision_engine.py: Contains the core rule-based logic for scoring decisions and generating outcomes.

data_processor.py: Handles validation and sanitization of form data.

database.py: Manages SQLite interactions, including schema initialization and simulation history.

ollama_engine.py: Manages streaming connections to the local Ollama API.

templates/ai-decision-simulator.html: The primary frontend interface, built with a dark-themed, responsive design.

Credits
Developers: Nakul (23-CSE-5305) & Deepanshu (24-CSE-5726).

Guide: Ms. Jyoti, CSE Dept, BRCM CET.