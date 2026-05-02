"""
app.py — AI Decision Simulator
Flask backend with:
  - Input validation via data_processor.py
  - Rule-based fallback via decision_engine.py
  - Tri-engine selector: local | ollama (streaming) | anthropic (streaming)
  - Smart auto-routing when no engine specified
  - Per-request latency tracking stored in DB
  - SQLite history via database.py (zero duplication)
"""

import os
import json
import re
import time
import logging
from flask import Flask, render_template, request, Response, jsonify, g
from dotenv import load_dotenv
import requests

from data_processor import process_input, ValidationError, summarise
from decision_engine import make_decision
from database import (
    get_connection, init_schema,
    insert_simulation, fetch_history, fetch_simulation, delete_simulation,
)
from ollama_engine import stream_ollama, is_available

load_dotenv()

app = Flask(__name__)
app.config["DATABASE"] = os.path.join(app.root_path, "database.db")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

ANTHROPIC_API = "https://api.anthropic.com/v1/messages"
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")


# ── Database helpers (no logic duplicated from database.py) ───────────────────

def get_db():
    if "db" not in g:
        g.db = get_connection(app.config["DATABASE"])
    return g.db

@app.teardown_appcontext
def close_db(error=None):
    db = g.pop("db", None)
    if db:
        db.close()

def save_simulation(params, result, engine_used, latency_ms: int = 0):
    try:
        insert_simulation(get_db(), params, result, engine_used, latency_ms)
    except Exception as e:
        app.logger.warning(f"DB save failed: {e}")


# ── Smart engine auto-router ──────────────────────────────────────────────────

def select_engine(requested: str | None) -> str:
    """
    If the user explicitly picked an engine, honour it.
    Otherwise pick the best available engine automatically:
      1. Anthropic (highest quality, needs API key)
      2. Ollama    (local LLM, needs server running)
      3. local     (always available, instant)
    """
    if requested and requested in ("local", "ollama", "anthropic"):
        return requested
    # Auto-select
    if ANTHROPIC_KEY:
        return "anthropic"
    if is_available():
        return "ollama"
    return "local"


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("ai-decision-simulator.html")


@app.route("/api/simulate", methods=["POST"])
def simulate():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON or wrong Content-Type"}), 400

    try:
        params = process_input(data)
    except ValidationError as e:
        return jsonify({"error": str(e)}), 400

    app.logger.info(f"Simulation: {summarise(params)}")

    # Engine selection — respects explicit choice, auto-routes if absent/invalid
    engine      = select_engine(data.get("engine"))
    ollama_model = data.get("model", "llama3")   # frontend can optionally send model name

    app.logger.info(f"Engine: {engine}")

    # ── LOCAL ─────────────────────────────────────────────────────────────────
    if engine == "local":
        t0     = time.time()
        result = make_decision(params).to_dict()
        ms     = int((time.time() - t0) * 1000)
        save_simulation(params, result, "local", ms)

        def local_stream():
            yield f"data: {json.dumps({'done': True, 'result': result, 'engine': 'local', 'latency_ms': ms})}\n\n"

        return Response(local_stream(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # ── OLLAMA (streaming) ────────────────────────────────────────────────────
    if engine == "ollama":
        if not is_available():
            app.logger.warning("Ollama not running — falling back to local engine")
            result = make_decision(params).to_dict()
            save_simulation(params, result, "ollama_fallback", 0)

            def ollama_unavail_stream():
                yield f"data: {json.dumps({'done': True, 'result': result, 'engine': 'ollama_fallback', 'latency_ms': 0})}\n\n"

            return Response(ollama_unavail_stream(), mimetype="text/event-stream",
                            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

        preflight = make_decision(params)
        prompt    = build_prompt(params, preflight)

        def ollama_stream():
            t0        = time.time()
            gen       = stream_ollama(prompt, model=ollama_model)
            result    = None
            engine_used = "ollama"

            try:
                while True:
                    delta = next(gen)
                    yield f"data: {json.dumps({'delta': delta})}\n\n"
            except StopIteration as e:
                result = e.value

            ms = int((time.time() - t0) * 1000)

            if not result:
                app.logger.warning("Ollama stream returned no parseable JSON — local fallback")
                result      = preflight.to_dict()
                engine_used = "ollama_fallback"

            save_simulation(params, result, engine_used, ms)
            yield f"data: {json.dumps({'done': True, 'result': result, 'engine': engine_used, 'latency_ms': ms})}\n\n"

        return Response(ollama_stream(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # ── ANTHROPIC (streaming) ─────────────────────────────────────────────────
    if not ANTHROPIC_KEY:
        app.logger.warning("No Anthropic key — auto-falling back to local engine")
        result = make_decision(params).to_dict()
        save_simulation(params, result, "local_fallback", 0)

        def no_key_stream():
            yield f"data: {json.dumps({'done': True, 'result': result, 'engine': 'local_fallback', 'latency_ms': 0})}\n\n"

        return Response(no_key_stream(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    preflight = make_decision(params)
    prompt    = build_prompt(params, preflight)

    def anthropic_stream():
        full_text   = ""
        t0          = time.time()
        engine_used = "anthropic"
        result      = None

        try:
            with requests.post(
                ANTHROPIC_API,
                headers={
                    "x-api-key":         ANTHROPIC_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type":      "application/json",
                },
                json={
                    "model":    "claude-sonnet-4-20250514",
                    "max_tokens": 1200,
                    "stream":   True,
                    "messages": [{"role": "user", "content": prompt}],
                },
                stream=True,
                timeout=60,
            ) as resp:
                if resp.status_code != 200:
                    raise RuntimeError(f"Anthropic {resp.status_code}: {resp.text[:200]}")

                for raw_line in resp.iter_lines():
                    if not raw_line:
                        continue
                    line = raw_line.decode("utf-8")
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                        delta = chunk.get("delta", {}).get("text", "")
                        if delta:
                            full_text += delta
                            yield f"data: {json.dumps({'delta': delta})}\n\n"
                    except json.JSONDecodeError:
                        pass

            ms = int((time.time() - t0) * 1000)
            m  = re.search(r"\{[\s\S]*\}", full_text)
            if m:
                try:
                    result = json.loads(m.group())
                except json.JSONDecodeError:
                    pass

            if not result:
                app.logger.warning("LLM unparseable — local fallback")
                result      = preflight.to_dict()
                engine_used = "local_fallback"

            save_simulation(params, result, engine_used, ms)
            yield f"data: {json.dumps({'done': True, 'result': result, 'engine': engine_used, 'latency_ms': ms})}\n\n"

        except requests.exceptions.Timeout:
            app.logger.warning("Anthropic timeout — local fallback")
            result = preflight.to_dict()
            ms = int((time.time() - t0) * 1000)
            save_simulation(params, result, "local_fallback", ms)
            yield f"data: {json.dumps({'done': True, 'result': result, 'engine': 'local_fallback', 'latency_ms': ms})}\n\n"

        except Exception as e:
            app.logger.error(f"Stream error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(anthropic_stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/history")
def history():
    return jsonify(fetch_history(get_db()))


@app.route("/api/history/<int:sim_id>")
def history_detail(sim_id):
    row = fetch_simulation(get_db(), sim_id)
    if not row:
        return jsonify({"error": "Not found"}), 404
    return jsonify(row)


@app.route("/api/history/<int:sim_id>", methods=["DELETE"])
def delete_history(sim_id):
    if not delete_simulation(get_db(), sim_id):
        return jsonify({"error": "Not found"}), 404
    return jsonify({"deleted": sim_id})


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_prompt(p, preflight):
    pf = preflight.to_dict()
    return f"""You are an AI Decision Simulator. Analyze this scenario and respond ONLY with valid JSON.

Scenario: "{p['scenario']}"
Domain: {p['domain']}
Budget: \u20b9{p['budget']} Lakhs
Risk Tolerance: {p['risk']}/10
Time Horizon: {p['time']} weeks
Priority: {p['priority']}

Pre-computed baseline (rule-based engine):
- Risk level: {pf['riskLevel']}
- Confidence: {pf['confidenceScore']}%
- Estimated ROI: {pf['expectedROI']}

Return this exact JSON (outcome probabilities must sum to ~100):
{{
  "summary": "2-sentence executive summary",
  "riskLevel": "Low|Medium|High|Critical",
  "confidenceScore": <integer 60-97>,
  "expectedROI": "<e.g. 12-18%>",
  "outcomes": [
    {{"rank":1,"title":"Primary Recommendation","description":"2-3 sentences","probability":<50-85>,"color":"green","badge":"Recommended"}},
    {{"rank":2,"title":"Alternative Strategy","description":"2-3 sentences","probability":<15-35>,"color":"amber","badge":"Alternative"}},
    {{"rank":3,"title":"Conservative Fallback","description":"2-3 sentences","probability":<5-20>,"color":"red","badge":"Fallback"}}
  ],
  "radarData": {{
    "labels": ["Feasibility","ROI Potential","Risk Control","Time Efficiency","Alignment"],
    "values": [<5 integers 40-95>]
  }},
  "insight": "One specific actionable insight for this exact scenario.",
  "keyRisks": ["Risk 1","Risk 2","Risk 3"]
}}"""


# ── Entry ─────────────────────────────────────────────────────────────────────

with app.app_context():
    init_schema(get_connection(app.config["DATABASE"]))

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=6000)