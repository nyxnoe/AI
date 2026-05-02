"""
app.py — AI Decision Simulator
Flask backend with:
  - Input validation via data_processor.py
  - Rule-based fallback via decision_engine.py
  - Hybrid AI mode toggle (local engine vs Anthropic LLM)
  - SQLite history logging
  - SSE streaming proxy to Anthropic
"""

import os
import json
import sqlite3
import re
import logging
from flask import Flask, render_template, request, Response, jsonify, g
from dotenv import load_dotenv
import requests

from data_processor import process_input, ValidationError, summarise
from decision_engine import make_decision

load_dotenv()

app = Flask(__name__)
app.config["DATABASE"] = os.path.join(app.root_path, "database.db")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

ANTHROPIC_API = "https://api.anthropic.com/v1/messages"
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")


# ── Database ──────────────────────────────────────────────────────────────────

def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(app.config["DATABASE"], detect_types=sqlite3.PARSE_DECLTYPES)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(error=None):
    db = g.pop("db", None)
    if db:
        db.close()

def init_db():
    db = get_db()
    db.execute("""
        CREATE TABLE IF NOT EXISTS simulations (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            domain      TEXT    NOT NULL,
            scenario    TEXT    NOT NULL,
            budget      INTEGER NOT NULL,
            risk        INTEGER NOT NULL,
            time_weeks  INTEGER NOT NULL,
            priority    TEXT    NOT NULL,
            engine_used TEXT    NOT NULL DEFAULT 'anthropic',
            result_json TEXT,
            risk_level  TEXT,
            confidence  INTEGER,
            created_at  REAL    DEFAULT (strftime('%s','now'))
        )
    """)
    db.commit()

def save_simulation(params, result, engine_used):
    try:
        db = get_db()
        db.execute(
            """INSERT INTO simulations
               (domain, scenario, budget, risk, time_weeks, priority,
                engine_used, result_json, risk_level, confidence)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                params["domain"], params["scenario"], params["budget"],
                params["risk"], params["time"], params["priority"],
                engine_used,
                json.dumps(result) if result else None,
                result.get("riskLevel") if result else None,
                result.get("confidenceScore") if result else None,
            ),
        )
        db.commit()
    except Exception as e:
        app.logger.warning(f"DB save failed: {e}")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("ai-decision-simulator.html")

@app.route("/api/simulate", methods=["POST"])
def simulate():
    data = request.get_json(force=True)

    # Step 2: Data Processing (data_processor.py)
    try:
        params = process_input(data)
    except ValidationError as e:
        return jsonify({"error": str(e)}), 400

    app.logger.info(f"Simulation: {summarise(params)}")

    # Mode toggle — frontend sends use_ai: true/false
    want_ai      = data.get("use_ai", True)
    use_anthropic = want_ai and bool(ANTHROPIC_KEY)

    if not use_anthropic:
        # Step 3+4: Local engine path (decision_engine.py)
        app.logger.info("Mode: local decision engine")
        result = make_decision(params).to_dict()
        save_simulation(params, result, "local")

        def local_stream():
            yield f"data: {json.dumps({'done': True, 'result': result, 'engine': 'local'})}\n\n"

        return Response(local_stream(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # Anthropic streaming path
    # Run local engine first to generate grounding context for the prompt
    preflight = make_decision(params)
    prompt    = build_prompt(params, preflight)

    def anthropic_stream():
        full_text = ""
        try:
            with requests.post(
                ANTHROPIC_API,
                headers={
                    "x-api-key":         ANTHROPIC_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type":      "application/json",
                },
                json={
                    "model":      "claude-sonnet-4-20250514",
                    "max_tokens": 1200,
                    "stream":     True,
                    "messages":   [{"role": "user", "content": prompt}],
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

            # Parse result
            result      = None
            engine_used = "anthropic"
            m = re.search(r"\{[\s\S]*\}", full_text)
            if m:
                try:
                    result = json.loads(m.group())
                except json.JSONDecodeError:
                    pass

            # Fallback to local engine if LLM returned garbage
            if not result:
                app.logger.warning("LLM unparseable — falling back to local engine")
                result      = preflight.to_dict()
                engine_used = "local_fallback"

            save_simulation(params, result, engine_used)
            yield f"data: {json.dumps({'done': True, 'result': result, 'engine': engine_used})}\n\n"

        except requests.exceptions.Timeout:
            app.logger.warning("Anthropic timeout — local fallback")
            result = preflight.to_dict()
            save_simulation(params, result, "local_fallback")
            yield f"data: {json.dumps({'done': True, 'result': result, 'engine': 'local_fallback'})}\n\n"

        except Exception as e:
            app.logger.error(f"Stream error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(anthropic_stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/history")
def history():
    db   = get_db()
    rows = db.execute(
        """SELECT id, domain, scenario, budget, risk, time_weeks, priority,
                  engine_used, risk_level, confidence, created_at
           FROM simulations ORDER BY created_at DESC LIMIT 20"""
    ).fetchall()
    return jsonify([dict(r) for r in rows])


@app.route("/api/history/<int:sim_id>")
def history_detail(sim_id):
    db  = get_db()
    row = db.execute("SELECT * FROM simulations WHERE id = ?", (sim_id,)).fetchone()
    if not row:
        return jsonify({"error": "Not found"}), 404
    d = dict(row)
    if d.get("result_json"):
        try:
            d["result"] = json.loads(d["result_json"])
        except Exception:
            d["result"] = None
    return jsonify(d)


@app.route("/api/history/<int:sim_id>", methods=["DELETE"])
def delete_history(sim_id):
    db = get_db()
    db.execute("DELETE FROM simulations WHERE id = ?", (sim_id,))
    db.commit()
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

Pre-computed baseline (rule-based engine — use as anchor, not constraint):
- Risk level: {pf['riskLevel']}
- Confidence: {pf['confidenceScore']}%
- Estimated ROI: {pf['expectedROI']}

Return this exact JSON (outcome probabilities must sum to approximately 100):
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
    init_db()

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
    