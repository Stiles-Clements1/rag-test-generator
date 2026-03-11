import hashlib
import os
import uuid
from collections import OrderedDict
from threading import Lock
from typing import Dict

from flask import Flask, jsonify, render_template, request

from test_maker_rag import TestMakerRAG, grade_test, strip_answers


MATERIALS_DIR = os.getenv("MATERIALS_DIR", "Materials")
CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_NUM_QUESTIONS = int(os.getenv("DEFAULT_NUM_QUESTIONS", "8"))
MAX_CACHED_TESTS = int(os.getenv("MAX_CACHED_TESTS", "200"))

RAG_CACHE: Dict[str, TestMakerRAG] = {}
TEST_CACHE: "OrderedDict[str, dict]" = OrderedDict()
RAG_LOCK = Lock()
TEST_LOCK = Lock()


app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False


def _hash_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def _get_rag_engine(api_key: str) -> TestMakerRAG:
    key = (api_key or os.getenv("OPENAI_API_KEY", "")).strip()
    if not key:
        raise ValueError(
            "OpenAI API key is required. Add it in the page form or set OPENAI_API_KEY."
        )

    cache_key = _hash_key(key)
    with RAG_LOCK:
        if cache_key not in RAG_CACHE:
            RAG_CACHE[cache_key] = TestMakerRAG(
                materials_dir=MATERIALS_DIR,
                persist_directory=CHROMA_DIR,
                model_name=OPENAI_MODEL,
                api_key=key,
                rebuild_index=False,
            )
        return RAG_CACHE[cache_key]


def _store_test(test_payload: dict) -> str:
    test_id = uuid.uuid4().hex
    with TEST_LOCK:
        TEST_CACHE[test_id] = test_payload
        while len(TEST_CACHE) > MAX_CACHED_TESTS:
            TEST_CACHE.popitem(last=False)
    return test_id


@app.get("/")
def index():
    return render_template(
        "index.html",
        default_questions=DEFAULT_NUM_QUESTIONS,
        model_name=OPENAI_MODEL,
        materials_dir=MATERIALS_DIR,
    )


@app.get("/api/health")
def health() -> tuple:
    return jsonify({"status": "ok", "materials_dir": MATERIALS_DIR}), 200


@app.post("/api/generate-test")
def generate_test() -> tuple:
    payload = request.get_json(silent=True) or {}
    user_prompt = str(payload.get("prompt", "")).strip()
    api_key = str(payload.get("api_key", "")).strip()

    if not user_prompt:
        return jsonify({"error": "Prompt is required."}), 400

    raw_num_questions = payload.get("num_questions", DEFAULT_NUM_QUESTIONS)
    try:
        num_questions = max(3, min(30, int(raw_num_questions)))
    except (TypeError, ValueError):
        return jsonify({"error": "num_questions must be an integer."}), 400

    try:
        rag = _get_rag_engine(api_key)
        full_test = rag.generate_test(user_prompt, num_questions=num_questions)
        test_id = _store_test(full_test)
        public_test = strip_answers(full_test)
        return jsonify({"test_id": test_id, "test": public_test}), 200
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Failed to generate test: {exc}"}), 500


@app.post("/api/grade-test")
def grade_generated_test() -> tuple:
    payload = request.get_json(silent=True) or {}
    test_id = str(payload.get("test_id", "")).strip()
    api_key = str(payload.get("api_key", "")).strip()
    responses = payload.get("responses", {})

    if not test_id:
        return jsonify({"error": "test_id is required."}), 400
    if not isinstance(responses, dict):
        return jsonify({"error": "responses must be an object keyed by question id."}), 400

    with TEST_LOCK:
        test_payload = TEST_CACHE.get(test_id)

    if not test_payload:
        return jsonify({"error": "Test session not found. Generate a new test."}), 404

    normalized_responses = {str(key): str(value) for key, value in responses.items()}
    has_short_answers = any(
        q.get("type", "short_answer") == "short_answer"
        for q in test_payload.get("questions", [])
    )

    llm = None
    if has_short_answers:
        try:
            llm = _get_rag_engine(api_key).llm
        except ValueError:
            return (
                jsonify(
                    {
                        "error": (
                            "OpenAI API key is required to grade short-answer questions "
                            "with rubric scoring."
                        )
                    }
                ),
                400,
            )

    result = grade_test(test_payload, normalized_responses, llm=llm)
    return jsonify(result), 200


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=os.getenv("FLASK_DEBUG") == "1")
