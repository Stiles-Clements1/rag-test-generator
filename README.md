# TestMaker RAG

A retrieval-augmented test generator for course materials (`PDF`, `R`, `Excel`, `CSV`) with a web UI for:
- prompting for custom tests
- taking generated tests
- auto-grading with per-question feedback
- LLM rubric grading for short-answer meaning (not just keywords)

## Project structure
- `test_maker_rag.py`: RAG engine, indexing, test generation, grading, and CLI.
- `app.py`: Flask API + web server.
- `templates/index.html`: dynamic frontend.
- `Materials/`: source course content used for retrieval.

## 1) Local setup
```bash
python -m venv venv
# Windows PowerShell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Set your key (optional if you enter it in UI):
```bash
# Windows PowerShell
$env:OPENAI_API_KEY="sk-..."
```

Run the app:
```bash
python app.py
```
Open `http://localhost:5000`.

## 2) CLI usage
The script now prompts for your OpenAI key if not found in `OPENAI_API_KEY`.

```bash
python test_maker_rag.py --rebuild-index
```

Single question:
```bash
python test_maker_rag.py --query "What is PCA?"
```

Single test generation:
```bash
python test_maker_rag.py --test-request "Generate me a test on chapter 1 PCA concepts" --num-questions 10
```

## 3) Docker
Build:
```bash
docker build -t testmaker-rag .
```

Run:
```bash
docker run --rm -p 10000:10000 -e OPENAI_API_KEY="sk-..." testmaker-rag
```
Open `http://localhost:10000`.

## 4) Deploy to Render (Docker)
1. Push this project to GitHub (including `Materials/` unless you plan to mount/use remote storage).
2. In Render, create a new **Web Service** from your repo.
3. Render will detect `Dockerfile` (or use `render.yaml` if selected).
4. Add environment variables:
   - `OPENAI_API_KEY` (recommended for server-side key)
   - optional: `OPENAI_MODEL=gpt-4o-mini`
   - `ENABLE_RERANKER=0` (recommended on Render free tier)
   - `MAX_CACHED_TESTS=20`
5. Deploy.
6. Verify health check: `/api/health`.

## Notes
- First run builds the Chroma index from `Materials/` and may take a few minutes.
- Generated tests are cached in server memory for grading.
- Short-answer grading uses an LLM rubric; provide `OPENAI_API_KEY` on server or in the page input.
- If `Materials/` is not present in deployment, retrieval will fail.
- If requests time out on first test generation, increase Gunicorn timeout via `GUNICORN_TIMEOUT` and `GUNICORN_GRACEFUL_TIMEOUT` (default in Dockerfile is `300`).
