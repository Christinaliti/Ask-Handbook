# Ask the Handbook

Ask questions about the HX-400 pump maintenance documents and get an answer with the pages it comes from. The five PDFs in `Data/` are split into chunks and searched by keyword and by meaning. Azure OpenAI then answers from the best chunks only, or says "I don't know" when they don't cover the question.

## What you need

- Python 3.12
- Node.js 20.19 or newer (for Vite)
- Azure OpenAI resource 

## Setup (once)

From the project root:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cd frontend && npm install && cd ..
```

Create `backend/.env` with your Azure settings:

```
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_API_KEY=<your key>
AZURE_OPENAI_API_VERSION=2024-10-21
AZURE_OPENAI_DEPLOYMENT=<your chat deployment, e.g. gpt-4o-mini>
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=<your embedding deployment, e.g. text-embedding-3-small>
```


## Build the search data

Do this the first time, and again whenever the PDFs in `Data/` change:

```bash
.venv/bin/python Ingest.py          # PDFs in Data/ -> backend/chunks.json
.venv/bin/python backend/embed.py   # chunks -> backend/embeddings.npz (calls Azure)
```

`embed.py` only sends chunks that are new or changed. Run it again and it says "0 chunks to embed" without calling Azure.

## Run the app

Use two terminals, both started in the project root.

Terminal 1, the backend on http://localhost:8000:

```bash
.venv/bin/uvicorn app:app --app-dir backend --reload
```

Terminal 2, the frontend on http://localhost:5173:

```bash
cd frontend && npm run dev
```

Open http://localhost:5173 and ask a question. To check that the backend is up on its own, open http://localhost:8000/health; it should show `{"status":"ok"}`.

After running `Ingest.py` or `embed.py` again, restart the backend. `--reload` only reacts to changes in `.py` files, not to new data.

To try the search on its own, without the AI answer:

```bash
.venv/bin/python backend/try_search.py "Which oil grade should I use?"
```

## If something goes wrong

- **"Can't reach the server. Is the backend running?"**: terminal 1 isn't running, or not on port 8000. The frontend calls the URL in `frontend/src/config.ts`.
- **The frontend opened on a port other than 5173** (because 5173 was busy): the backend only accepts requests from http://localhost:5173 (`ALLOWED_ORIGINS` in `backend/app.py`). Stop whatever is using 5173 and start the frontend again.
- **A red message that names an `AZURE_OPENAI_...` setting**: fix that setting in `backend/.env`, then restart the backend.
- **"address already in use"**: the server is already running in another terminal. Use that one or stop it first.
- **The backend warns about `embeddings.npz` or "no up-to-date embedding" at startup**: run `backend/embed.py`. Until then, search uses keywords only.

## Evaluation

Run every test question in `eval/questions.json` through the same pipeline as the web app and score it. From the project root:

```bash
.venv/bin/python eval/run_eval.py
```

No server needs to be running: the script imports the backend and runs the same two steps as `POST /ask`. It needs `backend/.env` and `backend/embeddings.npz`. A run takes about a minute and makes one chat call per question.

It prints a summary per question type and every failure, for example:

```
Answers: 17/23 correct · Citations: 17/23 · Refusals: 6/6
```

It also saves everything per question (answer, cited and retrieved pages, backend warnings) to `eval/results.json`.

Three checks per question:

- **Answer**: each value in the question's `key_values` must be in the answer. Numbers are compared as numbers (`€1284.00` matches `1,284.00 EUR`), anything else as text, ignoring case. `a|b` means either one counts.
- **Citation**: every expected document must be cited, with at least one of the pages listed for it.
- **Refusal**: unanswerable questions must get exactly "I don't know, the documents don't cover this."

**Results vary a little between runs.** Search results are identical every run, because question embeddings come from the cache in `backend/question_embeddings.json`. The model's answers are not, even at temperature 0: Azure doesn't guarantee the same output for the same input. With identical input, "What is the minimum speed I can set on a variable speed drive for the HX-400?" was refused in 3 of 5 calls and answered in the other 2. Expect one or two questions to change between runs, and compare a few runs before trusting a small change in the score.

To add a test question, add an entry to `eval/questions.json` with `key_values`: the few facts a correct answer must contain, copied from `expected_answer`. The script stops if a key value isn't in its `expected_answer`.

Why each question failed in the analysed run: [eval/failure_analysis.md](eval/failure_analysis.md).
