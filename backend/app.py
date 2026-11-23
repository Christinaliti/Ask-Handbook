"""HTTP API for the handbook.

    POST /ask     {"question": "..."} -> {"answer", "sources", "answered"}
    GET  /health  {"status": "ok"}

Start it from the project root:

    .venv/bin/uvicorn app:app --app-dir backend --reload

/ask runs the hybrid search (keywords plus embeddings, see search.py), then
asks the Azure OpenAI chat model (AZURE_OPENAI_DEPLOYMENT in .env) to answer
from the best chunks only, citing them. "sources" holds just the chunks the
answer cites. When the documents don't cover the question, the answer is
NO_ANSWER and answered is false. When Azure OpenAI fails, /ask replies with
HTTP 502 and a message in "detail" that says what went wrong.

The API key is never printed or logged.
"""

import json
import logging
import os
import re

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from embed import AZURE_SETTINGS, azure_client, describe_error, missing_settings

# Importing search builds the BM25 index from chunks.json and loads the chunk
# embeddings. That happens once, when the server starts; every request reuses
# it. After re-running Ingest.py or embed.py, restart the server.
from search import search

log = logging.getLogger("app")

TOP_K = 6  # chunks the model gets to answer from

# Below this, the question is answered with NO_ANSWER without asking the model.
# It is the highest cosine similarity (0 to 1) between the question and the
# chunks search returned; the merged search score can't be used, as it only
# says how a chunk ranks, not how well it matches. The model already refuses
# questions the chunks don't answer, so this only has to stop clearly off-topic
# questions, and it is set low so it never stops a real one. On
# eval/questions.json every answerable question reaches 0.51 or more. Questions
# that have nothing to do with pumps reach 0.09 to 0.21 ("What is the capital
# of France?" 0.09, "How do I reset my password?" 0.21), so they get NO_ANSWER
# without a model call. "What is the warranty?" (0.33), "What is the warranty
# period?" (0.37) and Dutch small talk, which matches the Dutch document's
# language ("Hoe laat is het?" 0.39), go to the model, which says it doesn't
# know.
MIN_SIMILARITY = 0.30

# Used instead when search runs on keywords only (embeddings missing or Azure
# unreachable): the best BM25 score must reach this. On the eval file the three
# checked questions score 6.5 to 16.0 and "What is the warranty?" 5.5.
MIN_KEYWORD_SCORE = 6.0

ALLOWED_ORIGINS = ["http://localhost:5173"]  # the React dev server

NO_ANSWER = "I don't know, the documents don't cover this."

CHAT_SETTINGS = AZURE_SETTINGS + ["AZURE_OPENAI_DEPLOYMENT"]
CHAT_MAX_RETRIES = 3     # retries per request (on rate limits, timeouts, server errors)
CHAT_TIMEOUT = 60        # seconds before a request is given up
MAX_ANSWER_TOKENS = 1000  # longest reply the model may write
TEMPERATURE = 0          # the same question and chunks give the same answer

SYSTEM_PROMPT = f"""You answer maintenance questions about HX pumps for engineers and technicians, using only the numbered excerpts from the maintenance documents in the user message.

Rules:
- Answer ONLY from the excerpts. Never guess, and never use outside knowledge, even when you think you know the answer.
- Cite every claim: end each sentence or list item that states a fact with its source in square brackets as [document, page], with the document name and page exactly as in the label of the excerpt that sentence comes from (never the excerpt number). Facts from different excerpts get different citations, so one citation at the end of the answer is not enough. For example: "Isolate the pump before opening the housing [WP-114 Isolation and Confined Space, p. 7]. Wear nitrile gloves when draining the oil [RIS-VGM-0042 Safety Register, p. 4]."
- Always answer in English, even when an excerpt is in Dutch: translate what you use.
- Give exact values and units as the excerpts state them. When a value depends on a condition (size, model, lubricated or dry), give each case.
- Be concise. Don't add advice or remarks the excerpts don't give.
- If the excerpts answer only part of the question, answer that part and say which part the documents don't cover.
- If the excerpts don't contain the answer, set "answer" to exactly "{NO_ANSWER}", "answered" to false and "cited_chunks" to [].

Reply in JSON: {{"answer": "...", "answered": true or false, "cited_chunks": [the numbers of the excerpts you cited], "named_products": [...]}}"""

# Structured output: Azure makes the model reply with exactly this JSON shape,
# writing the fields in this order. named_products comes last: listed first, it
# made the model refuse questions it otherwise answers. It is only used here
# (see unknown_products); /ask doesn't return it.
ANSWER_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "answer",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "The answer in English. Every sentence or list item that states a fact ends "
                                   "with its citation, e.g. [HX-400 IOM Manual, p. 11].",
                },
                "answered": {"type": "boolean"},
                "cited_chunks": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "The numbers of the excerpts the answer cites.",
                },
                "named_products": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Every brand, model number and part number the question names, copied exactly "
                                   "as the question writes it, e.g. 'HX-400', 'KSB', 'HX40-SEAL-27S'. Not generic "
                                   "words such as 'pump' or 'oil'. Empty when the question names none.",
                },
            },
            "required": ["answer", "answered", "cited_chunks", "named_products"],
            "additionalProperties": False,
        },
    },
}

app = FastAPI(title="Ask Handbook")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    question: str = Field(min_length=1)


class Source(BaseModel):
    document: str
    page: int
    page_end: int  # same as page unless the chunk runs onto the next page
    chunk_id: str
    score: float
    text: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]  # the chunks the answer cites, best first; empty when not answered
    answered: bool


class AnswerError(Exception):
    """Azure OpenAI could not give an answer. The message is safe to show to
    the user: it says what went wrong and never contains the key."""


# ------------------------------------------------------------- Azure OpenAI

def make_chat_client():
    """The Azure OpenAI client for answers, or None (with a warning) when
    settings are missing from .env; /ask then replies with a clear error."""
    missing = missing_settings(CHAT_SETTINGS)
    if missing:
        log.warning("Missing in .env: %s, so /ask can't answer questions.", ", ".join(missing))
        return None
    return azure_client(max_retries=CHAT_MAX_RETRIES, timeout=CHAT_TIMEOUT)


CHAT_CLIENT = make_chat_client()  # created once, when the server starts
CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")


def document_title(file_name):
    """'01_HX-400_IOM_Manual.pdf' -> 'HX-400 IOM Manual'"""
    return re.sub(r"^\d+_", "", file_name).removesuffix(".pdf").replace("_", " ")


def page_label(chunk):
    """'p. 11', or 'p. 11-12' for a chunk that runs onto the next page."""
    if chunk["page_end"] == chunk["page"]:
        return f"p. {chunk['page']}"
    return f"p. {chunk['page']}-{chunk['page_end']}"


def build_prompt(question, chunks):
    """The question and the chunks, each labelled with its number, document
    and page, as the model sees them."""
    lines = [f"Question: {question}"]
    for number, chunk in enumerate(chunks, start=1):
        # The number is not in square brackets, so it isn't mistaken for a citation.
        lines += ["", f"Excerpt {number}: {document_title(chunk['document'])}, {page_label(chunk)}", chunk["text"]]
    return "\n".join(lines)


def mentions(text, word):
    """True if text contains word as a whole word, ignoring case and any
    punctuation or spaces inside it: 'HX-400/S' matches 'HX-400 /S', but
    'KSB' doesn't match 'checks before'."""
    letters = re.sub(r"[^0-9a-z]", "", word.lower())
    if not letters:
        return True
    pattern = r"(?<![0-9a-z])" + r"[\W_]*".join(letters) + r"(?![0-9a-z])"
    return re.search(pattern, text.lower()) is not None


def unknown_products(named, chunks):
    """The products named in the question (as the model listed them) that
    appear in none of the chunks. Each word of a name must appear somewhere,
    so 'HX pump' passes when the chunks say 'HX-400' and 'pump', but 'HX-500'
    and 'KSB' fail when the chunks only cover the HX-400."""
    text = "\n".join(chunk["text"] for chunk in chunks)
    return [name for name in named if not all(mentions(text, word) for word in name.split())]


def valid_citations(cited, count):
    """The cited chunk numbers that really exist (1 to count), each once, in
    search order. Any other number the model sent is dropped, with a warning.
    (The JSON schema makes them integers; it can't make them exist.)"""
    dropped = [n for n in cited if not 1 <= n <= count]
    if dropped:
        log.warning("Dropped cited chunk numbers that don't exist: %s", dropped)
    return sorted({n for n in cited if 1 <= n <= count})


def ask_model(question, chunks):
    """Send the question and chunks to Azure OpenAI and return its reply as a
    dict in the ANSWER_FORMAT shape, or None if the model refuses to answer.
    Raises AnswerError with a clear message when the request fails."""
    import openai  # loaded by make_chat_client; needed here for its error types

    if CHAT_CLIENT is None:
        raise AnswerError("AI answers are not set up: settings are missing in backend/.env (see the server log).")

    def fail(message, error=None):
        if error is not None:
            log.warning("Azure OpenAI request failed: %s", describe_error(error))
        raise AnswerError(message)

    try:
        response = CHAT_CLIENT.chat.completions.create(
            model=CHAT_DEPLOYMENT,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(question, chunks)},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_ANSWER_TOKENS,
            response_format=ANSWER_FORMAT,
        )
    except openai.AuthenticationError as error:
        fail("Azure OpenAI rejected the API key. Check AZURE_OPENAI_API_KEY in backend/.env.", error)
    except openai.NotFoundError as error:
        fail(f"Azure OpenAI has no deployment named '{CHAT_DEPLOYMENT}'. "
             "Check AZURE_OPENAI_DEPLOYMENT in backend/.env.", error)
    except openai.RateLimitError as error:
        fail("Azure OpenAI is busy (rate limit reached). Wait a minute and try again.", error)
    except openai.APIConnectionError as error:  # includes timeouts
        fail("Can't reach Azure OpenAI. Check the internet connection and try again.", error)
    except openai.BadRequestError as error:
        if error.code == "content_filter":
            fail("Azure OpenAI's content filter blocked this question. Try rephrasing it.", error)
        fail("Azure OpenAI could not process this question (see the server log).", error)
    except openai.APIError as error:
        fail("Azure OpenAI returned an error. Try again in a moment.", error)

    choice = response.choices[0]
    if choice.finish_reason == "length":
        fail("The answer was too long and got cut off. Try a narrower question.")
    if choice.finish_reason == "content_filter":
        fail("Azure OpenAI's content filter blocked the answer. Try rephrasing the question.")
    if choice.message.refusal:
        return None
    try:
        return json.loads(choice.message.content or "")
    except ValueError:
        fail("The AI model sent a reply that could not be read. Try again.")


def generate_answer(question, chunks):
    """Answer the question from the chunks (search results, best first).

    Returns (answer text, cited chunks), or None when the chunks don't answer
    the question; /ask then replies with NO_ANSWER and answered: false.
    Raises AnswerError when Azure OpenAI fails.
    """
    reply = ask_model(question, chunks)
    if reply is None or not reply["answered"] or reply["answer"].strip() in ("", NO_ANSWER):
        return None
    unknown = unknown_products(reply["named_products"], chunks)
    if unknown:
        # The model tends to answer "How much oil does the HX-500 take?" with
        # the HX-400's figure. If a product the question names isn't in any
        # chunk, the chunks can't be about it.
        log.warning("The question names %s, which no chunk mentions; replying NO_ANSWER.", unknown)
        return None
    cited = [chunks[n - 1] for n in valid_citations(reply["cited_chunks"], len(chunks))]
    if not cited:
        # Every claim must cite a chunk, so an answer that cites none isn't trusted.
        log.warning("The model answered without citing any existing chunk; replying NO_ANSWER instead.")
        return None
    return reply["answer"].strip(), cited


def covered(chunks):
    """True if the best search result is close enough to the question for the
    documents to plausibly cover it (see MIN_SIMILARITY)."""
    similarities = [chunk["similarity"] for chunk in chunks if chunk["similarity"] is not None]
    if similarities:
        return max(similarities) >= MIN_SIMILARITY
    return max((chunk["bm25_score"] for chunk in chunks), default=0.0) >= MIN_KEYWORD_SCORE


@app.get("/health")
def health():
    return {"status": "ok"}


def respond(question, chunks):
    """The /ask reply for a question and its search results. The model is only
    asked when the results are close enough (covered). Raises AnswerError when
    Azure OpenAI fails. eval/run_eval.py calls this too."""
    if covered(chunks):
        result = generate_answer(question, chunks)
        if result is not None:
            answer, cited = result
            return AskResponse(answer=answer, sources=cited, answered=True)
    return AskResponse(answer=NO_ANSWER, sources=[], answered=False)


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    chunks = search(request.question, top_k=TOP_K)
    try:
        return respond(request.question, chunks)
    except AnswerError as error:
        raise HTTPException(status_code=502, detail=str(error))
