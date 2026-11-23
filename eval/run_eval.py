"""Score the handbook Q&A on every question in eval/questions.json.

    .venv/bin/python eval/run_eval.py

Each question goes through exactly the same pipeline as POST /ask: this script
imports the backend and runs the endpoint's two steps, search() and then
app.respond(), so the hybrid search, the "I don't know" threshold, the prompt
and the checks on the model's reply are the ones the web app uses, and the
recorded search results are exactly what the model saw. No server has to be
running.

Three checks per question (the rules are spelled out at each function):
  answer    are the question's key_values in the answer?        (answer_correct)
  citation  is the expected document and page among the cited sources?
  refusal   unanswerable questions must get "I don't know"

Reproducibility: the model runs at temperature 0 (set in app.py), and question
embeddings come from backend/question_embeddings.json, so a question is sent to
the embedding model only the very first time. Temperature 0 makes the model
pick the most likely words, but Azure doesn't promise byte-identical output,
so a rare answer can still change between runs.

Writes everything to eval/results.json and prints a summary per question type
plus every failure. Costs one chat call per question that passes the threshold.
"""

import json
import logging
import re
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_FILE = ROOT / "eval" / "questions.json"
RESULTS_FILE = ROOT / "eval" / "results.json"

sys.path.insert(0, str(ROOT / "backend"))
import app  # noqa: E402  (the backend: search index, embeddings and Azure client load here)
import search  # noqa: E402
from embed import text_hash  # noqa: E402

TYPES = ["fact", "catalogue", "dutch", "two-documents", "unanswerable"]


# --------------------------------------------------------- answer matching

# A number, as written in an answer: "380", "1284.00", "1,284.00", "19,5".
# Digits that belong to a code are skipped: the 40 in "HX40", the 28 in
# "SEAL-28S" and the 20 in "M20" follow a letter or a letter and a hyphen.
NUMBER = re.compile(r"(?<![A-Za-z])(?<![A-Za-z]-)\d+(?:[.,]\d+)*")

# A key value that is a number, with an optional currency sign before it and
# an optional unit after it: "380 Nm", "€82.98", "28%", "2.8 mm/s", "1125 mm".
NUMBER_KEY = re.compile(r"^[€$£]?\s*(\d[\d.,]*)\s*[A-Za-z%°/³²]*$")


def to_number(text):
    """'1,284.00' -> 1284.0, '4,000' -> 4000.0, '19,5' -> 19.5, '380' -> 380.0.
    A comma followed by exactly three digits separates thousands; any other
    comma is a decimal comma."""
    if "," in text and "." in text:
        text = text.replace(",", "")
    elif re.fullmatch(r"\d{1,3}(,\d{3})+", text):
        text = text.replace(",", "")
    else:
        text = text.replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None


def flat(text):
    """Lowercase, hyphens as spaces, single spaces: 'Air-Fed  hood' -> 'air fed hood'."""
    return " ".join(text.lower().replace("-", " ").split())


def key_value_found(key, answer):
    """Is this key value in the answer? The matching rules:

    1. A key value can list alternatives separated by "|"
       ("Criticality A|class A"); any one of them counts.
    2. A number, with or without a unit or currency ("380 Nm", "€1284.00",
       "28%"), counts if the same number appears in the answer, however it is
       written: 1284.00, 1,284.00 and 1284 are all equal. The unit is not
       checked, so "€1284.00" also matches "1284.00 EUR".
    3. Anything else (a part number, a name, a word) counts if it appears in
       the answer, ignoring case, starting at the beginning of a word, with
       hyphens counted as spaces: "seal" matches "Seals" but not "unsealed",
       and "air-fed" matches "air fed".
    """
    for option in key.split("|"):
        option = option.strip()
        number = NUMBER_KEY.match(option)
        if number:
            wanted = to_number(number.group(1))
            if wanted is not None and any(to_number(n) == wanted for n in NUMBER.findall(answer)):
                return True
        elif re.search(r"(?<![0-9a-z])" + re.escape(flat(option)), flat(answer)):
            return True
    return False


# -------------------------------------------------------- citation matching

def covers(sources, document, pages):
    """True if one of the sources is this document and spans one of these pages
    (a chunk on p.11-12 covers both page 11 and page 12)."""
    return any(
        source["document"] == document and any(source["page"] <= page <= source["page_end"] for page in pages)
        for source in sources
    )


def missing_documents(expected, sources):
    """The expected sources that none of the given sources covers. The rule:
    every expected document must be there, with at least one of the pages
    listed for it. So a two-documents question needs both documents, while for
    "pages": [11, 12] either page is enough."""
    return [entry for entry in expected if not covers(sources, entry["document"], entry["pages"])]


# ------------------------------------------------------------------ running

class Collect(logging.Handler):
    """Keeps the backend's warnings for each question (for example "The
    question names ['HX-500'], which no chunk mentions"), so they end up in
    results.json next to the question instead of in the terminal."""

    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def check_key_values(questions):
    """Each key value must come from the expected answer, so the test file
    can't quietly ask for something else. Returns a list of problems."""
    problems = []
    for number, q in enumerate(questions, start=1):
        if q["source"] and not q["key_values"]:
            problems.append(f"#{number}: answerable, but no key_values")
        for key in q["key_values"]:
            first = key.split("|")[0].strip()
            if first.lower() not in q["expected_answer"].lower():
                problems.append(f"#{number}: key value '{first}' is not in its expected_answer")
    return problems


def run_one(number, q, collect):
    """Ask one question the way POST /ask does, and score it."""
    collect.messages.clear()
    question = app.AskRequest(question=q["question"]).question  # validated and stripped like /ask
    answerable = bool(q["source"])

    retrieved = search.search(question, top_k=app.TOP_K)
    error = None
    try:
        response = app.respond(question, retrieved)
        answer, answered = response.answer, response.answered
        cited = [source.model_dump(include={"document", "page", "page_end"}) for source in response.sources]
    except app.AnswerError as exc:  # Azure failed: /ask would send this message as HTTP 502
        error, answer, answered, cited = str(exc), "", False, []

    refused = not answered and answer == app.NO_ANSWER
    if answerable:
        missing_values = [key for key in q["key_values"] if not key_value_found(key, answer)]
        answer_correct = answered and not missing_values
        missing_sources = missing_documents(q["source"], cited)
        citation_correct = answered and not missing_sources
        retrieval_missing = missing_documents(q["source"], retrieved)
    else:
        missing_values, missing_sources, retrieval_missing = [], [], []
        answer_correct = refused
        citation_correct = not cited

    return {
        "id": number,
        "question": question,
        "type": q["type"],
        "verified": q["verified"],
        "expected_answer": q["expected_answer"],
        "key_values": q["key_values"],
        "expected_sources": q["source"],
        "answer": answer,
        "answered": answered,
        "error": error,
        "cited": cited,
        "retrieved": [
            {key: chunk[key] for key in ("document", "page", "page_end", "similarity", "bm25_score")}
            for chunk in retrieved
        ],
        "reached_model": app.covered(retrieved),
        "backend_warnings": list(collect.messages),
        "checks": {
            "answer_correct": answer_correct,
            "missing_key_values": missing_values,
            "citation_correct": citation_correct,
            "missing_sources": missing_sources,
            # Did search return a page of every expected document? If not, the
            # model never had a chance (None for unanswerable).
            "expected_pages_retrieved": not retrieval_missing if answerable else None,
            "refused": refused,
            "refusal_correct": refused if not answerable else None,
        },
    }


def summarise(results):
    def count(rows, check):
        return {"correct": sum(bool(r["checks"][check]) for r in rows), "total": len(rows)}

    by_type = {}
    for kind in TYPES:
        rows = [r for r in results if r["type"] == kind]
        if rows:
            by_type[kind] = {"answers": count(rows, "answer_correct"), "citations": count(rows, "citation_correct")}

    unanswerable = [r for r in results if r["type"] == "unanswerable"]
    answerable = [r for r in results if r["type"] != "unanswerable"]
    return {
        "answers": count(results, "answer_correct"),
        "citations": count(results, "citation_correct"),
        "refusals": count(unanswerable, "refusal_correct"),
        "answerable_refused": sum(r["checks"]["refused"] for r in answerable),
        "errors": sum(r["error"] is not None for r in results),
        "by_type": by_type,
    }


# ------------------------------------------------------------------ output

def where(sources):
    """'HX-400 IOM Manual p.11-12, OV-220 Preventief Onderhoud NL p.33'"""
    parts = []
    for s in sources:
        if "pages" in s:  # an expected source: {"document", "pages": [...]}
            label = "p. " + "/".join(str(p) for p in s["pages"])
        else:
            label = app.page_label(s)
        parts.append(f"{app.document_title(s['document'])} {label}")
    return ", ".join(parts) or "none"


def wrap(label, text):
    return textwrap.fill(text, width=110, initial_indent=f"   {label:<10}", subsequent_indent=" " * 13)


def print_report(summary, results, new_embeddings):
    s = summary
    print(f"\nAnswers: {s['answers']['correct']}/{s['answers']['total']} correct · "
          f"Citations: {s['citations']['correct']}/{s['citations']['total']} · "
          f"Refusals: {s['refusals']['correct']}/{s['refusals']['total']}")
    print(f"Answerable questions answered with \"I don't know\": {s['answerable_refused']} · "
          f"Azure errors: {s['errors']} · New question embeddings: {new_embeddings}")

    print(f"\n{'type':<15}{'answers':>9}{'citations':>11}")
    for kind, row in s["by_type"].items():
        print(f"{kind:<15}{row['answers']['correct']:>5}/{row['answers']['total']:<3}"
              f"{row['citations']['correct']:>7}/{row['citations']['total']}")

    failures = [r for r in results if not (r["checks"]["answer_correct"] and r["checks"]["citation_correct"])]
    print(f"\nFAILURES ({len(failures)})")
    for r in failures:
        c = r["checks"]
        failed = []
        if r["error"]:
            failed.append(f"Azure error: {r['error']}")
        if r["type"] == "unanswerable":
            if not c["refused"]:
                failed.append("should have said \"I don't know\"")
        elif c["refused"]:
            failed.append("said \"I don't know\"")
        else:
            if not c["answer_correct"]:
                failed.append("answer is missing " + ", ".join(f"'{k}'" for k in c["missing_key_values"]))
            if not c["citation_correct"]:
                failed.append("citations are missing " + where(c["missing_sources"]))
        print(f"\n#{r['id']} [{r['type']}] {r['question']}")
        print(wrap("failed:", "; ".join(failed)))
        if r["type"] != "unanswerable":
            print(wrap("search:", ("found the expected pages" if c["expected_pages_retrieved"]
                                   else "did NOT find the expected pages")
                       + f" (top {app.TOP_K}: {where(r['retrieved'])})"))
        print(wrap("expected:", r["expected_answer"] + f"  [{where(r['expected_sources'])}]"))
        print(wrap("actual:", (r["answer"] or "(no answer)") + f"  [cited: {where(r['cited'])}]"))
        for message in r["backend_warnings"]:
            print(wrap("backend:", message))


def main():
    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    problems = check_key_values(questions)
    if problems:
        sys.exit("Fix eval/questions.json first:\n  " + "\n  ".join(problems))

    collect = Collect()
    for name in ("app", "search"):
        logging.getLogger(name).addHandler(collect)

    cached_before = len(search.QUESTION_VECTORS)
    progress = sys.stdout.isatty()  # a progress line only in a terminal, not in a log file
    results = []
    for number, q in enumerate(questions, start=1):
        if progress:
            print(f"\r{number}/{len(questions)} {q['question'][:70]:<70}", end="", flush=True)
        results.append(run_one(number, q, collect))
    if progress:
        print("\r" + " " * 80 + "\r", end="")
    new_embeddings = len(search.QUESTION_VECTORS) - cached_before

    summary = summarise(results)
    run = {
        "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "chat_deployment": app.CHAT_DEPLOYMENT,
        "embedding_deployment": search.DEPLOYMENT,
        "temperature": app.TEMPERATURE,
        "top_k": app.TOP_K,
        "min_similarity": app.MIN_SIMILARITY,
        # Changes whenever the prompt or the answer format changes, so results
        # from different prompt versions are easy to tell apart.
        "prompt_sha256": text_hash(app.SYSTEM_PROMPT + json.dumps(app.ANSWER_FORMAT))[:12],
        "new_question_embeddings": new_embeddings,
    }
    RESULTS_FILE.write_text(
        json.dumps({"run": run, "summary": summary, "results": results}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print_report(summary, results, new_embeddings)
    print(f"\nEverything per question: {RESULTS_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
