
import json
import logging
import os
import re
import threading
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from embed import (CHUNKS_FILE, EMBEDDINGS_FILE, SETTINGS, azure_client, describe_error,
                   missing_settings, text_hash)

QUESTION_CACHE_FILE = Path(__file__).resolve().parent / "question_embeddings.json"

RRF_K = 60        # the usual RRF constant: rank r in a ranking adds 1 / (RRF_K + r)

# Only each ranking's top CANDIDATES count. Deeper than that, a chunk that is
# only middling in both rankings collects two shares and beats a chunk that one
# ranking puts first: an English question never keyword-matches the Dutch
# OV-220, so its chunks only get the embedding share. On eval/questions.json,
# 10 finds an expected page in the top 6 for 14 of 17 answerable questions; 50
# finds 11 (keywords alone: 12). The torque, oil and part number checks pass
# with both.
CANDIDATES = 10
MAX_RETRIES = 3   # retries per Azure request (on rate limits, timeouts, server errors)
TIMEOUT = 10      # seconds before an Azure request is given up

log = logging.getLogger("search")


# ------------------------------------------------------------------ keywords

# A word is a run of letters and digits. Pieces joined by "-", "." or "," stay
# one word, so codes and numbers are kept whole: "HX-400", "400-0412",
# "HX25-SEAL-28S-GJL", "P-1201A", "0.45", "1,5". Anything else (spaces, "/",
# "|", brackets, a full stop at the end of a sentence) separates words.
TOKEN = re.compile(r"[^\W_]+(?:[-.,][^\W_]+)*")

# Filler words left out of both the chunks and the question, so they don't
# decide the ranking. English and Dutch, because OV-220 is in Dutch.
ENGLISH_STOP_WORDS = set("""
    a about above after again against all am an and any are as at be been before
    being below between both but by can could did do does doing down during each
    few for from further had has have having he her here hers herself him himself
    his how i if in into is it its itself just may me might more most must my
    myself no nor not now of off on once only or other our ours ourselves out over
    own same shall she should so some such than that the their theirs them
    themselves then there these they this those through to too under until up use
    used using very was we were what when where which while who whom why will with
    would you your yours yourself yourselves
""".split())
DUTCH_STOP_WORDS = set("""
    aan al alles als altijd andere ben bij daar dan dat de der deze die dit doch
    doen door dus een eens en er ge geen geweest haar had heb hebben heeft hem het
    hier hij hoe hun iemand iets ik in is ja je kan kon kunnen maar me meer men met
    mij mijn moet na naar niet niets nog nu of om omdat onder ons ook op over reeds
    te tegen toch toen tot u uit uw van veel voor want waren was wat werd wezen wie
    wil worden wordt zal ze zelf zich zij zijn zo zonder zou
""".split())
STOP_WORDS = ENGLISH_STOP_WORDS | DUTCH_STOP_WORDS


def stem(word):
    """Light stemming: strip plural endings, so "bolts" matches "bolt" and
    "oils" matches "oil". Numbers and codes (anything with a digit or a
    hyphen) are left exactly as they are."""
    if not word.isalpha() or len(word) <= 3:
        return word
    if word.endswith("sses"):
        return word[:-2]          # glasses -> glass
    if word.endswith("ies"):
        return word[:-3] + "y"    # assemblies -> assembly
    if word.endswith(("ss", "us", "is")):
        return word               # glass, status, analysis
    if word.endswith("s"):
        return word[:-1]          # bolts -> bolt
    return word


def tokenize(text):
    """Lowercase the text, split it into words, drop stop words and stem.
    Numbers and part codes are never dropped or changed."""
    words = []
    for word in TOKEN.findall(text.lower()):
        if word in STOP_WORDS:
            continue
        word = stem(word)
        if word not in STOP_WORDS:  # e.g. "uses" -> "use"
            words.append(word)
    return words


# ---------------------------------------------------------------- embeddings

def load_chunk_vectors():
    """The chunk embeddings from embeddings.npz as (vectors, has_vector,
    deployment). Row n of vectors belongs to CHUNKS[n] and has length 1, so a
    dot product is the cosine similarity. A chunk whose text changed after
    embed.py ran has no usable vector (has_vector False) and is found by
    keyword search only. Returns (None, None, None) if there is no file."""
    if not EMBEDDINGS_FILE.exists():
        log.warning("%s not found, so search uses keywords only. Run embed.py.", EMBEDDINGS_FILE.name)
        return None, None, None
    with np.load(EMBEDDINGS_FILE, allow_pickle=False) as data:
        saved = {str(i): (str(h), v) for i, h, v in zip(data["ids"], data["hashes"], data["vectors"])}
        deployment = str(data["deployment"])
        dimensions = data["vectors"].shape[1]

    vectors = np.zeros((len(CHUNKS), dimensions), dtype=np.float32)
    has_vector = np.zeros(len(CHUNKS), dtype=bool)
    for n, chunk in enumerate(CHUNKS):
        row = saved.get(chunk["id"])
        if row is not None and row[0] == text_hash(chunk["text"]):
            vectors[n] = row[1]
            has_vector[n] = True
    if not has_vector.all():
        log.warning("%d of %d chunks have no up-to-date embedding. Run embed.py.",
                    (~has_vector).sum(), len(CHUNKS))
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors /= np.where(norms == 0, 1, norms)
    return vectors, has_vector, deployment


def make_client(deployment):
    """An Azure OpenAI client for embedding questions, or None (with a warning)
    when the settings in .env are missing or name a different deployment than
    the one that embedded the chunks: vectors from two models can't be compared."""
    missing = missing_settings(SETTINGS)
    if missing:
        log.warning("Missing in .env: %s, so search uses keywords only.", ", ".join(missing))
        return None
    if os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"] != deployment:
        log.warning("The chunks were embedded with '%s' but .env names '%s', so search uses "
                    "keywords only. Run embed.py again.",
                    deployment, os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"])
        return None
    return azure_client(max_retries=MAX_RETRIES, timeout=TIMEOUT)


def load_question_cache(deployment):
    """{question: vector} for questions embedded before with this deployment."""
    try:
        data = json.loads(QUESTION_CACHE_FILE.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except ValueError:
        log.warning("%s is not valid JSON; starting a new cache.", QUESTION_CACHE_FILE.name)
        return {}
    if data.get("deployment") != deployment:
        return {}
    return data.get("questions", {})


def save_question_cache():
    """Write the cache under a temporary name, then rename it, so a crash
    never leaves a half-written file."""
    tmp = QUESTION_CACHE_FILE.with_name(QUESTION_CACHE_FILE.name + ".tmp")
    data = {"deployment": DEPLOYMENT, "questions": QUESTION_VECTORS}
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp.replace(QUESTION_CACHE_FILE)


def embed_question(question):
    """The question's unit-length embedding, from the cache or from Azure.
    Returns None if Azure fails; the caller then uses keywords only."""
    key = " ".join(question.split())  # extra spaces don't make a new question
    vector = QUESTION_VECTORS.get(key)
    if vector is None:
        # One request at a time fetches new questions, so two requests asking
        # the same new question at once still embed it only once.
        with CACHE_LOCK:
            vector = QUESTION_VECTORS.get(key)
            if vector is None:
                try:
                    response = CLIENT.embeddings.create(model=DEPLOYMENT, input=[key])
                except Exception as error:
                    log.warning("Embedding the question failed, so this search uses keywords only: %s",
                                describe_error(error))
                    return None
                vector = response.data[0].embedding
                QUESTION_VECTORS[key] = vector
                save_question_cache()
    vector = np.asarray(vector, dtype=np.float32)
    return vector / np.linalg.norm(vector)


# ------------------------------------------------- loaded once, at import time

with CHUNKS_FILE.open(encoding="utf-8") as f:
    CHUNKS = json.load(f)
INDEX = BM25Okapi([tokenize(chunk["text"]) for chunk in CHUNKS])

VECTORS, HAS_VECTOR, DEPLOYMENT = load_chunk_vectors()
CLIENT = make_client(DEPLOYMENT) if VECTORS is not None else None
QUESTION_VECTORS = load_question_cache(DEPLOYMENT) if CLIENT is not None else {}
CACHE_LOCK = threading.Lock()


# -------------------------------------------------------------------- search

def search(question, top_k=6):
    """Return the top_k chunks that best match the question, best first.

    Each result is a dict with document, page, page_end (the last page the
    chunk runs onto, usually the same), chunk_id, score and text, where
    score is the merged RRF score: the sum of 1 / (RRF_K + rank) over the two
    rankings (at most 2 / 61 = 0.0328, when both rank the chunk first). Two
    more fields show how well the chunk matches; app.py's "I don't know"
    threshold uses them: bm25_score (0 = no keyword match) and similarity
    (cosine similarity to the question, None without embeddings).
    Fewer than top_k results come back only when few chunks match at all.
    """
    if not question.strip():
        return []

    bm25 = INDEX.get_scores(tokenize(question))  # all zeros when no word is left
    keyword_ranking = [i for i in np.argsort(-bm25, kind="stable")[:CANDIDATES] if bm25[i] > 0]

    similarity = None
    meaning_ranking = []
    query = embed_question(question) if CLIENT is not None else None
    if query is not None:
        similarity = VECTORS @ query
        ranked = np.argsort(-np.where(HAS_VECTOR, similarity, -np.inf), kind="stable")
        meaning_ranking = [i for i in ranked[:CANDIDATES] if HAS_VECTOR[i]]

    fused = {}
    for ranking in (keyword_ranking, meaning_ranking):
        for rank, i in enumerate(ranking, start=1):
            fused[i] = fused.get(i, 0.0) + 1 / (RRF_K + rank)
    best = sorted(fused, key=fused.get, reverse=True)[:top_k]

    return [
        {
            "document": CHUNKS[i]["document"],
            "page": CHUNKS[i]["page"],
            "page_end": CHUNKS[i]["page_end"],
            "chunk_id": CHUNKS[i]["id"],
            "score": round(fused[i], 5),
            "text": CHUNKS[i]["text"],
            "bm25_score": round(float(bm25[i]), 3),
            "similarity": round(float(similarity[i]), 4) if similarity is not None and HAS_VECTOR[i] else None,
        }
        for i in best
    ]
