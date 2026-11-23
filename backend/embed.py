"""Create an embedding (a list of numbers that captures the meaning of a text)
for every chunk in chunks.json, using Azure OpenAI.

    python embed.py

The Azure settings come from .env in this folder: AZURE_OPENAI_ENDPOINT,
AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_VERSION and
AZURE_OPENAI_EMBEDDING_DEPLOYMENT (the name of your embedding deployment).

The vectors are saved in embeddings.npz next to chunks.json, in chunks.json
order, together with each chunk's id and a hash of its text.

Running it again is cheap. A chunk is sent to Azure only when it is new, when
its text changed since the last run, or when the embedding deployment changed.
When nothing needs embedding, the Azure client is never even created, so no
API calls are made. Embeddings of chunks no longer in chunks.json are dropped.

The API key is never printed, logged or saved.
"""

import hashlib
import json
import math
import os
import sys
import traceback
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
ENV_FILE = HERE / ".env"
CHUNKS_FILE = HERE / "chunks.json"
EMBEDDINGS_FILE = HERE / "embeddings.npz"

BATCH_SIZE = 64   # chunks sent in one API request
MAX_RETRIES = 5   # the SDK waits and retries on rate limits (429) and server errors

# The Azure settings every client needs; each user adds its deployment.
AZURE_SETTINGS = ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_API_VERSION"]
SETTINGS = AZURE_SETTINGS + ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def redact(message):
    """Blank out the API key in a message before it is printed."""
    key = os.getenv("AZURE_OPENAI_API_KEY")
    return message.replace(key, "***") if key else message


def describe_error(error):
    """'AuthenticationError: ...' with the API key blanked out, for the log."""
    return redact(f"{type(error).__name__}: {error}")


def missing_settings(names):
    """Load .env and return the names in the list that have no value."""
    load_dotenv(ENV_FILE)
    return [name for name in names if not os.getenv(name)]


def azure_client(**options):
    """An Azure OpenAI client built from the settings in .env. Options such as
    max_retries and timeout are passed on to the client.

    openai is imported here, not at the top, so scripts that don't need a
    client never load it."""
    os.environ.pop("OPENAI_LOG", None)  # keep the SDK's debug log (request details) off
    import openai
    return openai.AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        **options,
    )


def load_saved(deployment):
    """The embeddings from the last run, as {chunk id: (text hash, vector)}.

    Empty when there is no file yet, or when it was made with another
    deployment: vectors from different models can't be compared."""
    if not EMBEDDINGS_FILE.exists():
        return {}
    with np.load(EMBEDDINGS_FILE, allow_pickle=False) as data:
        saved_deployment = str(data["deployment"])
        if saved_deployment != deployment:
            print(f"{EMBEDDINGS_FILE.name} was made with deployment '{saved_deployment}', "
                  f"not '{deployment}', so every chunk is embedded again.")
            return {}
        return {
            str(chunk_id): (str(digest), vector)
            for chunk_id, digest, vector in zip(data["ids"], data["hashes"], data["vectors"])
        }


def save(chunks, embedded, deployment):
    """Write the embeddings done so far, in chunks.json order.

    The file is written under a temporary name and then renamed, so a run that
    is interrupted never leaves a half-written embeddings.npz behind."""
    ids = [chunk["id"] for chunk in chunks if chunk["id"] in embedded]
    tmp = EMBEDDINGS_FILE.with_name("embeddings.tmp.npz")
    np.savez(
        tmp,
        ids=np.array(ids, dtype=str),
        hashes=np.array([embedded[i][0] for i in ids], dtype=str),
        vectors=np.array([embedded[i][1] for i in ids], dtype=np.float32),
        deployment=np.array(deployment),
    )
    tmp.replace(EMBEDDINGS_FILE)


def embed(todo, chunks, embedded, deployment):
    """Send the chunks in todo to Azure in batches and add the vectors to
    embedded. Saves after every batch, so if a run stops halfway, the next run
    carries on where it stopped."""
    client = azure_client(max_retries=MAX_RETRIES)
    import openai  # already loaded by azure_client(); needed here for its error types

    def fail(message):
        sys.exit(f"{message}\n{len(embedded)} of {len(chunks)} chunks are embedded and saved; "
                 "run embed.py again to continue.")

    try:
        for start in range(0, len(todo), BATCH_SIZE):
            batch = todo[start:start + BATCH_SIZE]
            response = client.embeddings.create(model=deployment, input=[chunk["text"] for chunk, _ in batch])
            items = sorted(response.data, key=lambda item: item.index)
            if len(items) != len(batch):
                fail(f"Azure returned {len(items)} vectors for {len(batch)} chunks.")
            for (chunk, digest), item in zip(batch, items):
                embedded[chunk["id"]] = (digest, item.embedding)
            save(chunks, embedded, deployment)
            print(f"  {start + len(batch)}/{len(todo)} embedded")
    except openai.AuthenticationError:
        fail("Azure rejected the request (401). Check that AZURE_OPENAI_API_KEY belongs "
             "to the resource in AZURE_OPENAI_ENDPOINT.")
    except openai.NotFoundError:
        fail(f"Azure answered 404: no deployment '{deployment}' was found. Check "
             "AZURE_OPENAI_EMBEDDING_DEPLOYMENT, AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_VERSION.")
    except openai.APIError as error:
        fail(f"Azure request failed: {redact(str(error))}")
    finally:
        client.close()


def main():
    missing = missing_settings(SETTINGS)
    if missing:
        sys.exit(f"Missing in {ENV_FILE}: {', '.join(missing)}")
    deployment = os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]

    with CHUNKS_FILE.open(encoding="utf-8") as f:
        chunks = json.load(f)
    saved = load_saved(deployment)

    embedded = {}  # chunk id -> (text hash, vector) for chunks that are up to date
    todo = []      # (chunk, text hash) for chunks still to embed
    for chunk in chunks:
        digest = text_hash(chunk["text"])
        previous = saved.get(chunk["id"])
        if previous is not None and previous[0] == digest:
            embedded[chunk["id"]] = previous
        else:
            todo.append((chunk, digest))
    stale = len(saved.keys() - {chunk["id"] for chunk in chunks})

    print(f"{len(chunks)} chunks in {CHUNKS_FILE.name}: {len(embedded)} already embedded, "
          f"{len(todo)} chunks to embed.")
    if not todo:
        if stale:
            save(chunks, embedded, deployment)
            print(f"Dropped {stale} embeddings of chunks that are no longer in {CHUNKS_FILE.name}.")
        print("Nothing to embed, so no API calls were made.")
        return

    print(f"Embedding with deployment '{deployment}', {BATCH_SIZE} chunks per request.")
    embed(todo, chunks, embedded, deployment)  # exits on any failure, so all batches went through
    dimensions = len(next(iter(embedded.values()))[1])
    print(f"Done: {len(todo)} chunks embedded in {math.ceil(len(todo) / BATCH_SIZE)} API requests. "
          f"Saved {len(embedded)} vectors of {dimensions} numbers to {EMBEDDINGS_FILE.name}.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Show the full error, but with the API key blanked out in case it appears.
        sys.exit(redact(traceback.format_exc()))
