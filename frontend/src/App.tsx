import { useState, type FormEvent } from 'react'
import { ASK_URL, DOCUMENT_NAMES, EXAMPLE_QUESTIONS } from './config'
import { AlertIcon, BookIcon, ChevronIcon, InfoIcon, SearchIcon } from './icons'

// The JSON the backend's POST /ask returns.
type Source = {
  document: string
  page: number
  page_end: number // same as page unless the chunk runs onto the next page
  chunk_id: string
  text: string
}

type AskResponse = {
  answer: string
  sources: Source[]
  answered: boolean
}

type Status =
  | { kind: 'idle' }
  | { kind: 'searching' }
  | { kind: 'error'; message: string }
  | { kind: 'done'; result: AskResponse }

async function askBackend(question: string): Promise<AskResponse> {
  let response: Response
  try {
    response = await fetch(ASK_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    })
  } catch {
    // fetch only throws when no reply arrives: server down, wrong URL, or CORS.
    throw new Error("Can't reach the server. Is the backend running?")
  }
  if (!response.ok) {
    // The backend explains what went wrong in "detail", e.g. when Azure OpenAI fails.
    const detail = await response.json().then((body) => body?.detail, () => undefined)
    throw new Error(typeof detail === 'string' ? detail : `The server replied with an error (HTTP ${response.status}).`)
  }
  try {
    return await response.json()
  } catch {
    throw new Error('The server sent a reply that is not JSON.')
  }
}

// "01_HX-400_IOM_Manual.pdf" -> "HX-400 Manual" (see DOCUMENT_NAMES in config.ts)
function documentName(file: string): string {
  return DOCUMENT_NAMES[file] ?? file.replace(/^\d+_/, '').replace(/\.pdf$/i, '').replaceAll('_', ' ')
}

export default function App() {
  const [question, setQuestion] = useState('')
  const [status, setStatus] = useState<Status>({ kind: 'idle' })
  const searching = status.kind === 'searching'

  async function submit(text: string) {
    const trimmed = text.trim()
    if (!trimmed || searching) return
    setStatus({ kind: 'searching' })
    try {
      setStatus({ kind: 'done', result: await askBackend(trimmed) })
    } catch (error) {
      setStatus({ kind: 'error', message: error instanceof Error ? error.message : String(error) })
    }
  }

  function handleSubmit(event: FormEvent) {
    event.preventDefault() // the form also submits on Enter
    submit(question)
  }

  function askExample(example: string) {
    setQuestion(example)
    submit(example)
  }

  return (
    <div className="page">
      <header className="header">
        <div className="logo">
          <BookIcon />
        </div>
        <div>
          <h1>Ask the Handbook</h1>
          <p className="subtitle">Answers from the HX-400 maintenance documents, with sources</p>
        </div>
      </header>

      <main>
        <form className="search" role="search" onSubmit={handleSubmit}>
          <div className="search-field">
            <SearchIcon />
            <input
              type="text"
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder="Ask a question, e.g. Which oil grade should I use?"
              aria-label="Question"
              autoFocus
            />
          </div>
          <button className="ask-button" type="submit" disabled={searching || !question.trim()}>
            Ask
          </button>
        </form>

        <div className="examples">
          <span className="examples-label">Try:</span>
          {EXAMPLE_QUESTIONS.map((example) => (
            <button
              key={example}
              type="button"
              className="example"
              disabled={searching}
              onClick={() => askExample(example)}
            >
              {example}
            </button>
          ))}
        </div>

        {searching && (
          <div className="status" role="status">
            <span className="spinner" aria-hidden="true" />
            Searching…
          </div>
        )}
        {status.kind === 'error' && (
          <div className="notice notice-error" role="alert">
            <AlertIcon />
            <p>{status.message}</p>
          </div>
        )}
        {status.kind === 'done' && <Result result={status.result} />}
      </main>
    </div>
  )
}

function Result({ result }: { result: AskResponse }) {
  if (!result.answered) {
    return (
      <div className="notice notice-unanswered">
        <InfoIcon />
        <p>{result.answer}</p>
      </div>
    )
  }

  return (
    <>
      <section className="answer-card" aria-label="Answer">
        <div className="card-label">Answer</div>
        <p className="answer-text">{result.answer}</p>
      </section>

      <section className="sources" aria-labelledby="sources-title">
        <h2 id="sources-title">
          Sources <span className="count">{result.sources.length}</span>
        </h2>
        {/* An answer always cites at least one source: the backend refuses otherwise. */}
        <ul className="source-list">
          {result.sources.map((source) => (
            <li key={source.chunk_id}>
              <SourceCard source={source} />
            </li>
          ))}
        </ul>
      </section>
    </>
  )
}

function SourceCard({ source }: { source: Source }) {
  const twoPages = source.page_end !== source.page
  const pages = twoPages ? `${source.page}–${source.page_end}` : `${source.page}`
  return (
    <details className="source">
      <summary>
        <span className="chip" title={source.document}>
          <span className="chip-name">{documentName(source.document)}</span>
          <span className="chip-page">· p.{pages}</span>
        </span>
        <span className="toggle">
          <span className="toggle-show">Show text</span>
          <span className="toggle-hide">Hide text</span>
          <ChevronIcon />
        </span>
      </summary>
      <div className="source-body">
        <div className="source-file">
          {source.document}, {twoPages ? 'pages' : 'page'} {pages}
        </div>
        <div className="chunk">{source.text}</div>
      </div>
    </details>
  )
}
