
## Installation and Setup

### Prerequisites

Before installing the project, make sure the following are installed:

| Software | Required version |
|---|---|
| Python | 3.12 |
| Node.js | 22.12 or newer (Node 22 recommended) |
| npm | Included with Node.js |
| Git | Any recent version |

**Important:** Python 3.11 is not compatible with the pinned
NumPy 2.5.5 dependency. Node.js 20.11 is too old for the
frontend's Vite dependencies.

Check your versions:

```bash
python3 --version
node --version
npm --version
```

### 1. Clone the repository

```bash
git clone https://github.com/Christinaliti/Ask-Handbook.git
cd Ask-Handbook
```

### 2. Install dependencies

On macOS, if you have Homebrew:

```bash
brew install python@3.12 node@22
export PATH="$(brew --prefix node@22)/bin:$PATH"
```

Create a Python virtual environment using Python 3.12:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

If `python3.12` is not found after installing with
Homebrew, use:

```bash
"$(brew --prefix python@3.12)/bin/python3.12" -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Install the frontend dependencies:

```bash
cd frontend
npm ci
cd ..
```

### 3. Configure Azure OpenAI

Create your environment configuration:

```bash
cp backend/.env.example backend/.env
```

Open `backend/.env` and add the Azure OpenAI credentials
provided for the assignment.


### 4. Start the backend

From the project root, with the virtual environment
activated:

```bash
python -m uvicorn app:app --app-dir backend --reload
```

The backend should be accessible at:

http://127.0.0.1:8000

### 5. Start the frontend

Open a second terminal:

```bash
cd frontend
npm run dev
```

Open the URL displayed in the terminal, typically:

http://localhost:5173

### 6. Run evaluation tests

From the project root, with the Python virtual
environment activated:

```bash
python eval/run_eval.py
```

The evaluation uses the same answering pipeline as the
web application and writes results to `eval/results.json`.

See [Failure Analysis](eval/failure_analysis.md) for
the documented evaluation results and limitations.
