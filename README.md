# English Tutor

A web app that helps students practice English skills with AI-powered feedback.

## Features (planned)

- Writing prompts with AI feedback scaled by grade level
- Reading passages with multiple-choice or short-answer quizzes
- Grammar checker
- Literary term practice (using and identifying terms)
- Grade-aware vocabulary practice
- Spelling practice

## Setup

```bash
# Clone and enter the project
git clone https://github.com/yashy25/english-tutor.git
cd english-tutor

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Add your API key
cp .env.example .env
# Then edit .env and paste in your Gemini API key from aistudio.google.com

# Run the app
uvicorn main:app --reload
```

Open http://localhost:8000 in your browser.

## Tech stack

- **FastAPI** — web framework
- **Google Gemini API** — AI feedback (free tier)
- **Jinja2** — HTML templating
- **SQLite** — local database (added later)
