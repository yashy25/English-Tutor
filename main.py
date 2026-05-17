import json

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from ai_client import (
    get_writing_feedback,
    generate_reading_material,
    grade_quiz,
    QuizQuestion,
    generate_grammar_exercise,
    grade_grammar_exercise,
    GrammarSentence,
    generate_spelling_exercise,
    grade_spelling_exercise,
    SpellingWord,
    generate_literary_term_exercise,
    grade_literary_term_exercise,
    LiteraryTermQuestion,
    generate_vocab_exercise,
    generate_vocab_quiz,
    grade_vocab_exercise,
    VocabWord,
    VocabQuestion,
)

app = FastAPI(title="English Tutor")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

@app.get("/writing", response_class=HTMLResponse)
def writing_form(request: Request):
    return templates.TemplateResponse(
        "writing.html",
        {"request": request, "feedback": None, "submitted": {}},
    )


@app.post("/writing", response_class=HTMLResponse)
def writing_submit(
    request: Request,
    grade: int = Form(...),
    prompt: str = Form(...),
    essay: str = Form(...),
):
    feedback = get_writing_feedback(grade=grade, prompt=prompt, essay=essay)
    return templates.TemplateResponse(
        "writing.html",
        {
            "request": request,
            "feedback": feedback,
            "submitted": {"grade": grade, "prompt": prompt, "essay": essay},
        },
    )


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

@app.get("/reading", response_class=HTMLResponse)
def reading_form(request: Request):
    return templates.TemplateResponse(
        "reading.html",
        {"request": request, "phase": "form"},
    )


@app.post("/reading/passage", response_class=HTMLResponse)
def reading_generate(
    request: Request,
    grade: int = Form(...),
    topic: str = Form(...),
    num_questions: int = Form(4),
    quiz_format: str = Form(...),
):
    material = generate_reading_material(
        grade=grade,
        topic=topic,
        num_questions=num_questions,
        quiz_format=quiz_format,
    )
    return templates.TemplateResponse(
        "reading.html",
        {
            "request": request,
            "phase": "quiz",
            "grade": grade,
            "quiz_format": quiz_format,
            "passage": material.passage,
            "questions": material.questions,
            "questions_json": json.dumps(
                [q.model_dump() for q in material.questions]
            ),
        },
    )


@app.post("/reading/grade", response_class=HTMLResponse)
async def reading_grade(request: Request):
    form = await request.form()
    grade = int(form["grade"])
    passage = form["passage"]
    quiz_format = form["quiz_format"]
    questions = [QuizQuestion(**q) for q in json.loads(form["questions_json"])]
    student_answers = [
        form.get(f"answer_{i}", "") for i in range(len(questions))
    ]

    graded = grade_quiz(
        grade=grade,
        passage=passage,
        questions=questions,
        student_answers=student_answers,
        quiz_format=quiz_format,
    )

    results = [
        {
            "question": q,
            "student_answer": ans,
            "is_correct": g.is_correct,
            "feedback": g.feedback,
        }
        for q, ans, g in zip(questions, student_answers, graded.graded_answers)
    ]
    score = sum(1 for r in results if r["is_correct"])

    return templates.TemplateResponse(
        "reading.html",
        {
            "request": request,
            "phase": "results",
            "grade": grade,
            "quiz_format": quiz_format,
            "passage": passage,
            "results": results,
            "summary": graded.summary,
            "score": score,
            "total": len(questions),
        },
    )


# ---------------------------------------------------------------------------
# Grammar
# ---------------------------------------------------------------------------

@app.get("/grammar", response_class=HTMLResponse)
def grammar_form(request: Request):
    return templates.TemplateResponse(
        "grammar.html",
        {"request": request, "phase": "form"},
    )


@app.post("/grammar/generate", response_class=HTMLResponse)
def grammar_generate(
    request: Request,
    grade: int = Form(...),
    num_sentences: int = Form(5),
    focus: str = Form(...),
):
    exercise = generate_grammar_exercise(
        grade=grade, num_sentences=num_sentences, focus=focus
    )
    return templates.TemplateResponse(
        "grammar.html",
        {
            "request": request,
            "phase": "exercise",
            "grade": grade,
            "sentences": exercise.sentences,
            "sentences_json": json.dumps(
                [s.model_dump() for s in exercise.sentences]
            ),
        },
    )


@app.post("/grammar/grade", response_class=HTMLResponse)
async def grammar_grade(request: Request):
    form = await request.form()
    grade = int(form["grade"])
    sentences = [GrammarSentence(**s) for s in json.loads(form["sentences_json"])]
    student_answers = [
        form.get(f"answer_{i}", "") for i in range(len(sentences))
    ]

    graded = grade_grammar_exercise(
        grade=grade, sentences=sentences, student_answers=student_answers
    )

    results = [
        {
            "sentence": s,
            "student_answer": ans,
            "is_correct": g.is_correct,
            "feedback": g.feedback,
        }
        for s, ans, g in zip(sentences, student_answers, graded.graded_answers)
    ]
    score = sum(1 for r in results if r["is_correct"])

    return templates.TemplateResponse(
        "grammar.html",
        {
            "request": request,
            "phase": "results",
            "grade": grade,
            "results": results,
            "summary": graded.summary,
            "score": score,
            "total": len(sentences),
        },
    )


# ---------------------------------------------------------------------------
# Spelling
# ---------------------------------------------------------------------------

@app.get("/spelling", response_class=HTMLResponse)
def spelling_form(request: Request):
    return templates.TemplateResponse(
        "spelling.html",
        {"request": request, "phase": "form"},
    )


@app.post("/spelling/generate", response_class=HTMLResponse)
def spelling_generate(
    request: Request,
    grade: int = Form(...),
    num_words: int = Form(5),
):
    exercise = generate_spelling_exercise(grade=grade, num_words=num_words)
    return templates.TemplateResponse(
        "spelling.html",
        {
            "request": request,
            "phase": "exercise",
            "grade": grade,
            "words": exercise.words,
            "words_json": json.dumps(
                [w.model_dump() for w in exercise.words]
            ),
        },
    )


@app.post("/spelling/grade", response_class=HTMLResponse)
async def spelling_grade(request: Request):
    form = await request.form()
    grade = int(form["grade"])
    words = [SpellingWord(**w) for w in json.loads(form["words_json"])]
    student_answers = [
        form.get(f"answer_{i}", "") for i in range(len(words))
    ]

    graded = grade_spelling_exercise(
        grade=grade, words=words, student_answers=student_answers
    )

    results = [
        {
            "word": w,
            "student_answer": ans,
            "is_correct": g.is_correct,
            "feedback": g.feedback,
        }
        for w, ans, g in zip(words, student_answers, graded.graded_answers)
    ]
    score = sum(1 for r in results if r["is_correct"])

    return templates.TemplateResponse(
        "spelling.html",
        {
            "request": request,
            "phase": "results",
            "grade": grade,
            "results": results,
            "summary": graded.summary,
            "score": score,
            "total": len(words),
        },
    )


# ---------------------------------------------------------------------------
# Literary Terms
# ---------------------------------------------------------------------------

@app.get("/literary-terms", response_class=HTMLResponse)
def literary_terms_form(request: Request):
    return templates.TemplateResponse(
        "literary_terms.html",
        {"request": request, "phase": "form"},
    )


@app.post("/literary-terms/generate", response_class=HTMLResponse)
def literary_terms_generate(
    request: Request,
    grade: int = Form(...),
    num_questions: int = Form(5),
):
    exercise = generate_literary_term_exercise(
        grade=grade, num_questions=num_questions
    )
    return templates.TemplateResponse(
        "literary_terms.html",
        {
            "request": request,
            "phase": "exercise",
            "grade": grade,
            "questions": exercise.questions,
            "questions_json": json.dumps(
                [q.model_dump() for q in exercise.questions]
            ),
        },
    )


@app.post("/literary-terms/grade", response_class=HTMLResponse)
async def literary_terms_grade(request: Request):
    form = await request.form()
    grade = int(form["grade"])
    questions = [
        LiteraryTermQuestion(**q) for q in json.loads(form["questions_json"])
    ]
    student_answers = [
        form.get(f"answer_{i}", "") for i in range(len(questions))
    ]

    graded = grade_literary_term_exercise(
        grade=grade, questions=questions, student_answers=student_answers
    )

    results = [
        {
            "question": q,
            "student_answer": ans,
            "is_correct": g.is_correct,
            "feedback": g.feedback,
        }
        for q, ans, g in zip(questions, student_answers, graded.graded_answers)
    ]
    score = sum(1 for r in results if r["is_correct"])

    return templates.TemplateResponse(
        "literary_terms.html",
        {
            "request": request,
            "phase": "results",
            "grade": grade,
            "results": results,
            "summary": graded.summary,
            "score": score,
            "total": len(questions),
        },
    )


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

@app.get("/vocabulary", response_class=HTMLResponse)
def vocabulary_form(request: Request):
    return templates.TemplateResponse(
        "vocabulary.html",
        {"request": request, "phase": "form"},
    )


@app.post("/vocabulary/generate", response_class=HTMLResponse)
def vocabulary_generate(
    request: Request,
    grade: int = Form(...),
    num_words: int = Form(5),
    topic: str = Form(...),
):
    exercise = generate_vocab_exercise(grade=grade, num_words=num_words, topic=topic)
    quiz = generate_vocab_quiz(grade=grade, words=exercise.words)
    return templates.TemplateResponse(
        "vocabulary.html",
        {
            "request": request,
            "phase": "exercise",
            "grade": grade,
            "words": exercise.words,
            "questions": quiz.questions,
            "questions_json": json.dumps(
                [q.model_dump() for q in quiz.questions]
            ),
        },
    )


@app.post("/vocabulary/grade", response_class=HTMLResponse)
async def vocabulary_grade(request: Request):
    form = await request.form()
    grade = int(form["grade"])
    questions = [VocabQuestion(**q) for q in json.loads(form["questions_json"])]
    student_answers = [
        form.get(f"answer_{i}", "") for i in range(len(questions))
    ]

    graded = grade_vocab_exercise(
        grade=grade, questions=questions, student_answers=student_answers
    )

    results = [
        {
            "question": q,
            "student_answer": ans,
            "is_correct": g.is_correct,
            "feedback": g.feedback,
        }
        for q, ans, g in zip(questions, student_answers, graded.graded_answers)
    ]
    score = sum(1 for r in results if r["is_correct"])

    return templates.TemplateResponse(
        "vocabulary.html",
        {
            "request": request,
            "phase": "results",
            "grade": grade,
            "results": results,
            "summary": graded.summary,
            "score": score,
            "total": len(questions),
        },
    )
