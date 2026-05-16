import json

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from ai_client import (
    get_writing_feedback,
    generate_reading_material,
    grade_quiz,
    QuizQuestion,
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
