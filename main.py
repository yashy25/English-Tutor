from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from ai_client import get_writing_feedback

app = FastAPI(title="English Tutor")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


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
