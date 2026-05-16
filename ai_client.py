"""Wrapper around the Google Gemini API for the English Tutor app."""
from google import genai
from google.genai import types
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

client = genai.Client()  # reads GEMINI_API_KEY from env

MODEL = "gemini-2.5-flash"


# ---------------------------------------------------------------------------
# Data models for structured output
# ---------------------------------------------------------------------------

class QuizQuestion(BaseModel):
    """One question in a reading-comprehension quiz."""
    question: str
    # For multiple choice: list of 4 options (without A/B/C/D prefix).
    # For short answer: empty list.
    options: list[str] = []
    # For multiple choice: the letter of the correct option ("A", "B", "C", or "D").
    # For short answer: the expected answer text (used as a rubric).
    correct_answer: str


class ReadingMaterial(BaseModel):
    """A passage and its associated quiz questions."""
    passage: str
    questions: list[QuizQuestion]


class GradedAnswer(BaseModel):
    """Grading result for one student answer."""
    is_correct: bool
    feedback: str


class GradedQuiz(BaseModel):
    """Grading result for a whole quiz, plus a summary."""
    graded_answers: list[GradedAnswer]
    summary: str

WRITING_FEEDBACK_SYSTEM_PROMPT = """You are a friendly, encouraging English writing teacher who gives feedback on student writing.

Your feedback should be:
1. Encouraging - start with what the student did well
2. Specific - point to concrete examples from their writing
3. Actionable - give clear, concrete suggestions for improvement
4. Grade-appropriate - adjust your vocabulary, expectations, and tone to the student's grade level

When evaluating writing, consider these traits:
- Content and ideas: Does the writing address the prompt? Are ideas developed with detail?
- Organization: Is there a clear beginning, middle, and end? Do ideas flow logically?
- Voice and word choice: Does the writing show the student's personality? Are word choices effective?
- Sentence fluency: Do sentences vary in length and structure?
- Conventions: Grammar, spelling, punctuation, and capitalization

Grade-level guidance:

Grades 1 to 3 (Early Elementary):
- Use very simple language a young child can read
- Focus on complete sentences, capital letters at the start, periods at the end, and clear ideas
- Be very warm and encouraging
- Suggest only ONE or TWO specific things to work on - don't overwhelm them
- Celebrate small wins like correct capitalization or a vivid word

Grades 4 to 5 (Upper Elementary):
- Use accessible language
- Focus on paragraph structure, supporting details, and vivid word choice
- Suggest 2 to 3 things to improve
- Introduce ideas like "main idea" and "supporting details"

Grades 6 to 8 (Middle School):
- Address organization, voice, sentence variety, and basic essay structure
- Use more sophisticated vocabulary
- Suggest 3 to 4 things to work on
- Talk about transitions, tone, and audience

Grades 9 to 12 (High School):
- Address thesis development, argumentation, sophisticated word choice, and complex sentence structures
- Use academic vocabulary
- Be thoughtful and detailed
- Discuss rhetorical choices, evidence, and analysis

Format your response as plain text in this structure (do not use markdown, asterisks, or special characters):

Strengths
(2 to 3 specific things the student did well, with quoted examples from their writing)

Areas to improve
(Suggestions appropriate for their grade level, with specific examples)

Next step
(ONE concrete action the student can take to improve their next piece of writing)

Use blank lines between sections. Keep the tone warm and supportive throughout."""


def get_writing_feedback(grade: int, prompt: str, essay: str) -> str:
    """Get AI feedback on a student's writing.

    Args:
        grade: Student's grade level (1-12)
        prompt: The writing prompt the student was responding to
        essay: The student's written response

    Returns:
        Feedback text from Gemini, suitable for display in plain text or
        an HTML page that preserves whitespace.
    """
    user_message = (
        f"Student grade level: {grade}\n\n"
        f"Writing prompt the student responded to:\n{prompt}\n\n"
        f"Student's writing:\n{essay}\n\n"
        f"Please give grade-{grade}-appropriate feedback using the format described."
    )

    response = client.models.generate_content(
        model=MODEL,
        contents=user_message,
        config=types.GenerateContentConfig(
            system_instruction=WRITING_FEEDBACK_SYSTEM_PROMPT,
            max_output_tokens=4096,
        ),
    )

    return response.text


# ---------------------------------------------------------------------------
# Reading comprehension
# ---------------------------------------------------------------------------

def _passage_length_guidance(grade: int) -> str:
    """Suggested passage length for a given grade level."""
    if grade <= 3:
        return "100 to 180 words"
    if grade <= 5:
        return "200 to 300 words"
    if grade <= 8:
        return "300 to 500 words"
    return "500 to 800 words"


def generate_reading_material(
    grade: int,
    topic: str,
    num_questions: int,
    quiz_format: str,
) -> ReadingMaterial:
    """Generate a grade-appropriate reading passage and quiz.

    Args:
        grade: Student grade level (1-12)
        topic: Subject for the passage (e.g., "dolphins", "the moon landing")
        num_questions: How many quiz questions to generate
        quiz_format: "multiple_choice" or "short_answer"

    Returns:
        A ReadingMaterial with the passage and questions.
    """
    if quiz_format == "multiple_choice":
        format_instruction = (
            "Each question must have exactly 4 options (A, B, C, D), where exactly one is correct.\n"
            "Put the 4 option texts in the 'options' list (without 'A)', 'B)' prefixes).\n"
            "Set 'correct_answer' to the LETTER of the correct option ('A', 'B', 'C', or 'D')."
        )
    else:
        format_instruction = (
            "Each question is short-answer (no options).\n"
            "Leave 'options' as an empty list.\n"
            "Set 'correct_answer' to a 1-to-3 sentence model answer that captures the key idea. "
            "This will be used to grade the student's response."
        )

    prompt = f"""Create a reading-comprehension exercise for a grade {grade} student.

TOPIC: {topic}

PASSAGE REQUIREMENTS:
- Length: {_passage_length_guidance(grade)}
- Vocabulary and sentence complexity appropriate for grade {grade}
- Engaging and age-appropriate
- Self-contained — readers should not need outside knowledge

QUIZ REQUIREMENTS:
- Exactly {num_questions} questions
- Mix of literal questions (answers stated in text), inferential questions (read between lines), and (for grades 6+) analytical or vocabulary-in-context questions
- All questions must be answerable from the passage alone

QUESTION FORMAT:
{format_instruction}"""

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ReadingMaterial,
            max_output_tokens=4096,
        ),
    )

    return response.parsed


def grade_quiz(
    grade: int,
    passage: str,
    questions: list[QuizQuestion],
    student_answers: list[str],
    quiz_format: str,
) -> GradedQuiz:
    """Grade a student's quiz answers.

    Returns one GradedAnswer per question plus an overall summary.
    """
    answer_blocks = []
    for i, (q, ans) in enumerate(zip(questions, student_answers), start=1):
        if quiz_format == "multiple_choice":
            options_text = "\n".join(
                f"  {chr(65 + j)}. {opt}" for j, opt in enumerate(q.options)
            )
            answer_blocks.append(
                f"Question {i}: {q.question}\n"
                f"Options:\n{options_text}\n"
                f"Correct answer: {q.correct_answer}\n"
                f"Student's answer: {ans or '(no answer)'}"
            )
        else:
            answer_blocks.append(
                f"Question {i}: {q.question}\n"
                f"Model answer (use as rubric): {q.correct_answer}\n"
                f"Student's answer: {ans or '(no answer)'}"
            )

    grading_tone = (
        "Be very gentle and encouraging. Accept answers that show basic understanding "
        "even if spelling, grammar, or wording is imperfect."
        if grade <= 5
        else
        "Be fair but rigorous. Accept paraphrases that capture the key idea; "
        "do not penalize minor phrasing differences."
    )

    prompt = f"""You are grading a grade {grade} student's reading-comprehension quiz.

PASSAGE THE STUDENT READ:
{passage}

QUIZ ({quiz_format.replace('_', ' ')}) AND STUDENT ANSWERS:

{chr(10).join(answer_blocks)}

INSTRUCTIONS:
- {grading_tone}
- For each question, set 'is_correct' (true/false) and write 1-2 sentences of warm, helpful 'feedback' explaining why their answer is right or wrong. If wrong, mention the correct answer.
- After grading every question, write a 'summary' (2-3 sentences) of overall performance, ending with one encouragement or suggestion for next time."""

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=GradedQuiz,
            max_output_tokens=4096,
        ),
    )

    return response.parsed
