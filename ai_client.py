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


# ---------------------------------------------------------------------------
# Grammar practice
# ---------------------------------------------------------------------------

class GrammarSentence(BaseModel):
    """One sentence with a grammar error and its correction."""
    original: str
    corrected: str
    explanation: str


class GrammarExercise(BaseModel):
    """A set of grammar sentences for practice."""
    sentences: list[GrammarSentence]


class GrammarGradedAnswer(BaseModel):
    """Grading result for one grammar correction."""
    is_correct: bool
    feedback: str


class GrammarGradedExercise(BaseModel):
    """Grading results for the full grammar exercise."""
    graded_answers: list[GrammarGradedAnswer]
    summary: str


def generate_grammar_exercise(grade: int, num_sentences: int, focus: str) -> GrammarExercise:
    """Generate sentences with grammar errors for the student to correct."""
    prompt = f"""Create a grammar exercise for a grade {grade} student.

FOCUS AREA: {focus}

REQUIREMENTS:
- Generate exactly {num_sentences} sentences, each containing ONE grammar error related to the focus area.
- The errors should be grade-appropriate for grade {grade}.
- Each sentence should be realistic and something a student might write.
- Provide the corrected version and a brief explanation of the rule.

For grade 1-3: focus on basic errors like capitalization, simple subject-verb agreement, basic punctuation.
For grade 4-5: include verb tense, pronoun agreement, comma usage.
For grade 6-8: include more complex errors like misplaced modifiers, run-on sentences, comma splices.
For grade 9-12: include subtle errors like dangling participles, parallel structure, subjunctive mood."""

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=GrammarExercise,
            max_output_tokens=4096,
        ),
    )
    return response.parsed


def grade_grammar_exercise(
    grade: int,
    sentences: list[GrammarSentence],
    student_answers: list[str],
) -> GrammarGradedExercise:
    """Grade a student's grammar corrections."""
    answer_blocks = []
    for i, (s, ans) in enumerate(zip(sentences, student_answers), start=1):
        answer_blocks.append(
            f"Sentence {i}: {s.original}\n"
            f"Expected correction: {s.corrected}\n"
            f"Student's correction: {ans or '(no answer)'}"
        )

    prompt = f"""You are grading a grade {grade} student's grammar exercise.

The student was asked to correct sentences with grammar errors.

SENTENCES AND STUDENT ANSWERS:

{chr(10).join(answer_blocks)}

INSTRUCTIONS:
- For each sentence, determine if the student's correction fixes the grammar error.
- Accept corrections that fix the targeted error even if the wording differs slightly from the expected answer.
- Set 'is_correct' (true/false) and write 1-2 sentences of helpful feedback.
- If wrong, explain what the correct answer should be and why.
- Write a 'summary' (2-3 sentences) of overall performance with encouragement."""

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=GrammarGradedExercise,
            max_output_tokens=4096,
        ),
    )
    return response.parsed


# ---------------------------------------------------------------------------
# Spelling practice
# ---------------------------------------------------------------------------

class SpellingWord(BaseModel):
    """A word for the spelling quiz with a clue."""
    word: str
    clue: str
    sentence_with_blank: str


class SpellingExercise(BaseModel):
    """A set of spelling words for practice."""
    words: list[SpellingWord]


class SpellingGradedAnswer(BaseModel):
    """Grading result for one spelling answer."""
    is_correct: bool
    feedback: str


class SpellingGradedExercise(BaseModel):
    """Grading results for the full spelling exercise."""
    graded_answers: list[SpellingGradedAnswer]
    summary: str


def generate_spelling_exercise(grade: int, num_words: int) -> SpellingExercise:
    """Generate a spelling quiz with clues and fill-in-the-blank sentences."""
    prompt = f"""Create a spelling exercise for a grade {grade} student.

REQUIREMENTS:
- Generate exactly {num_words} words appropriate for grade {grade} spelling level.
- For each word provide:
  1. The correctly spelled word
  2. A short clue or definition (1 sentence)
  3. A sentence using the word, but with the word replaced by a blank (use "___")

Grade-level guidance:
- Grades 1-3: common sight words, CVC patterns, basic phonics words (3-5 letters)
- Grades 4-5: multi-syllable words, common prefixes/suffixes, frequently misspelled words
- Grades 6-8: academic vocabulary, Greek/Latin roots, commonly confused words
- Grades 9-12: SAT/ACT vocabulary, advanced academic words, words with tricky spellings"""

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=SpellingExercise,
            max_output_tokens=4096,
        ),
    )
    return response.parsed


def grade_spelling_exercise(
    grade: int,
    words: list[SpellingWord],
    student_answers: list[str],
) -> SpellingGradedExercise:
    """Grade a student's spelling answers."""
    answer_blocks = []
    for i, (w, ans) in enumerate(zip(words, student_answers), start=1):
        answer_blocks.append(
            f"Word {i}:\n"
            f"  Clue: {w.clue}\n"
            f"  Correct spelling: {w.word}\n"
            f"  Student's answer: {ans or '(no answer)'}"
        )

    tone = (
        "Be very encouraging. Accept the answer if it's spelled correctly even with different capitalization."
        if grade <= 5
        else "Be fair. The spelling must be exact (case-insensitive)."
    )

    prompt = f"""You are grading a grade {grade} student's spelling exercise.

WORDS AND STUDENT ANSWERS:

{chr(10).join(answer_blocks)}

INSTRUCTIONS:
- {tone}
- For each word, set 'is_correct' (true/false) and write brief feedback.
- If incorrect, show the correct spelling and offer a memory tip.
- Write a 'summary' (2-3 sentences) of overall performance with encouragement."""

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=SpellingGradedExercise,
            max_output_tokens=4096,
        ),
    )
    return response.parsed


# ---------------------------------------------------------------------------
# Literary term practice
# ---------------------------------------------------------------------------

class LiteraryTermQuestion(BaseModel):
    """A question about a literary term with a passage example."""
    term: str
    definition: str
    passage_example: str
    question: str
    options: list[str]
    correct_answer: str


class LiteraryTermExercise(BaseModel):
    """A set of literary term questions."""
    questions: list[LiteraryTermQuestion]


class LiteraryTermGradedAnswer(BaseModel):
    """Grading result for one literary term answer."""
    is_correct: bool
    feedback: str


class LiteraryTermGradedExercise(BaseModel):
    """Grading results for the full literary term exercise."""
    graded_answers: list[LiteraryTermGradedAnswer]
    summary: str


def generate_literary_term_exercise(grade: int, num_questions: int) -> LiteraryTermExercise:
    """Generate a literary term identification exercise."""
    prompt = f"""Create a literary term practice exercise for a grade {grade} student.

REQUIREMENTS:
- Generate exactly {num_questions} questions about literary terms/devices.
- Each question includes:
  1. A literary term (the answer)
  2. A short definition of the term
  3. A short passage or sentence that demonstrates the term (original, not from any published work)
  4. A question asking the student to identify which literary device is being used
  5. Four multiple-choice options (the correct term plus 3 plausible distractors)
  6. The correct answer letter (A, B, C, or D)

Grade-level guidance:
- Grades 4-5: simile, metaphor, alliteration, personification, hyperbole, onomatopoeia
- Grades 6-8: add imagery, symbolism, foreshadowing, irony, tone, mood, theme, conflict types
- Grades 9-12: add motif, allegory, juxtaposition, satire, paradox, allusion, synecdoche, metonymy, epistrophe, anaphora

Put the 4 option texts in 'options' without letter prefixes.
Set 'correct_answer' to the LETTER (A, B, C, or D) of the correct option."""

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=LiteraryTermExercise,
            max_output_tokens=4096,
        ),
    )
    return response.parsed


def grade_literary_term_exercise(
    grade: int,
    questions: list[LiteraryTermQuestion],
    student_answers: list[str],
) -> LiteraryTermGradedExercise:
    """Grade a student's literary term answers."""
    answer_blocks = []
    for i, (q, ans) in enumerate(zip(questions, student_answers), start=1):
        options_text = "\n".join(
            f"  {chr(65 + j)}. {opt}" for j, opt in enumerate(q.options)
        )
        answer_blocks.append(
            f"Question {i}: {q.question}\n"
            f"Passage: {q.passage_example}\n"
            f"Options:\n{options_text}\n"
            f"Correct answer: {q.correct_answer} ({q.term})\n"
            f"Student's answer: {ans or '(no answer)'}"
        )

    prompt = f"""You are grading a grade {grade} student's literary term exercise.

QUESTIONS AND STUDENT ANSWERS:

{chr(10).join(answer_blocks)}

INSTRUCTIONS:
- For each question, set 'is_correct' (true/false) and write 1-2 sentences of feedback.
- If correct, briefly reinforce why the example demonstrates that literary device.
- If incorrect, explain what the correct answer is and help them understand the device.
- Write a 'summary' (2-3 sentences) of overall performance with encouragement."""

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=LiteraryTermGradedExercise,
            max_output_tokens=4096,
        ),
    )
    return response.parsed


# ---------------------------------------------------------------------------
# Vocabulary practice
# ---------------------------------------------------------------------------

class VocabWord(BaseModel):
    """A vocabulary word with definition and usage."""
    word: str
    definition: str
    example_sentence: str
    part_of_speech: str


class VocabExercise(BaseModel):
    """A set of vocabulary words and questions."""
    words: list[VocabWord]


class VocabQuestion(BaseModel):
    """A vocabulary quiz question."""
    question: str
    options: list[str]
    correct_answer: str


class VocabQuiz(BaseModel):
    """Generated quiz questions for vocabulary words."""
    questions: list[VocabQuestion]


class VocabGradedAnswer(BaseModel):
    """Grading result for one vocabulary answer."""
    is_correct: bool
    feedback: str


class VocabGradedExercise(BaseModel):
    """Grading results for the full vocabulary exercise."""
    graded_answers: list[VocabGradedAnswer]
    summary: str


def generate_vocab_exercise(grade: int, num_words: int, topic: str) -> VocabExercise:
    """Generate vocabulary words with definitions and example sentences."""
    prompt = f"""Create a vocabulary exercise for a grade {grade} student.

TOPIC/CATEGORY: {topic}

REQUIREMENTS:
- Generate exactly {num_words} vocabulary words appropriate for grade {grade}.
- For each word provide:
  1. The word itself
  2. A clear, student-friendly definition
  3. An example sentence demonstrating proper usage
  4. The part of speech (noun, verb, adjective, adverb, etc.)

Grade-level guidance:
- Grades 1-3: basic sight words, common nouns/verbs, simple descriptive adjectives
- Grades 4-5: grade-level academic vocabulary, words with prefixes/suffixes, content-area words
- Grades 6-8: academic vocabulary, words from literature, content-specific terms, words with Greek/Latin roots
- Grades 9-12: SAT/ACT prep words, advanced academic vocabulary, nuanced synonyms, domain-specific terminology"""

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=VocabExercise,
            max_output_tokens=4096,
        ),
    )
    return response.parsed


def generate_vocab_quiz(grade: int, words: list[VocabWord]) -> VocabQuiz:
    """Generate quiz questions for the vocabulary words."""
    word_blocks = []
    for i, w in enumerate(words, start=1):
        word_blocks.append(
            f"Word {i}: {w.word}\n"
            f"  Definition: {w.definition}\n"
            f"  Part of speech: {w.part_of_speech}\n"
            f"  Example: {w.example_sentence}"
        )

    prompt = f"""Create a multiple-choice vocabulary quiz for a grade {grade} student based on these words:

{chr(10).join(word_blocks)}

REQUIREMENTS:
- Create exactly {len(words)} questions, one per word.
- Mix question types:
  * "Which word means..." (definition to word)
  * "What does [word] mean?" (word to definition)
  * "Choose the word that best completes the sentence: ___" (context clues)
- Each question must have exactly 4 options using words or definitions from the list.
- Put the 4 option texts in 'options' without letter prefixes.
- Set 'correct_answer' to the LETTER (A, B, C, or D) of the correct option."""

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=VocabQuiz,
            max_output_tokens=4096,
        ),
    )
    return response.parsed


def grade_vocab_exercise(
    grade: int,
    questions: list[VocabQuestion],
    student_answers: list[str],
) -> VocabGradedExercise:
    """Grade a student's vocabulary quiz answers."""
    answer_blocks = []
    for i, (q, ans) in enumerate(zip(questions, student_answers), start=1):
        options_text = "\n".join(
            f"  {chr(65 + j)}. {opt}" for j, opt in enumerate(q.options)
        )
        answer_blocks.append(
            f"Question {i}: {q.question}\n"
            f"Options:\n{options_text}\n"
            f"Correct answer: {q.correct_answer}\n"
            f"Student's answer: {ans or '(no answer)'}"
        )

    prompt = f"""You are grading a grade {grade} student's vocabulary quiz.

QUESTIONS AND STUDENT ANSWERS:

{chr(10).join(answer_blocks)}

INSTRUCTIONS:
- For each question, set 'is_correct' (true/false) and write 1-2 sentences of feedback.
- If correct, reinforce the meaning or give a quick usage tip.
- If incorrect, explain the correct answer and help them remember the word.
- Write a 'summary' (2-3 sentences) of overall performance with encouragement."""

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=VocabGradedExercise,
            max_output_tokens=4096,
        ),
    )
    return response.parsed
