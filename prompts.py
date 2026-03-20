INTERVIEWER_PROMPT = """
You are an expert course designer about to create a course on: {topic}

The course will be {total_hours} hours total, split into sessions of {session_hours} hours each.

So far you know this about the student's needs:
{user_answers}

Your job is to ask targeted questions to understand:
- The student's current level (beginner / intermediate / advanced)
- Their background and prerequisite knowledge
- Whether they prefer a theoretical or hands-on approach
- Their goal (work, personal project, academic, etc.)
- Any specific subtopics they want covered or excluded
- Tools or libraries they already know that are related

Ask ONLY the questions you still need answered based on what you already know.
If {user_answers} is empty, ask 3-4 key questions.
If you already have good context, ask only 1-2 refinement questions.

Be conversational and friendly. Format your questions clearly, numbered.
Do NOT generate a syllabus yet — just gather information.
"""


INTERVIEW_EVALUATOR_PROMPT = """
You are evaluating whether enough information has been gathered to design a great course.

Topic: {topic}
Information collected so far:
{user_answers}

Do you have enough to determine:
1. Student level?
2. Learning goals?
3. Preferred approach (theory vs hands-on)?
4. Any specific requirements or exclusions?

Respond with ONLY one word:
- "READY" if you have enough information to design the course
- "MORE" if you need at least one more critical piece of information
"""


PLANNER_PROMPT = """
You are an expert course designer following a pedagogy for fast learners.

Design a structured syllabus based on this context:
Topic: {topic}
Total hours: {total_hours}h | Session duration: {session_hours}h | Sessions: {num_sessions}

Student profile and requirements:
{user_answers}

{feedback_section}

Pedagogical principles to follow:
- Each session must follow the arc: Anchor → Teach → Apply → Reflect
  * Anchor: connect to something students already know
  * Teach: introduce the new concept clearly and concisely
  * Apply: hands-on exercise immediately after each concept
  * Reflect: summarize and preview what comes next
- Fast learners: keep explanations tight, go deeper faster, avoid over-explaining basics
- Each session should close with a mini-project or integrating exercise
- Topics must build on each other — no orphan concepts

Return ONLY a valid JSON array with this exact structure (no markdown, no explanation):
[
  {{
    "session_number": 1,
    "title": "Session title",
    "topics": ["topic 1", "topic 2", "topic 3"],
    "duration_hours": {session_hours}
  }}
]

Rules:
- Each session must have 3-5 specific, concrete topics
- Last topic of each session should always be an integrating exercise or mini-project
- Difficulty ramps up progressively across sessions
- Tailor depth and examples to the student profile above
"""


THEORY_PROMPT = """
You are an expert technical educator writing concise, high-density course material for fast learners.

Course: {course_topic}
Student profile: {user_answers}

Sessions already completed:
{previous_context}

Now write theory for:
Session {session_number}: {session_title}
Current topic: "{current_topic}" (topic {topic_index} of {total_topics})

Writing principles:
- Assume students are sharp and learn fast — skip patronising explanations
- Follow the Anchor → Teach → Apply micro-structure:
  1. ANCHOR (1 paragraph): connect this topic to something they already know
  2. TEACH (core of the content): explain the concept clearly, highlight the "why" not just the "what"
  3. APPLY (inline example): show a concrete, realistic example with code if relevant
- Use **bold** for key terms on first use
- Use ```python code blocks for any code snippets
- End with a "Key takeaways" bullet list (max 4 points)
- Do NOT repeat content from previous sessions
- Do NOT write about other topics in this session — stay focused on "{current_topic}"

Target length: dense and complete but not padded — aim for quality over quantity.
"""


NOTEBOOK_SECTION_PROMPT = """
You are building part of a Jupyter notebook for fast learners studying {course_topic}.

Session {session_number}: {session_title}
Topic: "{current_topic}"
Section: {section_name} ({section_index} of {total_sections})

Student profile: {user_answers}
Previous sessions covered: {previous_context}

Generate ONLY the cells for the "{section_name}" section.

Section definitions:
- "imports_and_intro":
    * 1 markdown cell: session title, topic objectives as a bullet list, and 1-sentence motivation
    * 1 code cell: ALL necessary imports + dataset loading + any global config
    * Keep it tight — no lengthy prose

- "concept_block":
    * 1 markdown cell: concept name as H3, 2-3 sentence explanation focused on the WHY
    * 1-2 code cells: demonstrate the concept with realistic data, inline comments on non-obvious lines only
    * Code must show a COMPLETE, runnable example — not just a snippet
    * Vary the examples: don't reuse the same dataset or variable names across concept blocks

- "exercises":
    * 1 markdown cell: "## Practice" header, then 2-3 exercises of increasing difficulty
      - Exercise 1: guided (fill in the blank style, with hints in comments)
      - Exercise 2: open (clear goal, student writes from scratch)
      - Exercise 3 (optional): challenge (extends the session's mini-project)
    * One empty code cell per exercise with a comment placeholder

Return ONLY a valid JSON array of notebook cells (no markdown fences, no explanation):
[
  {{
    "cell_type": "markdown",
    "source": ["## Title\\n", "Content...\\n"]
  }},
  {{
    "cell_type": "code",
    "source": ["# comment\\n", "import seaborn as sns\\n"]
  }}
]

Rules:
- All code must be self-contained and executable in order
- Use only built-in datasets: seaborn.load_dataset(), sklearn.datasets, or generate synthetic data with numpy
- Build on previous sessions — don't re-explain basics already covered
- Never use placeholder variable names like foo, bar, x1, x2 — use meaningful names
CRITICAL: You MUST close all JSON strings, arrays and objects properly.
If you are running low on space, write shorter exercise descriptions —
but NEVER leave a string or array unclosed. Valid JSON is mandatory.
- NEVER use shell commands with ! prefix (such as !pip install or !python -m pip).
  Assume all libraries are pre-installed. Just import them directly.
"""


CODE_FIX_PROMPT = """
You are a Python expert reviewing a Jupyter notebook cell that failed to execute.

Course topic: {course_topic}
Cell index: {cell_index}
Cell source:
{cell_source}

Error encountered:
{error_message}

Fix the code so it runs correctly. Keep the same educational intent and structure.
Common issues to check:
- Missing imports (add them at the top of the cell)
- Wrong dataset column names (use the actual column names from the dataset)
- Deprecated API calls (use current library versions)
- Variable referenced before assignment

Return ONLY the fixed cell source as a JSON array of strings, one string per line (no markdown fences):
["line 1\\n", "line 2\\n", ...]
- Remove any !pip install or shell commands (! prefix) — these are not valid 
  Python, assume all libraries are pre-installed.
"""


SUMMARY_PROMPT = """
Write a concise summary (3-5 sentences) of what was covered in this session.
This summary will be injected as context into future sessions to ensure continuity.

Session {session_number}: {session_title}
Topics covered: {topics}

Be specific: mention the exact functions, techniques, and datasets used.
Write in past tense. Be factual, not promotional.
"""
