# 🎓 AI Course Generator

An AI-powered course generator built with **LangGraph** and **GPT-4o** that creates fully structured learning materials on any topic — complete with session theory documents and executable Jupyter notebooks, tailored to the student's profile through an interactive interview.

---

## ✨ Features

- **Interactive onboarding** — the agent interviews you before generating anything, adapting the course to your level, goals, and preferences
- **Human-in-the-loop validation** — review and request changes to the syllabus before any content is generated
- **Pedagogy-first content** — all material follows an Anchor → Teach → Apply structure designed for fast learners
- **Granular notebook generation** — notebooks are built section by section per topic (intro, concept blocks, exercises), avoiding LLM token limits
- **Automatic code validation** — every code cell is executed in a subprocess; failures are sent back to the LLM for fixing before saving
- **Session continuity** — each session generates a summary injected as context into the next, avoiding repetition

---

## 🏗️ Architecture

The project uses a **LangGraph state machine** with conditional edges and human-in-the-loop interrupts:

```
START
  │
  ▼
[interviewer] ◄──────────────────────────────┐
  │  interrupt()                              │
  ▼                                           │
[evaluate_interview] ── "needs more" ─────────┘
  │ "ready"
  ▼
[planner] ◄──────────────────────────────────┐
  │                                           │
  ▼                                           │
[validate_syllabus]  ── "revise" ─────────────┘
  │  interrupt()
  │ "approved"
  ▼
[theory_writer]  ◄─────────────────────────────┐
  │                                             │
  ▼                                             │
[notebook_section] ◄──┐                         │
  │                   │ more sections            │
  ▼                   │                         │
route ────────────────┘                         │
  │ all sections done                           │
  ▼                                             │
[validate_code] ◄──┐                            │
  │                │ failures + retries left     │
  ▼                │                            │
route ─────────────┘                            │
  │ clean (or max retries)                      │
  ▼                                             │
[advance_topic] ── more topics ─────────────────┘
  │ session complete          more sessions ────┘
  ▼
[save_outputs] → END
```

---

## 📁 Project structure

```
ai-course-generator/
├── state.py          # Pydantic models for graph state
├── prompts.py        # All LLM prompt templates
├── main.py           # LangGraph graph + all nodes + entry point
├── .env.example      # Environment variable template
├── requirements.txt
└── output/           # Generated courses (git-ignored)
```

### `state.py` — Pydantic models

```python
class SessionPlan(BaseModel):
    session_number: int
    title: str
    topics: list[str]
    duration_hours: float

class CodeValidationResult(BaseModel):
    cell_index: int
    success: bool
    error: str = ""
    fixed_source: list[str] = []

class CourseState(BaseModel):
    # input, interview state, syllabus, granular progress,
    # session continuity summaries, generated output, validation results
```

### `prompts.py` — Prompt templates

| Prompt | Purpose |
|--------|---------|
| `INTERVIEWER_PROMPT` | Generates dynamic questions to profile the student |
| `INTERVIEW_EVALUATOR_PROMPT` | Decides if enough info has been gathered |
| `PLANNER_PROMPT` | Builds the syllabus following Anchor→Teach→Apply pedagogy |
| `THEORY_PROMPT` | Writes topic theory: anchor, explanation, inline example, key takeaways |
| `NOTEBOOK_SECTION_PROMPT` | Generates one notebook section: intro / concept block / exercises |
| `CODE_FIX_PROMPT` | Repairs a failing code cell given its error message |
| `SUMMARY_PROMPT` | Summarises a completed session for continuity context |

### `main.py` — Graph nodes

| Node | Role |
|------|------|
| `interviewer` | LLM generates questions, `interrupt()` waits for user answers |
| `evaluate_interview` | Decides if another round is needed (max 3 rounds) |
| `planner` | Generates or revises the syllabus |
| `validate_syllabus` | Shows syllabus to user, `interrupt()` waits for approval |
| `theory_writer` | Writes theory per topic, computes dynamic section count |
| `notebook_section` | Generates one notebook section via `llm_large` (16k tokens) |
| `validate_code` | Executes cells cumulatively in a subprocess, LLM-fixes failures |
| `advance_topic` | Advances progress counter, assembles session notebook |
| `save_outputs` | Writes `.md` theory files, `.ipynb` notebooks, and `README.md` |

---

## 🔑 Key design decisions

**No supervisor agent** — the flow is deterministic. A conditional edge handles the session/topic loop cleanly without adding an extra LLM call for routing.

**Pydantic over TypedDict** — stronger validation, better IDE support, and cleaner serialisation across checkpoint saves.

**Dynamic notebook sections** — `total_notebook_sections` is computed per topic based on its complexity, not hardcoded. Range: 3–6 sections.

**Cumulative code execution** — cells are validated by running all preceding cells together in a single subprocess, correctly simulating notebook state (imports, variables, dataframes defined earlier are available).

**Two LLM instances** — `llm` (8k tokens) for short structured tasks; `llm_large` (16k tokens) only for notebook sections where output can be very long.

**Resilient JSON parsing** — `_safe_parse_cells()` recovers truncated LLM responses by finding the last complete JSON object rather than crashing.

---

## 🚀 Getting started

### 1. Clone and install

```bash
git clone git@github.com:alejandrogolfe/ai-course-generator.git
cd ai-course-generator
pip install -r requirements.txt
```

### 2. Set up your API key

```bash
cp .env.example .env
# Add your OpenAI key to .env
```

### 3. Run

```bash
python main.py
```

The agent will interview you, propose a syllabus for your approval, then generate all content automatically.

To change the topic, edit the bottom of `main.py`:

```python
run_course_generator(
    topic="FastAPI",
    total_hours=6,
    session_hours=2,
)
```

---

## 📂 Output example

For `topic="Seaborn", total_hours=4, session_hours=2`:

```
output/seaborn/
├── README.md
├── session_01_foundations_topic_01_figure_anatomy.md
├── session_01_foundations_topic_02_color_palettes.md
├── session_01_foundations_topic_03_mini_project.md
├── session_01_foundations.ipynb
├── session_02_statistical_plots_topic_01_distributions.md
├── session_02_statistical_plots_topic_02_relationships.md
├── session_02_statistical_plots_topic_03_mini_project.md
└── session_02_statistical_plots.ipynb
```

---

## 🛠️ Tech stack

| Tool | Role |
|------|------|
| [LangGraph](https://github.com/langchain-ai/langgraph) | State machine orchestration with human-in-the-loop |
| [LangChain OpenAI](https://python.langchain.com/) | GPT-4o integration |
| [Pydantic v2](https://docs.pydantic.dev/) | State validation and modelling |
| GPT-4o | All LLM tasks |

---

## 📄 License

MIT
