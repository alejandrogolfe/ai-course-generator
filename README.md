# 🎓 AI Course Generator

An AI-powered course generator built with **LangGraph** and **GPT-4o** that automatically creates structured learning materials on any topic — including session-by-session theory documents and executable Jupyter notebooks.

---

## 🧠 What it does

Given a topic (e.g. *"Seaborn"*, *"FastAPI"*, *"Reinforcement Learning"*) and a time structure (e.g. 4 hours split into 2-hour sessions), the agent:

1. **Plans** a full course syllabus with progressive session topics
2. **Writes** detailed theory for each session in Markdown
3. **Generates** a ready-to-run Jupyter notebook (`.ipynb`) per session
4. **Saves** everything to a structured `output/` folder

---

## 🏗️ Architecture

The project uses a **LangGraph state machine** with a conditional loop — no unnecessary supervisor agent:

```
START → [planner] → [theory_writer] → [notebook_writer] → (more sessions?) → [theory_writer] ...
                                                                ↓ (done)
                                                         [save_outputs] → END
```

The conditional edge in `notebook_writer` checks whether all sessions have been processed and either loops back or terminates.

---

## 📁 Project structure

```
ai-course-generator/
├── state.py          # Pydantic models for graph state
├── prompts.py        # All LLM prompt templates
├── main.py           # LangGraph graph definition + entry point
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

class CourseState(BaseModel):
    topic: str
    total_hours: float
    session_hours: float
    num_sessions: int = 0
    syllabus: list[SessionPlan] = []
    current_session: int = 0
    theory_docs: list[str] = []
    notebooks: list[dict] = []
```

### `prompts.py` — Prompt templates

Contains three prompt templates:
- `PLANNER_PROMPT` — generates the syllabus as structured JSON
- `THEORY_PROMPT` — writes detailed markdown theory for a session
- `NOTEBOOK_PROMPT` — generates `.ipynb` cell content for a session

### `main.py` — Graph

Defines the four nodes (`planner`, `theory_writer`, `notebook_writer`, `save_outputs`), wires them together with a conditional loop, and exposes a `run_course_generator()` entry point.

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
# Edit .env and add your OpenAI key
```

```env
OPENAI_API_KEY=sk-your-key-here
```

### 3. Run

```bash
python main.py
```

By default it generates a **Seaborn** course (4h / 2 sessions). To change it, edit the bottom of `main.py`:

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
├── README.md                                        ← full syllabus
├── session_01_introduction_to_seaborn_theory.md
├── session_01_introduction_to_seaborn.ipynb
├── session_02_advanced_visualizations_theory.md
└── session_02_advanced_visualizations.ipynb
```

---

## 🛠️ Tech stack

| Tool | Role |
|------|------|
| [LangGraph](https://github.com/langchain-ai/langgraph) | Agent graph orchestration |
| [LangChain OpenAI](https://python.langchain.com/) | GPT-4o integration |
| [Pydantic v2](https://docs.pydantic.dev/) | State validation and modeling |
| GPT-4o | Course planning, theory writing, notebook generation |

---

## 💡 Design decisions

- **No supervisor agent** — the flow is deterministic, so a supervisor would only add latency and cost. A conditional edge handles the session loop cleanly.
- **Pydantic over TypedDict** — stronger validation, better IDE support, and cleaner serialization.
- **Prompts as a separate module** — makes it easy to iterate on prompt quality without touching the graph logic.
- **Self-contained notebooks** — all generated code uses built-in datasets (seaborn, sklearn) so notebooks run without extra setup.

---

## 📄 License

MIT
