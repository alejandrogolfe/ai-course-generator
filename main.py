import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from prompts import NOTEBOOK_PROMPT, PLANNER_PROMPT, THEORY_PROMPT
from state import CourseState, SessionPlan

load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0.3)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def planner_node(state: CourseState) -> dict:
    """Generates the full course syllabus as a list of SessionPlan objects."""
    print(f"\n📚 Planning course: '{state.topic}' — {state.num_sessions} sessions")

    prompt = PLANNER_PROMPT.format(
        topic=state.topic,
        total_hours=state.total_hours,
        session_hours=state.session_hours,
        num_sessions=state.num_sessions,
    )

    response = llm.invoke(prompt)
    raw = response.content.strip()

    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    sessions_data = json.loads(raw.strip())
    syllabus = [SessionPlan(**s) for s in sessions_data]

    print(f"   ✅ Syllabus ready: {len(syllabus)} sessions")
    return {"syllabus": syllabus}


def theory_writer_node(state: CourseState) -> dict:
    """Writes theory markdown for the current session."""
    session = state.syllabus[state.current_session]
    print(f"\n✍️  Writing theory — Session {session.session_number}: {session.title}")

    prompt = THEORY_PROMPT.format(
        course_topic=state.topic,
        session_number=session.session_number,
        session_title=session.title,
        topics=", ".join(session.topics),
        session_hours=session.duration_hours,
    )

    response = llm.invoke(prompt)
    theory_docs = state.theory_docs + [response.content]

    print(f"   ✅ Theory written ({len(response.content)} chars)")
    return {"theory_docs": theory_docs}


def notebook_writer_node(state: CourseState) -> dict:
    """Generates a .ipynb notebook for the current session."""
    session = state.syllabus[state.current_session]
    print(f"\n💻 Generating notebook — Session {session.session_number}: {session.title}")

    prompt = NOTEBOOK_PROMPT.format(
        course_topic=state.topic,
        session_number=session.session_number,
        session_title=session.title,
        topics=", ".join(session.topics),
        session_hours=session.duration_hours,
    )

    response = llm.invoke(prompt)
    raw = response.content.strip()

    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    notebook_data = json.loads(raw.strip())
    notebooks = state.notebooks + [notebook_data]

    # Advance to next session
    next_session = state.current_session + 1

    print(f"   ✅ Notebook generated — moving to session index {next_session}")
    return {"notebooks": notebooks, "current_session": next_session}


def save_outputs_node(state: CourseState) -> dict:
    """Saves all generated content to disk."""
    output_dir = Path("output") / state.topic.replace(" ", "_").lower()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n💾 Saving course to: {output_dir}/")

    for i, (theory, notebook) in enumerate(zip(state.theory_docs, state.notebooks)):
        session = state.syllabus[i]
        slug = f"session_{session.session_number:02d}_{session.title.replace(' ', '_').lower()}"

        # Save theory markdown
        theory_path = output_dir / f"{slug}_theory.md"
        theory_path.write_text(theory, encoding="utf-8")
        print(f"   📄 {theory_path.name}")

        # Build proper .ipynb structure and save
        notebook_full = {
            "nbformat": 4,
            "nbformat_minor": 5,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {"name": "python", "version": "3.10.0"},
            },
            "cells": [
                {
                    "cell_type": cell["cell_type"],
                    "metadata": {},
                    "source": cell["source"],
                    **({"outputs": [], "execution_count": None} if cell["cell_type"] == "code" else {}),
                }
                for cell in notebook.get("cells", [])
            ],
        }

        notebook_path = output_dir / f"{slug}.ipynb"
        notebook_path.write_text(json.dumps(notebook_full, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"   📓 {notebook_path.name}")

    # Save syllabus summary
    syllabus_md = f"# {state.topic} — Course Syllabus\n\n"
    syllabus_md += f"**Total hours:** {state.total_hours}h | **Sessions:** {state.num_sessions} × {state.session_hours}h\n\n"
    for s in state.syllabus:
        syllabus_md += f"## Session {s.session_number}: {s.title}\n"
        for t in s.topics:
            syllabus_md += f"- {t}\n"
        syllabus_md += "\n"

    (output_dir / "README.md").write_text(syllabus_md, encoding="utf-8")
    print(f"   📋 README.md")
    print(f"\n🎉 Course saved to: {output_dir}/")

    return {}


# ---------------------------------------------------------------------------
# Conditional edge
# ---------------------------------------------------------------------------

def should_continue(state: CourseState) -> str:
    """Loop back to theory_writer if sessions remain, otherwise save."""
    if state.current_session < len(state.syllabus):
        return "theory_writer"
    return "save_outputs"


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    graph = StateGraph(CourseState)

    graph.add_node("planner", planner_node)
    graph.add_node("theory_writer", theory_writer_node)
    graph.add_node("notebook_writer", notebook_writer_node)
    graph.add_node("save_outputs", save_outputs_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "theory_writer")
    graph.add_edge("theory_writer", "notebook_writer")
    graph.add_conditional_edges("notebook_writer", should_continue)
    graph.add_edge("save_outputs", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_course_generator(topic: str, total_hours: float, session_hours: float):
    num_sessions = int(total_hours / session_hours)

    initial_state = CourseState(
        topic=topic,
        total_hours=total_hours,
        session_hours=session_hours,
        num_sessions=num_sessions,
    )

    app = build_graph()

    print(f"\n🚀 Generating course: '{topic}'")
    print(f"   {num_sessions} sessions × {session_hours}h = {total_hours}h total\n")

    final_state = app.invoke(initial_state)
    return final_state


if __name__ == "__main__":
    run_course_generator(
        topic="Seaborn",
        total_hours=4,
        session_hours=2,
    )
