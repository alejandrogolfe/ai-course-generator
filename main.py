import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from prompts import (
    INTERVIEW_EVALUATOR_PROMPT,
    INTERVIEWER_PROMPT,
    NOTEBOOK_SECTION_PROMPT,
    PLANNER_PROMPT,
    SUMMARY_PROMPT,
    THEORY_PROMPT,
)
from state import CourseState, SessionPlan

load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0.3, max_tokens=4096)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_json(text: str) -> str:
    """Remove markdown code fences from LLM JSON responses."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return text.strip()


def _previous_context(state: CourseState) -> str:
    if not state.completed_summaries:
        return "Nothing yet — this is the first session."
    return "\n".join(
        f"- {summary}" for summary in state.completed_summaries
    )


def _section_name(section_index: int, total: int) -> str:
    if section_index == 0:
        return "imports_and_intro"
    elif section_index == total - 1:
        return "exercises"
    else:
        return "explanation_and_examples"


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def interviewer_node(state: CourseState) -> dict:
    """LLM generates dynamic questions about the topic."""
    print("\n🎤 Interviewer is thinking...")

    prompt = INTERVIEWER_PROMPT.format(
        topic=state.topic,
        total_hours=state.total_hours,
        session_hours=state.session_hours,
        user_answers=state.user_answers or "None yet.",
    )

    response = llm.invoke(prompt)
    questions = response.content

    # Interrupt: show questions to user and wait for answers
    print(f"\n{'='*60}")
    print(questions)
    print(f"{'='*60}")

    user_response = interrupt({"questions": questions})

    # Accumulate answers
    accumulated = state.user_answers
    if accumulated:
        accumulated += f"\n\n--- Round {state.interview_rounds + 1} ---\n"
    accumulated += f"Q: {questions}\nA: {user_response}"

    return {
        "user_answers": accumulated,
        "interview_rounds": state.interview_rounds + 1,
        "interview_messages": state.interview_messages + [
            {"role": "assistant", "content": questions},
            {"role": "user", "content": user_response},
        ],
    }


def evaluate_interview_node(state: CourseState) -> dict:
    """Decides if enough info has been gathered or another round is needed."""
    prompt = INTERVIEW_EVALUATOR_PROMPT.format(
        topic=state.topic,
        user_answers=state.user_answers,
    )
    response = llm.invoke(prompt)
    is_ready = "READY" in response.content.upper()

    # Force finish after 3 rounds to avoid infinite loops
    if state.interview_rounds >= 3:
        is_ready = True

    print(f"\n   🔍 Interview evaluation: {'READY ✅' if is_ready else 'NEED MORE INFO 🔄'}")
    return {"interview_done": is_ready}


def planner_node(state: CourseState) -> dict:
    """Generates (or revises) the course syllabus."""
    print(f"\n📚 {'Revising' if state.planner_rounds > 0 else 'Planning'} syllabus...")

    feedback_section = ""
    if state.syllabus_feedback and not state.syllabus_approved:
        feedback_section = f"""
The student reviewed the previous syllabus and requested these changes:
{state.syllabus_feedback}

Please revise the syllabus accordingly.
"""

    prompt = PLANNER_PROMPT.format(
        topic=state.topic,
        total_hours=state.total_hours,
        session_hours=state.session_hours,
        num_sessions=state.num_sessions,
        user_answers=state.user_answers,
        feedback_section=feedback_section,
    )

    response = llm.invoke(prompt)
    sessions_data = json.loads(_strip_json(response.content))
    syllabus = [SessionPlan(**s) for s in sessions_data]

    print(f"   ✅ Syllabus ready: {len(syllabus)} sessions")
    return {"syllabus": syllabus, "planner_rounds": state.planner_rounds + 1}


def validate_syllabus_node(state: CourseState) -> dict:
    """Shows the syllabus to the user and waits for approval or feedback."""

    # Format syllabus for display
    display = f"\n{'='*60}\n📋 PROPOSED SYLLABUS — {state.topic}\n{'='*60}\n"
    display += f"Total: {state.total_hours}h | {state.num_sessions} sessions × {state.session_hours}h\n\n"
    for s in state.syllabus:
        display += f"Session {s.session_number}: {s.title}\n"
        for t in s.topics:
            display += f"  • {t}\n"
        display += "\n"
    display += "="*60
    display += "\n\nType 'ok' to approve, or describe what you'd like to change:"

    print(display)

    user_response = interrupt({"syllabus": [s.model_dump() for s in state.syllabus]})

    approved = user_response.strip().lower() in {"ok", "yes", "sí", "si", "approved", "looks good", "perfect"}

    print(f"\n   {'✅ Syllabus approved!' if approved else '🔄 Revision requested.'}")
    return {
        "syllabus_approved": approved,
        "syllabus_feedback": user_response if not approved else "",
    }


def theory_writer_node(state: CourseState) -> dict:
    """Writes theory for the current topic of the current session."""
    session = state.syllabus[state.current_session]
    topic = session.topics[state.current_topic]

    print(f"\n✍️  Theory — Session {session.session_number}, topic {state.current_topic + 1}/{len(session.topics)}: '{topic}'")

    prompt = THEORY_PROMPT.format(
        course_topic=state.topic,
        user_answers=state.user_answers,
        previous_context=_previous_context(state),
        session_number=session.session_number,
        session_title=session.title,
        current_topic=topic,
        topic_index=state.current_topic + 1,
        total_topics=len(session.topics),
    )

    response = llm.invoke(prompt)
    theory_docs = state.theory_docs + [response.content]

    print(f"   ✅ Theory written ({len(response.content)} chars)")
    return {"theory_docs": theory_docs, "current_notebook_section": 0}


def notebook_section_node(state: CourseState) -> dict:
    """Generates one section of the notebook for the current topic."""
    session = state.syllabus[state.current_session]
    topic = session.topics[state.current_topic]
    section_idx = state.current_notebook_section
    total_sections = state.total_notebook_sections
    section_name = _section_name(section_idx, total_sections)

    print(f"   💻 Notebook section '{section_name}' ({section_idx + 1}/{total_sections}) for '{topic}'")

    prompt = NOTEBOOK_SECTION_PROMPT.format(
        course_topic=state.topic,
        session_number=session.session_number,
        session_title=session.title,
        current_topic=topic,
        section_name=section_name,
        section_index=section_idx + 1,
        total_sections=total_sections,
        user_answers=state.user_answers,
        previous_context=_previous_context(state),
    )

    response = llm.invoke(prompt)
    new_cells = json.loads(_strip_json(response.content))
    updated_cells = state.current_session_cells + new_cells

    return {
        "current_session_cells": updated_cells,
        "current_notebook_section": section_idx + 1,
    }


def advance_topic_node(state: CourseState) -> dict:
    """
    Called when all notebook sections for a topic are done.
    Advances to the next topic or next session and handles session assembly.
    """
    session = state.syllabus[state.current_session]
    next_topic = state.current_topic + 1

    # Are there more topics in this session?
    if next_topic < len(session.topics):
        print(f"\n   ➡️  Moving to topic {next_topic + 1} of session {session.session_number}")
        return {"current_topic": next_topic, "current_notebook_section": 0}

    # Session complete — assemble notebook and generate summary
    print(f"\n📦 Session {session.session_number} complete — assembling notebook...")

    notebook_full = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "cells": [
            {
                "cell_type": cell["cell_type"],
                "metadata": {},
                "source": cell["source"],
                **({"outputs": [], "execution_count": None} if cell["cell_type"] == "code" else {}),
            }
            for cell in state.current_session_cells
        ],
    }

    notebooks = state.notebooks + [notebook_full]

    # Generate session summary for context continuity
    summary_prompt = SUMMARY_PROMPT.format(
        session_number=session.session_number,
        session_title=session.title,
        topics=", ".join(session.topics),
    )
    summary_response = llm.invoke(summary_prompt)
    summaries = state.completed_summaries + [
        f"Session {session.session_number} ({session.title}): {summary_response.content}"
    ]

    return {
        "notebooks": notebooks,
        "current_session_cells": [],
        "current_topic": 0,
        "current_notebook_section": 0,
        "current_session": state.current_session + 1,
        "completed_summaries": summaries,
    }


def save_outputs_node(state: CourseState) -> dict:
    """Saves all generated content to disk."""
    output_dir = Path("output") / state.topic.replace(" ", "_").lower()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n💾 Saving course to: {output_dir}/")

    # Group theory docs by session/topic
    topic_count = 0
    for i, session in enumerate(state.syllabus):
        slug = f"session_{session.session_number:02d}_{session.title.replace(' ', '_').lower()}"

        # Save theory per topic
        for j, topic in enumerate(session.topics):
            if topic_count < len(state.theory_docs):
                topic_slug = topic.replace(" ", "_").lower()
                theory_path = output_dir / f"{slug}_topic_{j+1:02d}_{topic_slug}.md"
                theory_path.write_text(state.theory_docs[topic_count], encoding="utf-8")
                print(f"   📄 {theory_path.name}")
                topic_count += 1

        # Save assembled notebook
        if i < len(state.notebooks):
            notebook_path = output_dir / f"{slug}.ipynb"
            notebook_path.write_text(
                json.dumps(state.notebooks[i], indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
            print(f"   📓 {notebook_path.name}")

    # Save README with syllabus + student profile
    readme = f"# {state.topic} — Course\n\n"
    readme += f"**Student profile:**\n{state.user_answers}\n\n"
    readme += f"**Total:** {state.total_hours}h | {state.num_sessions} sessions × {state.session_hours}h\n\n"
    for s in state.syllabus:
        readme += f"## Session {s.session_number}: {s.title}\n"
        for t in s.topics:
            readme += f"- {t}\n"
        readme += "\n"

    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(f"   📋 README.md")
    print(f"\n🎉 Done! Course saved to: {output_dir}/")
    return {}


# ---------------------------------------------------------------------------
# Conditional edges
# ---------------------------------------------------------------------------

def route_after_interview(state: CourseState) -> str:
    return "planner" if state.interview_done else "interviewer"


def route_after_validation(state: CourseState) -> str:
    return "theory_writer" if state.syllabus_approved else "planner"


def route_after_notebook_section(state: CourseState) -> str:
    """More sections in this topic? Or move to next topic/session?"""
    if state.current_notebook_section < state.total_notebook_sections:
        return "notebook_section"
    return "advance_topic"


def route_after_advance(state: CourseState) -> str:
    """More sessions to process? Or save everything?"""
    if state.current_session < len(state.syllabus):
        return "theory_writer"
    return "save_outputs"


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    graph = StateGraph(CourseState)

    graph.add_node("interviewer", interviewer_node)
    graph.add_node("evaluate_interview", evaluate_interview_node)
    graph.add_node("planner", planner_node)
    graph.add_node("validate_syllabus", validate_syllabus_node)
    graph.add_node("theory_writer", theory_writer_node)
    graph.add_node("notebook_section", notebook_section_node)
    graph.add_node("advance_topic", advance_topic_node)
    graph.add_node("save_outputs", save_outputs_node)

    graph.add_edge(START, "interviewer")
    graph.add_edge("interviewer", "evaluate_interview")
    graph.add_conditional_edges("evaluate_interview", route_after_interview)
    graph.add_edge("planner", "validate_syllabus")
    graph.add_conditional_edges("validate_syllabus", route_after_validation)
    graph.add_edge("theory_writer", "notebook_section")
    graph.add_conditional_edges("notebook_section", route_after_notebook_section)
    graph.add_conditional_edges("advance_topic", route_after_advance)
    graph.add_edge("save_outputs", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_course_generator(topic: str, total_hours: float, session_hours: float):
    num_sessions = int(total_hours / session_hours)
    app = build_graph()

    config = {"configurable": {"thread_id": f"{topic.replace(' ', '_').lower()}_001"}}

    initial_state = CourseState(
        topic=topic,
        total_hours=total_hours,
        session_hours=session_hours,
        num_sessions=num_sessions,
    )

    print(f"\n🚀 Starting course generator for: '{topic}'")
    print(f"   {num_sessions} sessions × {session_hours}h = {total_hours}h total\n")

    # Stream events to handle interrupts
    for event in app.stream(initial_state, config=config):
        for node_name, node_output in event.items():
            if node_name == "__interrupt__":
                interrupt_data = node_output[0].value

                if "questions" in interrupt_data:
                    # Already printed in the node — just get user input
                    user_input = input("\nYour answer: ").strip()
                elif "syllabus" in interrupt_data:
                    user_input = input("\nYour response: ").strip()
                else:
                    user_input = input("\nYour response: ").strip()

                # Resume the graph with user input
                for resume_event in app.stream(
                    {"type": "human", "content": user_input},
                    config=config,
                ):
                    pass


if __name__ == "__main__":
    run_course_generator(
        topic="Seaborn",
        total_hours=4,
        session_hours=2,
    )
