import json
import subprocess
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from prompts import (
    CODE_FIX_PROMPT,
    INTERVIEW_EVALUATOR_PROMPT,
    INTERVIEWER_PROMPT,
    NOTEBOOK_SECTION_PROMPT,
    PLANNER_PROMPT,
    SUMMARY_PROMPT,
    THEORY_PROMPT,
)
from state import CodeValidationResult, CourseState, SessionPlan


load_dotenv(override=True)


llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.3,
    max_tokens=8192,
).with_config({"tags": ["llm-standard"]})

llm_large = ChatOpenAI(
    model="gpt-4o",
    temperature=0.3,
    max_tokens=16000,
).with_config({"tags": ["llm-large"]})


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


def _safe_parse_cells(text: str) -> list[dict]:
    """
    Parse LLM cell JSON robustly.
    If the JSON is truncated (common with long outputs), attempts to recover
    by closing the array at the last complete object boundary.
    """
    cleaned = _strip_json(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Find the last complete cell by looking for the last "}," or "}" before end
        last_close = max(cleaned.rfind("},"), cleaned.rfind("}\n]"), cleaned.rfind("}]"))
        if last_close == -1:
            raise ValueError(f"Could not recover truncated JSON. Raw response:\n{cleaned[:300]}")
        # Truncate at last complete object and close the array
        recovered = cleaned[:last_close + 1] + "]"
        try:
            cells = json.loads(recovered)
            print(f"   WARNING: JSON was truncated, recovered {len(cells)} cells from partial response")
            return cells
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON recovery failed: {e}\nRaw: {cleaned[:300]}")


def _previous_context(state: CourseState) -> str:
    if not state.completed_summaries:
        return "Nothing yet — this is the first session."
    return "\n".join(f"- {s}" for s in state.completed_summaries)


def _calc_sections(num_subtopics: int) -> int:
    """
    Dynamic section count per topic:
      1 imports_and_intro + N concept_blocks (one per subtopic) + 1 exercises
    Minimum 3, maximum 6 to keep generation manageable.
    """
    return max(3, min(6, 1 + num_subtopics + 1))


def _section_name(section_index: int, total: int) -> str:
    if section_index == 0:
        return "imports_and_intro"
    elif section_index == total - 1:
        return "exercises"
    else:
        return "concept_block"


def _execute_cells_cumulative(cells: list[dict], up_to_index: int) -> tuple[bool, str]:
    """
    Execute all code cells from index 0 up to and including up_to_index
    in a single subprocess, simulating notebook state accumulation.
    Returns (success, error_message).
    """
    # Gather all code cells up to the target index
    code_blocks = []
    for i, cell in enumerate(cells[:up_to_index + 1]):
        if cell["cell_type"] == "code":
            code_blocks.append(f"# --- Cell {i} ---")
            code_blocks.append("".join(cell["source"]))

    if not code_blocks:
        return True, ""

    full_code = "\n".join(code_blocks)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(full_code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0:
            return True, ""
        return False, result.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "TimeoutExpired: execution exceeded 60 seconds"
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def interviewer_node(state: CourseState) -> dict:
    """LLM generates dynamic questions about the topic."""
    print("\n Interviewer is thinking...")

    prompt = INTERVIEWER_PROMPT.format(
        topic=state.topic,
        total_hours=state.total_hours,
        session_hours=state.session_hours,
        user_answers=state.user_answers or "None yet.",
    )

    response = llm.invoke(prompt)
    questions = response.content

    print(f"\n{'='*60}")
    print(questions)
    print(f"{'='*60}")

    user_response = interrupt({"questions": questions})

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

    if state.interview_rounds >= 3:
        is_ready = True

    print(f"\n   Interview evaluation: {'READY' if is_ready else 'NEED MORE INFO'}")
    return {"interview_done": is_ready}


def planner_node(state: CourseState) -> dict:
    """Generates (or revises) the course syllabus."""
    print(f"\n {'Revising' if state.planner_rounds > 0 else 'Planning'} syllabus...")

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
    try:
        sessions_data = json.loads(_strip_json(response.content))
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Planner returned invalid JSON: {e}\nRaw response:\n{response.content[:300]}"
        )
    syllabus = [SessionPlan(**s) for s in sessions_data]

    print(f"   Syllabus ready: {len(syllabus)} sessions")
    return {"syllabus": syllabus, "planner_rounds": state.planner_rounds + 1}


def validate_syllabus_node(state: CourseState) -> dict:
    """Shows the syllabus to the user and waits for approval or feedback."""
    display = f"\n{'='*60}\nPROPOSED SYLLABUS: {state.topic}\n{'='*60}\n"
    display += f"Total: {state.total_hours}h | {state.num_sessions} sessions x {state.session_hours}h\n\n"
    for s in state.syllabus:
        display += f"Session {s.session_number}: {s.title}\n"
        for t in s.topics:
            display += f"  - {t}\n"
        display += "\n"
    display += "="*60
    display += "\n\nType 'ok' to approve, or describe what you'd like to change:"

    print(display)

    user_response = interrupt({"syllabus": [s.model_dump() for s in state.syllabus]})

    approved = user_response.strip().lower() in {
        "ok", "yes", "si", "sí", "approved", "looks good", "perfect", "vale"
    }

    print(f"\n   {'Syllabus approved!' if approved else 'Revision requested.'}")
    return {
        "syllabus_approved": approved,
        "syllabus_feedback": user_response if not approved else "",
    }


def theory_writer_node(state: CourseState) -> dict:
    """Writes theory for the current topic and calculates dynamic section count."""
    session = state.syllabus[state.current_session]
    topic = session.topics[state.current_topic]

    print(f"\n Theory — Session {session.session_number}, "
          f"topic {state.current_topic + 1}/{len(session.topics)}: '{topic}'")

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

    # Dynamic section count: 1 intro + 1 concept_block per subtopic word + 1 exercises
    # Approximate subtopics from topic name word count (simple heuristic)
    subtopic_count = min(3, max(1, len(topic.split()) // 2))
    total_sections = _calc_sections(subtopic_count)

    print(f"   Theory written ({len(response.content)} chars) | "
          f"Notebook sections for this topic: {total_sections}")

    return {
        "theory_docs": theory_docs,
        "current_notebook_section": 0,
        "total_notebook_sections": total_sections,
        "validation_attempts": 0,
        "validation_results": [],
    }


def notebook_section_node(state: CourseState) -> dict:
    session = state.syllabus[state.current_session]
    topic = session.topics[state.current_topic]
    section_idx = state.current_notebook_section
    total_sections = state.total_notebook_sections
    section_name = _section_name(section_idx, total_sections)

    print(f"   Notebook section '{section_name}' "
          f"({section_idx + 1}/{total_sections}) for '{topic}'")

    # Ejercicios: 3 llamadas separadas en lugar de una
    if section_name == "exercises":
        all_cells = []
        for exercise_num in range(1, 4):
            prompt = NOTEBOOK_SECTION_PROMPT.format(
                course_topic=state.topic,
                session_number=session.session_number,
                session_title=session.title,
                current_topic=topic,
                section_name=f"exercises",
                section_index=section_idx + 1,
                total_sections=total_sections,
                user_answers=state.user_answers,
                previous_context=_previous_context(state),
            ) + f"""

                IMPORTANT: Generate ONLY exercise {exercise_num} of 3.
                Return exactly 2 cells: one markdown cell and one code cell.
                Keep the markdown description under 3 sentences.
                Keep the code cell under 15 lines.
                Valid closed JSON is mandatory — do not write more than this."""

            response = llm.invoke(prompt)  # llm normal es suficiente para 1 ejercicio
            cells = _safe_parse_cells(response.content)
            all_cells.extend(cells)

        return {
            "current_session_cells": state.current_session_cells + all_cells,
            "current_notebook_section": section_idx + 1,
        }

    # Resto de secciones: comportamiento original
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

    response = llm_large.invoke(prompt)
    new_cells = _safe_parse_cells(response.content)

    return {
        "current_session_cells": state.current_session_cells + new_cells,
        "current_notebook_section": section_idx + 1,
    }


def validate_code_node(state: CourseState) -> dict:
    """
    Executes all code cells in current_session_cells.
    Fixes failures using the LLM (up to max_validation_attempts per topic).
    """
    print(f"\n   Validating code cells (attempt {state.validation_attempts + 1}/"
          f"{state.max_validation_attempts})...")

    session = state.syllabus[state.current_session]
    topic = session.topics[state.current_topic]

    cells = state.current_session_cells
    results: list[CodeValidationResult] = []
    fixed_cells = list(cells)
    any_fixed = False

    for i, cell in enumerate(cells):
        if cell["cell_type"] != "code":
            continue

        # Skip exercise placeholder cells — they are intentionally empty
        source_str = "".join(cell["source"]).strip()
        is_placeholder = (
            not source_str
            or source_str.startswith("# Exercise")
            or source_str.startswith("# TODO")
            or source_str.startswith("# Your code")
            or all(line.strip().startswith("#") or not line.strip() for line in cell["source"])
        )
        if is_placeholder:
            results.append(CodeValidationResult(cell_index=i, success=True))
            print(f"      Cell {i}: skipped (exercise placeholder)")
            continue

        success, error = _execute_cells_cumulative(fixed_cells, i)

        if success:
            results.append(CodeValidationResult(cell_index=i, success=True))
            print(f"      Cell {i}: OK")
        else:
            print(f"      Cell {i}: FAILED — {error[:80]}...")

            if state.validation_attempts < state.max_validation_attempts:
                # Ask LLM to fix it
                fix_prompt = CODE_FIX_PROMPT.format(
                    course_topic=topic,
                    cell_index=i,
                    cell_source="".join(cell["source"]),
                    error_message=error,
                )
                fix_response = llm.invoke(fix_prompt)
                fixed_source = json.loads(_strip_json(fix_response.content))

                fixed_cells[i] = {**cell, "source": fixed_source}
                results.append(CodeValidationResult(
                    cell_index=i,
                    success=False,
                    error=error,
                    fixed_source=fixed_source,
                ))
                any_fixed = True
                print(f"      Cell {i}: fix applied by LLM")
            else:
                # Max attempts reached — keep as-is with a warning comment
                warning = [f"# WARNING: This cell failed validation. Error: {error[:100]}\n"]
                fixed_cells[i] = {**cell, "source": warning + cell["source"]}
                results.append(CodeValidationResult(
                    cell_index=i, success=False, error=error
                ))
                print(f"      Cell {i}: max attempts reached, added warning comment")

    return {
        "current_session_cells": fixed_cells,
        "validation_results": results,
        "validation_attempts": state.validation_attempts + 1,
    }


def advance_topic_node(state: CourseState) -> dict:
    """
    Called when all notebook sections for a topic are done.
    Advances to next topic or assembles the session notebook.
    """
    session = state.syllabus[state.current_session]
    next_topic = state.current_topic + 1

    if next_topic < len(session.topics):
        print(f"\n   Moving to topic {next_topic + 1} of session {session.session_number}")
        return {
            "current_topic": next_topic,
            "current_notebook_section": 0,
            "validation_attempts": 0,
            "validation_results": [],
        }

    # Session complete — assemble notebook
    print(f"\n Session {session.session_number} complete — assembling notebook...")

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
                **(
                    {"outputs": [], "execution_count": None}
                    if cell["cell_type"] == "code"
                    else {}
                ),
            }
            for cell in state.current_session_cells
        ],
    }

    notebooks = state.notebooks + [notebook_full]

    # Generate summary for context continuity
    summary_response = llm.invoke(SUMMARY_PROMPT.format(
        session_number=session.session_number,
        session_title=session.title,
        topics=", ".join(session.topics),
    ))
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
        "validation_attempts": 0,
        "validation_results": [],
    }


def save_outputs_node(state: CourseState) -> dict:
    """Saves all generated content to disk."""
    output_dir = Path("/opt/project/output") / state.topic.replace(" ", "_").lower()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n Saving course to: {output_dir}/")

    topic_count = 0
    for i, session in enumerate(state.syllabus):
        slug = (
            f"session_{session.session_number:02d}_"
            f"{session.title.replace(' ', '_').lower()}"
        )

        for j, topic in enumerate(session.topics):
            if topic_count < len(state.theory_docs):
                topic_slug = topic.replace(" ", "_").lower()[:40]
                theory_path = output_dir / f"{slug}_topic_{j+1:02d}_{topic_slug}.md"
                theory_path.write_text(state.theory_docs[topic_count], encoding="utf-8")
                print(f"   {theory_path.name}")
                topic_count += 1

        if i < len(state.notebooks):
            notebook_path = output_dir / f"{slug}.ipynb"
            notebook_path.write_text(
                json.dumps(state.notebooks[i], indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"   {notebook_path.name}")

    readme = f"# {state.topic} — Course\n\n"
    readme += f"**Student profile:**\n{state.user_answers}\n\n"
    readme += (
        f"**Total:** {state.total_hours}h | "
        f"{state.num_sessions} sessions x {state.session_hours}h\n\n"
    )
    for s in state.syllabus:
        readme += f"## Session {s.session_number}: {s.title}\n"
        for t in s.topics:
            readme += f"- {t}\n"
        readme += "\n"

    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(f"   README.md")
    print(f"\n Done! Course saved to: {output_dir}/")
    return {}


# ---------------------------------------------------------------------------
# Conditional edges
# ---------------------------------------------------------------------------

def route_after_interview(state: CourseState) -> str:
    return "planner" if state.interview_done else "interviewer"


def route_after_validation_syllabus(state: CourseState) -> str:
    return "theory_writer" if state.syllabus_approved else "planner"


def route_after_notebook_section(state: CourseState) -> str:
    if state.current_notebook_section < state.total_notebook_sections:
        return "notebook_section"
    if not state.validate_code:  # nuevo
        return "advance_topic"
    return "validate_code"


def route_after_code_validation(state: CourseState) -> str:
    """
    If there were failures AND we haven't hit max attempts, re-validate.
    Otherwise advance to next topic.
    """
    has_failures = any(not r.success for r in state.validation_results)
    can_retry = state.validation_attempts < state.max_validation_attempts

    if has_failures and can_retry:
        print("   Re-validating after fixes...")
        return "validate_code"
    return "advance_topic"


def route_after_advance(state: CourseState) -> str:
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
    graph.add_node("validate_code", validate_code_node)
    graph.add_node("advance_topic", advance_topic_node)
    graph.add_node("save_outputs", save_outputs_node)

    graph.add_edge(START, "interviewer")
    graph.add_edge("interviewer", "evaluate_interview")
    graph.add_conditional_edges("evaluate_interview", route_after_interview)
    graph.add_edge("planner", "validate_syllabus")
    graph.add_conditional_edges("validate_syllabus", route_after_validation_syllabus)
    graph.add_edge("theory_writer", "notebook_section")
    graph.add_conditional_edges("notebook_section", route_after_notebook_section)
    graph.add_conditional_edges("validate_code", route_after_code_validation)
    graph.add_conditional_edges("advance_topic", route_after_advance)
    graph.add_edge("save_outputs", END)

    return graph.compile(checkpointer=MemorySaver())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_course_generator(topic: str, total_hours: float, session_hours: float):
    num_sessions = int(total_hours / session_hours)
    app = build_graph()

    config = {
        "configurable": {"thread_id": f"{topic.replace(' ', '_').lower()}_001"},
        "metadata": {
            "topic": topic,
            "total_hours": total_hours,
            "session_hours": session_hours,
        },
        "tags": ["course-generator"],
    }

    initial_state = CourseState(
        topic=topic,
        total_hours=total_hours,
        session_hours=session_hours,
        num_sessions=num_sessions,
        validate_code=False,
    )

    print(f"\n Starting course generator for: '{topic}'")
    print(f"   {num_sessions} sessions x {session_hours}h = {total_hours}h total\n")

    input_data = initial_state

    while True:
        for event in app.stream(input_data, config=config, stream_mode="updates"):
            pass
        input_data = None  # reset: subsequent iterations resume from checkpoint

        snapshot = app.get_state(config)

        if not snapshot.next:
            break

        interrupt_data = (
            snapshot.tasks[0].interrupts[0].value if snapshot.tasks else {}
        )

        if "questions" in interrupt_data:
            user_input = input("\nYour answer: ").strip()
        elif "syllabus" in interrupt_data:
            user_input = input("\nYour response: ").strip()
        else:
            user_input = input("\nYour response: ").strip()

        input_data = Command(resume=user_input)


if __name__ == "__main__":
    run_course_generator(
        topic="Seaborn",
        total_hours=4,
        session_hours=2,
    )
