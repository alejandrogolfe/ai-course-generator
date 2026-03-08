from pydantic import BaseModel, Field


class SessionPlan(BaseModel):
    session_number: int
    title: str
    topics: list[str]
    duration_hours: float


class CodeValidationResult(BaseModel):
    cell_index: int
    success: bool
    error: str = ""
    fixed_source: list[str] = Field(default_factory=list)


class CourseState(BaseModel):
    # --- Input básico ---
    topic: str
    total_hours: float
    session_hours: float
    num_sessions: int = 0

    # --- Entrevista ---
    interview_messages: list[dict] = Field(default_factory=list)
    user_answers: str = ""
    interview_rounds: int = 0
    interview_done: bool = False

    # --- Planificación ---
    syllabus: list[SessionPlan] = Field(default_factory=list)
    syllabus_feedback: str = ""
    syllabus_approved: bool = False
    planner_rounds: int = 0

    # --- Progreso granular ---
    current_session: int = 0
    current_topic: int = 0
    current_notebook_section: int = 0
    # Calculado dinámicamente en theory_writer según nº de subtopics del tema:
    # 1 intro + N bloques de desarrollo (1 por subtopic) + 1 ejercicios
    total_notebook_sections: int = 3

    # --- Contexto acumulativo entre sesiones ---
    completed_summaries: list[str] = Field(default_factory=list)

    # --- Output ---
    theory_docs: list[str] = Field(default_factory=list)
    notebooks: list[dict] = Field(default_factory=list)
    current_session_cells: list[dict] = Field(default_factory=list)

    # --- Validación de código ---
    validation_results: list[CodeValidationResult] = Field(default_factory=list)
    validation_attempts: int = 0
    max_validation_attempts: int = 2
