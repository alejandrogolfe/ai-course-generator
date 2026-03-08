from pydantic import BaseModel, Field


class SessionPlan(BaseModel):
    session_number: int
    title: str
    topics: list[str]
    duration_hours: float


class CourseState(BaseModel):
    # --- Input básico ---
    topic: str
    total_hours: float
    session_hours: float
    num_sessions: int = 0

    # --- Entrevista ---
    interview_messages: list[dict] = Field(default_factory=list)  # historial LLM
    user_answers: str = ""        # respuestas acumuladas del usuario
    interview_rounds: int = 0     # nº de rondas de preguntas
    interview_done: bool = False

    # --- Planificación ---
    syllabus: list[SessionPlan] = Field(default_factory=list)
    syllabus_feedback: str = ""   # texto libre del usuario al revisar
    syllabus_approved: bool = False
    planner_rounds: int = 0       # nº de revisiones del syllabus

    # --- Progreso granular ---
    current_session: int = 0
    current_topic: int = 0
    current_notebook_section: int = 0
    total_notebook_sections: int = 3  # secciones por tema: intro, desarrollo, ejercicios

    # --- Contexto acumulativo entre sesiones ---
    completed_summaries: list[str] = Field(default_factory=list)

    # --- Output ---
    theory_docs: list[str] = Field(default_factory=list)
    notebooks: list[dict] = Field(default_factory=list)
    current_session_cells: list[dict] = Field(default_factory=list)  # se ensambla sección a sección
