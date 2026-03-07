from pydantic import BaseModel, Field


class SessionPlan(BaseModel):
    session_number: int
    title: str
    topics: list[str]
    duration_hours: float


class CourseState(BaseModel):
    # --- Input ---
    topic: str
    total_hours: float
    session_hours: float

    # --- Computed ---
    num_sessions: int = 0

    # --- Planner output ---
    syllabus: list[SessionPlan] = Field(default_factory=list)

    # --- Progress ---
    current_session: int = 0

    # --- Generated content ---
    theory_docs: list[str] = Field(default_factory=list)
    notebooks: list[dict] = Field(default_factory=list)
