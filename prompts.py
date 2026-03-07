PLANNER_PROMPT = """
You are an expert course designer. Given a topic and time constraints, create a structured syllabus.

Course details:
- Topic: {topic}
- Total hours: {total_hours}
- Session duration: {session_hours} hours
- Number of sessions: {num_sessions}

Return ONLY a valid JSON array with this exact structure (no markdown, no explanation):
[
  {{
    "session_number": 1,
    "title": "Session title",
    "topics": ["topic 1", "topic 2", "topic 3"],
    "duration_hours": {session_hours}
  }},
  ...
]

Rules:
- Cover the subject progressively from basics to advanced
- Each session should have 3-5 specific topics
- Topics must be practical and hands-on oriented
- Ensure logical progression between sessions
"""


THEORY_PROMPT = """
You are an expert technical writer and educator. Write detailed theory content for a course session.

Course topic: {course_topic}
Session {session_number}: {session_title}
Topics to cover: {topics}
Session duration: {session_hours} hours

Write comprehensive theory content in Markdown format. Include:
1. A brief introduction to the session
2. Detailed explanation of each topic with examples
3. Key concepts highlighted in bold
4. Code snippets where relevant (use ```python blocks)
5. A summary section at the end
6. 2-3 suggested exercises

The content should be appropriate for {session_hours} hours of learning.
Be thorough, practical, and use clear explanations with real-world examples.
"""


NOTEBOOK_PROMPT = """
You are an expert Python developer and educator. Create a Jupyter notebook for a course session.

Course topic: {course_topic}
Session {session_number}: {session_title}
Topics to cover: {topics}
Session duration: {session_hours} hours

Return ONLY a valid JSON object representing a Jupyter notebook with this exact structure:
{{
  "cells": [
    {{
      "cell_type": "markdown",
      "source": ["# Session {session_number}: {session_title}\\n", "Brief intro..."]
    }},
    {{
      "cell_type": "code",
      "source": ["# Import libraries\\n", "import ..."]
    }},
    ...
  ]
}}

Rules:
- Start with a markdown cell with the session title and objectives
- Include an imports cell
- For each topic, add a markdown explanation cell followed by code cell(s)
- Use realistic, runnable Python code with {course_topic}
- Add comments in code cells explaining what each block does
- End with a "Practice exercises" markdown cell and empty code cells for students
- Use common datasets (seaborn built-in, sklearn datasets, etc.) — no external downloads needed
- All code must be self-contained and executable in order
"""
