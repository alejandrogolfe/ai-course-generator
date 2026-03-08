INTERVIEWER_PROMPT = """
You are an expert course designer about to create a course on: {topic}

The course will be {total_hours} hours total, split into sessions of {session_hours} hours each.

So far you know this about the student's needs:
{user_answers}

Your job is to ask targeted questions to understand:
- The student's current level (beginner / intermediate / advanced)
- Their background and prerequisite knowledge
- Whether they prefer a theoretical or hands-on approach
- Their goal (learn for work, personal project, academic, etc.)
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
You are an expert course designer. Design a structured syllabus based on this context:

Topic: {topic}
Total hours: {total_hours}h | Session duration: {session_hours}h | Sessions: {num_sessions}

Student profile and requirements:
{user_answers}

{feedback_section}

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
- Tailor difficulty and depth to the student profile
- Each session must have 3-5 specific, concrete topics
- Topics must progress logically within and across sessions
- Balance theory and practice based on student preference
- First session should establish foundations, last session should consolidate
"""

THEORY_PROMPT = """
You are an expert technical writer creating course content.

Course: {course_topic}
Student profile: {user_answers}

Previous sessions covered:
{previous_context}

Now write theory for:
Session {session_number}: {session_title}
Current topic: "{current_topic}" (topic {topic_index} of {total_topics})

Write detailed Markdown content for THIS TOPIC ONLY. Include:
1. Clear explanation of the concept
2. Real-world motivation / why it matters
3. Key points in **bold**
4. Code snippets where relevant (```python blocks)
5. A concrete example with explanation

Do NOT write about other topics in this session.
Do NOT repeat content from previous sessions.
Assume the student already knows: {previous_context}
"""

NOTEBOOK_SECTION_PROMPT = """
You are creating part of a Jupyter notebook for a course on {course_topic}.

Session {session_number}: {session_title}
Topic: "{current_topic}"
Section: {section_name} ({section_index} of {total_sections})

Student profile: {user_answers}
Previous sessions covered: {previous_context}

Generate ONLY the cells for the "{section_name}" section of this topic.

Section definitions:
- "imports_and_intro": A markdown cell with topic title + objectives, then a code cell with all necessary imports and any setup (dataset loading, config). This section only appears once per topic.
- "explanation_and_examples": A markdown cell explaining a concept clearly, followed by code cell(s) demonstrating it with comments. Focus on ONE concept per block.  
- "exercises": A markdown cell describing 2-3 practice exercises, followed by empty code cells with comment placeholders for students to fill in.

Return ONLY a valid JSON array of cells (no markdown fences, no explanation):
[
  {{
    "cell_type": "markdown",
    "source": ["## Topic title\\n", "Content here..."]
  }},
  {{
    "cell_type": "code",
    "source": ["# comment\\n", "import seaborn as sns\\n"]
  }}
]

Rules:
- All code must be self-contained and runnable
- Use only built-in datasets (seaborn.load_dataset, sklearn.datasets, etc.)
- Add inline comments explaining what each code block does
- Build on previous sessions' knowledge — don't re-explain basics already covered
"""

SUMMARY_PROMPT = """
Write a brief summary (3-5 sentences) of what was covered in this session for reference in future sessions.

Session {session_number}: {session_title}
Topics covered: {topics}

Be specific about the concepts, functions, and techniques taught.
This will be used as context for future sessions to avoid repetition.
"""
