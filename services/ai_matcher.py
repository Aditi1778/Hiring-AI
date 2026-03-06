import os
from datetime import datetime

from crewai import Agent, Crew, Process, Task
from crewai_tools import LlamaIndexTool
from langchain_google_genai import ChatGoogleGenerativeAI

from core.config import settings

# ----------------------------------------------------------
# 1. Gemini Setup (The Brain)
# ----------------------------------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", google_api_key=settings.GEMINI_API_KEY
)

# ----------------------------------------------------------
# 2. Agent 1 setup (The Data Architect - (Extraction Specialist))
# ----------------------------------------------------------

# Extract structured info from parsed indexes
# Responsibility: Navigates the Vector Index to build a logical profile.
data_architect = Agent(
    role="Expert Resume Data Architect",
    goal="Synthesize raw layout-aware document nodes into a clean, structured professional profile.",
    backstory="""You are an expert at decoding complex career histories.
        You utilize layout-aware data to identify core competencies, career progression,
        and technical depth. You do not just extract text; you extract professional meaning
        and context from structured document nodes.""",
    llm=llm,
    verbose=True,
    allow_delegation=False,
)


# ---------------------------------------------------------------------------
# 3. Agent 2 setup (The Strategic Matcher - (The Evaluator))
# ---------------------------------------------------------------------------

# Responsibility: Compares the synthesized profile against the specific Job Description.
strategic_matcher = Agent(
    role="Senior Technical Recruiter",
    goal="Critically evaluate the candidate profile against the Job Description to provide a Match Score.",
    backstory="""With 20 years of technical hiring experience, you look beyond simple keywords.
        You analyze how a candidate's specific projects, seniority levels, and tech stack
        align with the unique requirements and challenges of the role provided.""",
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

# -----------------------------------------------------
# 4. Binding agents with tasks ( pipeline setup )
# -----------------------------------------------------


def process_resume_matching(query_engine, jd_text: str):
    """
    Orchestrates the Recruitment Pipeline:
    Step 1: Data Architect extracts structured insights from the Vector Index.
    Step 2: Strategic Matcher evaluates those insights against the Job Description.
    """

    # 1. TOOL SETUP: Engine ko Tool mein convert karo taaki Agent ise "Action" ki tarah use kare
    resume_search_tool = LlamaIndexTool.from_query_engine(
        query_engine,
        name="Resume_Search_Tool",
        description="Mandatory tool to extract actual data from the uploaded resume. Use this for all resume-related queries.",
    )
    
    # 2. AGENT-TOOL CONNECTION: tell the architect to use this tool
    # as data architect is defined outside of the function 
    data_architect.tools = [resume_search_tool]
    
    
    # Task 1: Semantic Data Extraction
    # Uses the LlamaIndex Query Engine to perform deep retrieval.
    extraction_task = Task(
        description="""CRITICAL: Use the 'Resume_Search_Tool' to search and retrieve:
        - Using the provided query engine, retrieve and summarize:
        - Professional Timeline (Roles, Companies, and Durations).
        - Key Technical Skills (categorized by demonstrated proficiency).
        - Major Projects and their quantifiable impact.
        DO NOT provide example data or placeholders. If the tool returns no data, say 'No data found'.
        Ensure you leverage the layout-aware headers to maintain section context.""",
        expected_output="A structured and comprehensive professional summary of the candidate.",
        agent=data_architect,
    )

    # Task 2: Comparative Analysis & Scoring
    # Uses the output from Task 1 as context for final evaluation.
    matching_task = Task(
        description=f"""Based on the summary provided by the Data Architect,
        critically evaluate the candidate against this Job Description:
        ---
        {jd_text}
        ---
        Calculate a Match Score (0-100) and provide a detailed 'Reasoning'
        documenting strengths, potential risks, and missing technical gaps.
        OUTPUT MUST BE IN JSON FORMAT.""",
        expected_output="A JSON-formatted report containing 'match_score', 'reasoning', and 'gap_analysis'.",
        agent=strategic_matcher,
        context=[extraction_task],  # Passes Architect's output directly to the Matcher
    )

    # Crew Orchestration (Sequential Workflow)
    recruit_crew = Crew(
        agents=[data_architect, strategic_matcher],
        tasks=[extraction_task, matching_task],
        process=Process.sequential,  # Ensures Stage 1 completes before Stage 2 starts
        verbose=True,
    )

    # # Kickoff the agentic process
    # return recruit_crew.kickoff()

    # 1. Kickoff the process and store in a variable
    result = recruit_crew.kickoff()

    # 2. Extract raw string (CrewOutput se string nikalna)
    # result.raw mein wo JSON string hogi jo Agent ne di hai
    raw_output = str(result.raw)

    # 3. Data ko clean karo taaki MongoDB crash na kare
    # Hum results ko as a string bhejenge taaki serialization error na aaye
    return {
        "status": "completed",
        "results": raw_output,  # CrewAI ka output string format mein
        "analysis_date": datetime.utcnow().isoformat(),  # Date ko string bana diya
    }
