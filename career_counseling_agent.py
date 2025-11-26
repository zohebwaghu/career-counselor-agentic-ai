import streamlit as st
from langchain_community.llms import Ollama
from langchain_core.tools import Tool
from langchain_classic.agents import AgentExecutor, initialize_agent, AgentType
from langchain_classic.memory import ConversationBufferMemory
import re

def skills_gap_analyzer(input_str: str) -> str:
    """
    Analyze skills gap by comparing user_skills list to required skills.
    Input format: "user_skills: [skills], target_job: [job]"
    Return missing skills and suggest learning resources.
    """
    try:
        # Parse input - expect format like "user_skills: python, java, git, target_job: software engineer"
        parts = input_str.split("target_job:")
        if len(parts) != 2:
            return "Error: Please provide input in format 'user_skills: [skills], target_job: [job]'"
        
        user_skills_str = parts[0].replace("user_skills:", "").strip()
        target_job = parts[1].strip()
        
        # Simulated job skill requirements (would ideally fetch from API/DB)
        jobs_skills_db = {
            "software engineer": ["python", "data structures", "algorithms", "git", "cloud"],
            "data scientist": ["python", "statistics", "machine learning", "sql", "communication"],
            "product manager": ["communication", "agile", "roadmap", "stakeholder management", "analytics"]
        }
        user_skills_set = set(s.strip().lower() for s in re.split(r',|\n', user_skills_str))
        required_skills = set(jobs_skills_db.get(target_job.lower(), []))
        missing_skills = required_skills - user_skills_set
        if not required_skills:
            return f"Unknown job title '{target_job}'. Cannot analyze skills gap."
        suggestions = ", ".join(missing_skills) if missing_skills else "No significant gaps found."
        return f"Missing skills: {suggestions}. Suggested learning paths: Take online courses and practice projects on these areas."
    except Exception as e:
        return f"Error processing skills gap analysis: {str(e)}"

def resume_scorer(input_str: str) -> str:
    """
    Score resume by length, action verbs, and format hints.
    Input: resume text as string
    Return a score out of 10 and actionable feedback.
    """
    try:
        resume_text = input_str.strip()
        if not resume_text:
            return "Error: Please provide resume text to analyze."
        
        # Very rudimentary scoring for demo purposes
        length_score = min(len(resume_text) / 1000, 1)
        action_verbs = ["achieved", "managed", "led", "developed", "created", "improved", "implemented", "designed", "optimized"]
        action_verb_count = sum(resume_text.lower().count(verb) for verb in action_verbs)
        verb_score = min(action_verb_count / 5, 1)
        score = round((length_score + verb_score) / 2 * 10, 1)
        feedback = []
        if length_score < 0.5:
            feedback.append("Consider adding more detailed bullet points.")
        if verb_score < 0.5:
            feedback.append("Use more action verbs to showcase impact.")
        if len(resume_text) > 2000:
            feedback.append("Resume is quite long; make it more concise.")
        feedback_str = " ".join(feedback) if feedback else "Good resume content."
        return f"Resume score: {score}/10. Feedback: {feedback_str}"
    except Exception as e:
        return f"Error processing resume: {str(e)}"

def salary_estimator(input_str: str) -> str:
    """
    Provide estimated salary range based on job title, location, and experience.
    Input format: "job_title: [title], location: [location], years_experience: [years]"
    This is a mock implementation using fixed ranges.
    """
    try:
        # Parse input - expect format like "job_title: software engineer, location: san francisco, years_experience: 3"
        job_title = ""
        location = ""
        years_experience = "0"
        
        if "job_title:" in input_str:
            job_title = input_str.split("job_title:")[1].split(",")[0].strip()
        if "location:" in input_str:
            location = input_str.split("location:")[1].split(",")[0].strip()
        if "years_experience:" in input_str:
            years_experience = input_str.split("years_experience:")[1].strip()
        
        base_salaries = {
            "software engineer": 80000,
            "data scientist": 85000,
            "product manager": 90000
        }
        location_factors = {
            "san francisco": 1.3,
            "new york": 1.2,
            "boston": 1.1,
            "austin": 1.0,
            "remote": 0.9
        }
        base = base_salaries.get(job_title.lower(), 70000)
        location_multiplier = location_factors.get(location.lower(), 1.0)
        try:
            experience_multiplier = 1 + (float(years_experience) * 0.05)
        except ValueError:
            experience_multiplier = 1.0
        min_salary = int(base * location_multiplier * experience_multiplier * 0.85)
        max_salary = int(base * location_multiplier * experience_multiplier * 1.15)
        return f"Estimated salary range for {job_title} in {location} with {years_experience} years experience: ${min_salary:,} - ${max_salary:,}."
    except Exception as e:
        return f"Error processing salary estimation: {str(e)}"

def interview_question_generator(input_str: str) -> str:
    """
    Generate a list of interview questions based on the role and difficulty.
    Input format: "role: [role], difficulty: [easy/medium/hard]"
    """
    try:
        # Parse input - expect format like "role: software engineer, difficulty: medium"
        role = ""
        difficulty = "medium"
        
        if "role:" in input_str:
            role = input_str.split("role:")[1].split(",")[0].strip()
        if "difficulty:" in input_str:
            difficulty = input_str.split("difficulty:")[1].strip()
        
        questions_db = {
            "software engineer": {
                "easy": [
                    "Explain the difference between a list and a tuple in Python.",
                    "What is a function and how do you define one?",
                    "What is the difference between == and is in Python?"
                ],
                "medium": [
                    "How does a hash map work? Provide an example use case.",
                    "Describe the quicksort algorithm and its time complexity.",
                    "Explain the difference between a stack and a queue."
                ],
                "hard": [
                    "Design a scalable notification system.",
                    "Explain concurrency issues in multithreading and how to handle them.",
                    "How would you implement a distributed cache?"
                ]
            },
            "data scientist": {
                "easy": [
                    "What is the difference between supervised and unsupervised learning?",
                    "Explain the concept of p-value in statistics.",
                    "What is overfitting and how can you prevent it?"
                ],
                "medium": [
                    "How do you select features for a machine learning model?",
                    "Explain bias-variance tradeoff.",
                    "Describe cross-validation and why it's important."
                ],
                "hard": [
                    "Design a system to detect fraudulent transactions.",
                    "Discuss how to handle imbalanced datasets.",
                    "Explain how gradient boosting works."
                ]
            },
            "product manager": {
                "easy": [
                    "How do you prioritize features in a product roadmap?",
                    "Explain the difference between Agile and Waterfall methodologies.",
                    "What is a user story?"
                ],
                "medium": [
                    "Describe a time you had to handle conflicting stakeholder priorities.",
                    "How do you measure product success?",
                    "Explain the product development lifecycle."
                ],
                "hard": [
                    "Design a go-to-market strategy for a new product line.",
                    "Discuss ways to improve user retention.",
                    "How would you handle a product that's not meeting its KPIs?"
                ]
            }
        }
        role_lower = role.lower()
        difficulty_lower = difficulty.lower()
        questions = questions_db.get(role_lower, {}).get(difficulty_lower, [])
        if not questions:
            return f"No questions found for role '{role}' with difficulty '{difficulty}'. Available roles: software engineer, data scientist, product manager. Available difficulties: easy, medium, hard."
        return "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
    except Exception as e:
        return f"Error generating interview questions: {str(e)}"

def setup_tools():
    tools = [
        Tool(
            name="Skills_Gap_Analyzer",
            func=skills_gap_analyzer,
            description="Analyzes skills gap between user skills and target job requirements. Input format: 'user_skills: [comma-separated skills], target_job: [job title]'. Returns missing skills and learning suggestions."
        ),
        Tool(
            name="Resume_Scorer",
            func=resume_scorer,
            description="Evaluates resume content and provides actionable feedback with a score out of 10. Input: resume text as a string. Returns score and specific improvement suggestions."
        ),
        Tool(
            name="Salary_Estimator",
            func=salary_estimator,
            description="Provides realistic salary ranges based on job title, location, and years of experience. Input format: 'job_title: [title], location: [city], years_experience: [number]'. Returns estimated salary range."
        ),
        Tool(
            name="Interview_Question_Generator",
            func=interview_question_generator,
            description="Produces relevant technical and behavioral interview questions for different roles and difficulty levels. Input format: 'role: [job role], difficulty: [easy/medium/hard]'. Returns a list of interview questions."
        )
    ]
    return tools

def main():
    st.title("Career Counseling Agent")
    st.markdown("**A professional career counseling assistant powered by LangChain and Ollama**")

    # Initialize session state
    if "agent_executor" not in st.session_state:
        st.session_state.agent_executor = None
    if "generated" not in st.session_state:
        st.session_state["generated"] = []
    if "past" not in st.session_state:
        st.session_state["past"] = []

    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        model_name = st.text_input("Ollama Model Name:", value="llama3.2")
        
        if st.button("Initialize Agent"):
            try:
                with st.spinner("Initializing agent..."):
                    # Initialize Ollama LLM
                    llm = Ollama(model=model_name, temperature=0.7)
                    
                    # Setup tools
                    tools = setup_tools()
                    
                    # Create memory
                    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
                    
                    # Initialize agent with conversational-react-description
                    # This agent type supports conversation history and tool usage
                    st.session_state.agent_executor = initialize_agent(
                        tools=tools,
                        llm=llm,
                        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
                        memory=memory,
                        verbose=True,
                        handle_parsing_errors=True,
                        max_iterations=5
                    )
                    
                    st.success("Agent initialized successfully!")
            except Exception as e:
                st.error(f"Error initializing agent: {str(e)}")
                st.info("Make sure Ollama is running and the model is available. Run: ollama pull llama3.2")

    # Main chat interface
    if st.session_state.agent_executor is None:
        st.info("👈 Please initialize the agent from the sidebar first.")
        st.markdown("""
        ### Example Queries:
        - "Analyze my skills gap. I know Python, Java, and Git. I want to be a software engineer."
        - "Score my resume: I have 5 years of experience as a software developer. I led a team of 5 developers and improved system performance by 30%."
        - "What's the salary range for a data scientist in San Francisco with 3 years of experience?"
        - "Generate interview questions for a software engineer role with medium difficulty."
        """)
    else:
        # Display chat history
        if st.session_state["generated"]:
            for i in range(len(st.session_state["generated"]) - 1, -1, -1):
                with st.chat_message("user"):
                    st.write(st.session_state['past'][i])
                with st.chat_message("assistant"):
                    st.write(st.session_state['generated'][i])
        
        # User input
        user_input = st.chat_input("Ask a career-related question...")
        
        if user_input:
            try:
                with st.spinner("Thinking..."):
                    # Add user message to history
                    st.session_state.past.append(user_input)
                    
                    # Run agent
                    result = st.session_state.agent_executor.invoke({"input": user_input})
                    output = result.get("output", "I apologize, but I couldn't process that request.")
                    
                    # Add agent response to history
                    st.session_state.generated.append(output)
                    
                    # Rerun to display new messages
                    st.rerun()
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.session_state.past.append(user_input)
                st.session_state.generated.append(f"I encountered an error: {str(e)}")
                st.rerun()

if __name__ == "__main__":
    main()
