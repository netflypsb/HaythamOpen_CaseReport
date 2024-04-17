import streamlit as st
from langchain_openai import ChatOpenAI
from crewai import Agent, Task, Crew, Process
from langchain_community.tools.pubmed.tool import PubmedQueryRun
from langchain_community.tools.semanticscholar.tool import SemanticScholarQueryRun
import tempfile

class ReportWriterCrew:
    def __init__(self, api_key, model_choice):
        self.api_key = api_key
        self.model_choice = model_choice
        self.base_url = self.determine_base_url(model_choice)

    def determine_base_url(self, model_choice):
        openrouter_models = ["databricks/dbrx-instruct:nitro", "mistralai/mixtral-8x7b-instruct:nitro", 
                             "meta-llama/llama-2-70b-chat:nitro", "perplexity/pplx-70b-chat"]
        return "https://openrouter.ai/api/v1" if model_choice in openrouter_models else "https://api.openai.com/v1"

    def setup_llm_and_agents(self):
        llm = ChatOpenAI(model=self.model_choice, api_key=self.api_key, base_url=self.base_url)

        # Initialize tools
        pubmed_query = PubmedQueryRun()
        semantic_scholar_query = SemanticScholarQueryRun()

        # Define Agents
        self.report_agent = Agent(role='Medical Report Rewriter', 
                                  goal='To refine the case report for clarity, ensuring it meets publication standards.',
                                  backstory='Experienced in medical writing and editing, this agent has a keen eye for detail and a deep understanding of medical terminology, ensuring reports are precise and understandable.',
                                  tools=[pubmed_query, semantic_scholar_query],
                                  max_iter=20,
                                  verbose=True,
                                  allow_delegation=False,
                                  llm=llm)
        self.summary_agent = Agent(role='Summary Generator',
                                   goal='To condense the case report into a succinct summary highlighting key findings.',
                                   backstory='Skilled at distilling complex information into concise summaries, this agent helps capture the essence of medical reports for quick understanding.',
                                   tools=[pubmed_query, semantic_scholar_query],
                                   max_iter=20,
                                   verbose=True,
                                   allow_delegation=False,
                                   llm=llm)
        self.background_agent = Agent(role='Background Generator',
                                      goal='To draft a detailed background section that provides context for the case report.',
                                      backstory='With expertise in medical research, this agent crafts comprehensive backgrounds that set the stage for each case, grounding them in current scientific knowledge.',
                                      tools=[pubmed_query, semantic_scholar_query],
                                      max_iter=20, 
                                      verbose=True, 
                                      allow_delegation=False,
                                      llm=llm)
        self.discussion_agent = Agent(role='Discussion Generator',
                                      goal='To create a discussion that interprets the case findings and suggests future research directions.',
                                      backstory='Analytical and insightful, this agent excels at evaluating case outcomes and weaving them into the broader scientific discourse.',
                                      tools=[pubmed_query, semantic_scholar_query],
                                      max_iter=20,
                                      verbose=True, 
                                      allow_delegation=False,
                                      llm=llm)
        self.learning_agent = Agent(role='Learning Points Generator',
                                    goal='To highlight key learning points from the case, offering actionable insights for practitioners.',
                                    backstory='Focused on educational value, this agent identifies and elucidates critical learnings, enhancing the practical utility of the case report.',
                                    tools=[pubmed_query, semantic_scholar_query],
                                    max_iter=20, 
                                    verbose=True, 
                                    allow_delegation=False,
                                    llm=llm)

    def generate_case_report(self, case_summary):
        if not case_summary:
            st.warning("Please provide a case summary.")
            return None

        self.setup_llm_and_agents()

        # Define Tasks (simplified for brevity)
        report_task = Task(description=f'Write the CASE REPORT section by rewriting {case_summary} using proper case report structure and correcting grammatical errors.',
                           agent=self.report_agent,
                           expected_output='A medically accurate, clear, and well-structured case report ready for publication.')

        summary_task = Task(description='Write the SUMMARY section by summarising the case report written by the report agent, highlighting the key points of the case in less than 300 words.',
                            agent=self.summary_agent,
                            expected_output='A summary of the case report, highlighting clinically important symptoms, signs, and investigation findings and their implications.',
                            context=[report_task])

        background_task = Task(description='Write the BACKGROUND section for the case report, covering Introduction to the condition, Existing Literatures and Similar Cases, Rationale for the report and Objectives.',
                               agent=self.background_agent,
                               expected_output='A detailed background section that provides a thorough context for the case, referencing relevant studies and literature.',
                               context=[report_task])

        discussion_task = Task(description='Write the DISCUSSION section, interpreting the findings and integrating them with existing knowledge.',
                               agent=self.discussion_agent,
                               expected_output='An insightful discussion that offers interpretations of the findings, compares them with existing literature, and suggests areas for future research.',
                               context=[summary_task, background_task])
                                 
        learning_task = Task(description='Write the LEARNING POINTS section, identifying and articulating the key learning points from the case, emphasizing their relevance to medical practice.',
                             agent=self.learning_agent,
                             expected_output='Clearly defined learning points that distill the case’s educational value.',
                             context=[discussion_task])
        
        case_report_writer_crew = Crew(agents=[report_agent, summary_agent, background_agent, discussion_agent, learning_agent], 
                                       tasks=[report_task, summary_task, background_task, discussion_task, learning_task], 
                                       verbose=True, 
                                       process=Process.sequential,
                                       )
        case_report_writer_crew.kickoff({"case_summary": case_summary})

        # Handle output with temporary file storage
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w") as tmpfile:
            file_path = tmpfile.name
            tmpfile.write(f"Case Summary: {case_summary}\n\n")
            tmpfile.write("Case Report:\n" + report_task.output.raw_output + "\n\n")
            tmpfile.write("Summary:\n" + summary_task.output.raw_output + "\n\n")
            tmpfile.write("Background:\n" + background_task.output.raw_output + "\n\n")
            tmpfile.write("Discussion:\n" + discussion_task.output.raw_output + "\n")
            tmpfile.write("Learning Points:\n" + learning_task.output.raw_output + "\n")

        return file_path

# Streamlit UI for interacting with the ReportWriterCrew
st.set_page_config(page_title="Case Report", page_icon="📄")
api_key = st.text_input("Enter your OpenRouter API Key", type="password")
model_choice = st.selectbox("Choose the LLM model to use:", ["databricks/dbrx-instruct:nitro", "mistralai/mixtral-8x7b-instruct:nitro", "meta-llama/llama-2-70b-chat:nitro", "perplexity/pplx-70b-chat"]) # Model options

case_summary = st.text_area("📋 Case Summary", height=150, placeholder="✍️ Enter Your Case Summary Here")

if st.button("Generate Report 🚀"):
    report_writer = ReportWriterCrew(api_key, model_choice)
    file_path = report_writer.generate_case_report(case_summary)
    if file_path:
        st.success("Report generated successfully! 🎉")
        with open(file_path, "rb") as file:
            st.download_button(label="Download Case Report 📥", data=file, file_name="case_report.txt", mime="text/plain")
          
