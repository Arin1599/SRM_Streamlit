# Import necessary tools to build our smart agents
from langchain.agents import initialize_agent, Tool
from langchain.memory import ConversationBufferMemory
from custom_llm import CustomOpenAILLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Create our Research Robot! 🔍
def create_research_agent():
    # Set up the AI brain with special research skills
    llm = CustomOpenAILLM(system_prompt="You are a specialized agent.")
    # Give the robot a memory to remember conversations
    memory = ConversationBufferMemory(memory_key="chat_history")
    
    # Create a template for how the robot should ask questions
    research_prompt = PromptTemplate(
        input_variables=["input"],
        template="You are a research expert. Find key information about: {input}"
    )
    
    # Connect the AI brain with the question template
    research_chain = LLMChain(llm=llm, prompt=research_prompt)
    
    # Give the robot its research tools
    tools = [
        Tool(
            name="Research",
            func=research_chain.run,
            description="Useful for finding information"
        )
    ]
    
    # Build the complete research robot with all its parts
    return initialize_agent(
        tools,
        llm,
        agent="conversational-react-description",
        memory=memory,
        verbose=True
    )

# Create our Analysis Robot! 🤔
def create_analysis_agent():
    # Set up the AI brain with special analysis skills
    llm = CustomOpenAILLM(system_prompt="You are a specialized agent.")
    # Give the robot a memory for conversations
    memory = ConversationBufferMemory(memory_key="chat_history")
    
    # Create a template for how to analyze information
    analysis_prompt = PromptTemplate(
        input_variables=["input"],
        template="You are an analysis expert. Analyze this information: {input}"
    )
    
    # Connect the AI brain with the analysis template
    analysis_chain = LLMChain(llm=llm, prompt=analysis_prompt)
    
    # Give the robot its analysis tools
    tools = [
        Tool(
            name="Analyze",
            func=analysis_chain.run,
            description="Useful for analyzing information"
        )
    ]
    
    # Build the complete analysis robot with all its parts
    return initialize_agent(
        tools,
        llm,
        agent="conversational-react-description",
        memory=memory,
        verbose=True
    )

# The Conductor that helps robots work together! 🎵
def orchestrate_agents(topic):
    # Create both robots
    research_agent = create_research_agent()
    analysis_agent = create_analysis_agent()
    
    # Ask Research Robot to find information
    research_results = research_agent.run(f"Research about {topic}")
    
    # Send the research to Analysis Robot for understanding
    final_analysis = analysis_agent.run(f"Analyze this research: {research_results}")
    
    # Package up both results nicely
    return {
        "research": research_results,
        "analysis": final_analysis
    }

# Start the program! 🚀
if __name__ == "__main__":
    # Ask our robots about renewable energy
    results = orchestrate_agents("renewable energy trends")
    # Show what both robots found
    print("Research Results:", results["research"])
    print("Analysis:", results["analysis"])
