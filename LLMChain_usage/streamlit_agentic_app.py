import streamlit as st
from datetime import datetime
from srm_test_agentic import TataSteelAnalyzer

def main():
    # Set up the app
    st.set_page_config(page_title="Tata Steel Analyzer", page_icon="📊", layout="wide")

    # Initialize the analyzer
    @st.cache_resource
    def get_analyzer():
        return TataSteelAnalyzer()

    analyzer = get_analyzer()

    # App title and description
    st.title("Tata Steel Cost Analysis Assistant")
    st.markdown(
        """
        This application helps analyze cost data for Tata Steel locations. 
        Enter your query below and click 'Analyze' to get insights based on the provided data.
        """
    )

    # Create columns for better layout
    col1, col2 = st.columns([2, 1])

    with col1:
        # Query input section
        st.header("Query Input")
        with st.form("query_form"):
            user_query = st.text_area(
                "Enter your query (e.g., 'What is the cost variance in TSK for IM division in FY25?')",
                height=100,
                key="user_query"
            )
            
            submitted = st.form_submit_button("Analyze")

    with col2:
        # Example Queries Section
        st.header("Example Queries")
        example_queries = [
            "Overall summary of Tata Steel's Jamshedpur and Kalinganagar operations for FY25",
            "Explain the cost variance in TSK for IM division in FY25",
            "What is the total cost for TSM in July 2024?",
            "Compare the actual costs with planned costs for Iron Making in TSG up to the current month.",
            "What is the overall summary for Shared Services in FY25?",
            "How does the cost distribution compare between Iron Making and Steel Making in FY25?"
        ]
        # Define a function to update the query input
        def set_query(q):
            st.session_state.user_query = q

        # Create columns within col2 for better alignment of buttons
        button_col1, button_col2 = st.columns(2)

        # Create buttons with emojis for visual appeal
        for i, query in enumerate(example_queries):
            if i % 2 == 0:
                col = button_col1
            else:
                col = button_col2
            button_label = f"💡 {query[:30]}..."
            col.button(
                button_label,
                on_click=lambda x: set_query(x),
                args=(query,),
            )

    # Process the query if submitted
    if submitted and user_query.strip():
        with st.spinner("Analyzing your query..."):
            try:
                result = analyzer.analyze_query(user_query)
                st.success("Analysis completed!")
                
                # Display analysis result
                st.subheader("Analysis Result")
                st.markdown(result.get("analysis") or result.get("headlines", "No results found."))
                
                # # Display conversation history
                # st.subheader("Conversation History")
                # history = analyzer.get_analysis_history()
                # for entry in history:
                #     role = entry["role"].capitalize()
                #     content = entry["content"]
                #     st.write(f"**{role}**: {content}")
                #     st.write("---")
                
            except Exception as e:
                st.error(f"An error occurred during analysis: {str(e)}")
                st.write("Please check your query and try again.")
    else:
        st.warning("Please enter a query and click 'Analyze' to start.")

if __name__ == "__main__":
    main()