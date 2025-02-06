import streamlit as st
import plotly.express as px
import pandas as pd
import requests
from srm_test import TataSteelAnalyzer
from datetime import datetime

def initialize_analyzer():
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = TataSteelAnalyzer()

def get_visualization_data(query: str):
    if not query or query.isspace():
        st.warning("No query available for visualization")
        return []
    
    print(f"Executing visualization query: {query}")
    
    response = requests.post(
        "http://157.0.151.132:58872/srm/get_llm_data/",
        json={"query": query.strip()},
        headers={'Content-Type': 'application/json'}
    )
    
    if response.status_code == 200:
        return response.json()['data']
    else:
        st.error(f"Error fetching data: {response.status_code}")
        return []

def display_metrics(df):
    st.markdown("""
        <style>
        .metric-container {
            background-color: #262730;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .stMetric label {
            color: #FFFFFF !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Actual Cost", f"₹{df['ACTUAL'].sum():.2f}Cr")
        with col2:
            st.metric("Total Planned Cost", f"₹{df['PLAN'].sum():.2f}Cr")
        with col3:
            variance = df['ACTUAL'].sum() - df['PLAN'].sum()
            st.metric("Overall Variance", f"₹{variance:.2f}Cr",
                     delta=-variance if variance > 0 else variance)

def create_visualizations(df):
    with st.container():
        # Cost Trends - Bar Graph with Division Selector
        st.subheader("📈 Plan vs Actual Cost by Plant")
        
        # Prepare data for bar graph
        df_bar = df.groupby(['PLANT', 'DIVISION'])[['PLAN', 'ACTUAL']].sum().reset_index()
        
        # Create bar graph with division selector
        fig1 = px.bar(df_bar,
                     x="PLANT",
                     y=["PLAN", "ACTUAL"],
                     barmode='group',
                     color_discrete_sequence=['#2ecc71', '#e74c3c'],
                     title="Plan vs Actual Cost by Plant",
                     labels={"value": "Cost (Cr.)", "variable": "Cost Type"},
                     facet_col="DIVISION",
                     category_orders={"DIVISION": ["IM", "SM", "SS"]})
        
        # Update layout
        fig1.update_layout(
            updatemenus=[{
                'buttons': [
                    {'method': 'update',
                     'label': div,
                     'args': [{'visible': [True if d == div else False for d in df_bar['DIVISION'].unique()]}]}
                    for div in ['IM', 'SM', 'SS']
                ],
                'direction': 'down',
                'showactive': True,
                'x': 0.1,
                'y': 1.15
            }]
        )
        
        st.plotly_chart(fig1, use_container_width=True)
        
        # Rest of your visualizations
        col1, col2 = st.columns(2)
        with col1:
            fig2 = px.bar(df,
                         x="DIVISION",
                         y="ACTUAL",
                         color="LOCATION",
                         title="Division-wise Cost Distribution",
                         labels={"ACTUAL": "Cost (Cr.)"})
            st.plotly_chart(fig2, use_container_width=True)
        
        with col2:
            df['VARIANCE'] = df['ACTUAL'] - df['PLAN']
            fig3 = px.scatter(df,
                            x="MONTH",
                            y="VARIANCE",
                            size=df['ACTUAL'].abs(),
                            color="DIVISION",
                            title="Cost Variance Over Time",
                            labels={"VARIANCE": "Variance (Cr.)"})
            st.plotly_chart(fig3, use_container_width=True)

def display_analysis(result):
    st.markdown("### 📝 Analysis Results")
    if "analysis" in result:
        st.markdown(result["analysis"])
    elif "headlines" in result:
        st.markdown(result["headlines"])

def create_ui():
    st.title("🏭 Tata Steel Cost Analysis Dashboard")
    
    if 'current_query' not in st.session_state:
        st.session_state.current_query = ""
    
    st.sidebar.header("Display Options")
    show_viz = st.sidebar.toggle('Show Visualizations', value=True)
    
    query = st.text_input(
        "Enter your analysis query",
        placeholder="Example: What is the cost variance for IM division in TSJ for May 2024?"
    )
    
    if st.button("Analyze", type="primary"):
        if query and not query.isspace():
            with st.spinner("Processing your query..."):
                result = st.session_state.analyzer.analyze_query(query)
                st.session_state.current_query = st.session_state.analyzer.get_current_query()
                
                if show_viz and st.session_state.current_query:
                    viz_data = get_visualization_data(st.session_state.current_query)
                    if viz_data:
                        df = pd.DataFrame(viz_data)
                        # Filter out 'Total' rows
                        df = df[(df['PLANT'] != 'Total') & (df['PLAN'] != 0)]
                        
                        create_visualizations(df)
                        
                        # st.download_button(
                        #     label="📥 Download Analysis Data",
                        #     data=df.to_csv(index=False),
                        #     file_name=f"tata_steel_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        #     mime="text/csv"
                        # )
                
                display_analysis(result)

def main():
    st.set_page_config(
        page_title="Tata Steel Analysis",
        page_icon="🏭",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
        <style>
        .stButton>button {
            width: 100%;
        }
        .stMetric {
            background-color: #262730;
            padding: 10px;
            border-radius: 5px;
            color: #FFFFFF;
        }
        </style>
    """, unsafe_allow_html=True)
    
    initialize_analyzer()
    create_ui()

if __name__ == "__main__":
    main()
