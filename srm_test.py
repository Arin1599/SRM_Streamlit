from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from custom_llm import DualLLM
import requests
from datetime import datetime
import ast
from global_areas import get_location_context
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel
from typing import List, Dict, Any


class TataSteelAnalyzer:
    def __init__(self):
        self.llm = DualLLM(verbose=True)
        self.valid_divisions = ['IM', 'SM', 'SS']
        # Initialize all chains
        self.classification_chain = self.create_classification_chain()
        self.location_chain = self.create_location_chain()
        self.division_chain = self.create_division_chain()
        self.month_chain = self.create_month_chain()
        self.year_chain = self.create_year_chain()
        self.reasoning_chain = self.create_reasoning_chain()
        self.retrieval_chain = self.create_retrieval_chain()
        self.state = {} 
        self._current_query = ""

    def create_classification_chain(self):
        template = """you are an AI classifier agent.
        based on user query **{query}** 
        you classify it into following sections:
        -if the query asks to retrieve or obtain or provide the value or data or simple direct numerical information, 
        classify it as "Retrieval".
        -if the query asks to understand or explain or deduce or reason or summerize or insights 
        or anything related to reasoning something similar along the line, classify it as "Reasoning"
        if the query does not mean any of the above then , classify it as "Retrieval"
        if the query consists of gettings like hi, hello etc, classify it as 'greetings'
        do not write any code. just classify 
        result must be only classification in string."""
        prompt = PromptTemplate(template=template, input_variables=["query"])
        return  prompt | self.llm
    
    def create_location_chain(self):
        template = """You are an AI location classifier agent.

        Based on user query {query}, classify it into following locations:

        if the query contains something like TSJ or Jamshedpur or JSR followed by anything, classify it as "TSJ"

        if the query contains something like TSK or KPO or kallinganagar followed by anything, classify it as "TSK"

        if the query contains something like TSM or Angul or Meramandali followed by anything, classify it as "TSM"

        if the query contains something like TSG or ghamaria or TSLPL or Tata steel long products gahmaria followed by anything, classify it as "TSG"

        if the query contains something like "all plants" or "overall" or "TSL" or "Tata Steel" or no specific location is mentioned, return ['TSJ','TSK','TSM','TSG']

        if the query consists of greetings like hi, hello etc, classify it as 'greetings'

        Do not write any code. Just classify.
        Result must contain only list like ['TSJ'].
        If more than one location is mentioned, return all the locations in a list.
        Example: TSJ and TSK / TSJ or TSK
        response: ['TSJ','TSK']"""
        prompt = PromptTemplate(template=template, input_variables=["query"])
        
        return prompt | self.llm

    def create_division_chain(self):
        template = """You are an AI classifier agent which classifies plant areas.

        Based on user query "{query}", Determination of plant is already done as following : {classification}.

        You have to determine the areas weather it is IM,SM 
                or SS or combination of these areas
                Classify into these areas only:
                - IM (Iron Making)
                - SM (Steel Making) 
                - SS (Shared Services)
                
                Return only the area code (IM/SM/SS) without any location prefix.
                If no specific area matches or overall is mentioned or all the plants/area is mentioned
                without mentioning any specific area then
                return ['IM','SM','SS'].
                If there are more than one areas mentioned in the query then return a single list containing all the area codes
                Example response:
                ['IM','SM','SS']
                Respond with just the list of area code - no other text."""
        prompt = PromptTemplate(template=template, input_variables=["query", "classification"])
        return prompt | self.llm

    def create_month_chain(self):
        system_prompt = f"""
        You are an AI-based month extractor and formatter agent.
        
        Financial Year (FY) Rules:
        - FY YY covers from April YY-1 to March YY
        - When FY YY is mentioned, return ALL months from April (4) to March (3)
        
        Results must be a list of integers representing months:
        [4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3]

        For single month mentions:
        - May -> [5]
        - January -> [1]
        
        For multiple months:
        - May, June, July -> [5, 6, 7]
        - May and June -> [5, 6]
        - May to September -> [5, 6, 7, 8, 9]
        
        The only output should be a list of integers nothing else.
        """
        template = """Based on user query **{query}**, extract the month numbers:
        
        For single month mentions:
        - May -> [5]
        - January -> [1]
        
        For multiple months:
        - May, June, July -> [5, 6, 7]
        - May and June -> [5, 6]
        - May to September -> [5, 6, 7, 8, 9]

        
        For FY mentions:
        - FY25 -> [4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3]
        
        If no month specified, default to current month
        
        For yearly/annual mentions:
        - If no month is mentioned bu just FY is mentioned return all months of FY as above example
        - Return all months of specified FY
        - If till date mentioned: return months from April to current month
        - If range mentioned (e.g., May to September): return all months in range

        **NOTE:** Only return a list of integers representing months in final response nothing else."""
        prompt = PromptTemplate(template=template, input_variables=["query"])
        return prompt | self.llm


    def get_current_month_and_year(self):
        current_date = datetime.now()
        return f"[{current_date.month}]"

    def create_year_chain(self):
        template = """Extract financial years from: {query}
        Return only a Python list of years: [2024] or [2025]
        If no year found, return: [{default_year}]
        **NOTE** Give only a Python list of years: [2024] or [2025] or [2024,2025] nothing else"""
        prompt = PromptTemplate(template=template, input_variables=["query", "default_year"])
        return prompt | self.llm

    

    def analyze_query(self, query: str):
        # Always start with default model
        self.llm.use_alternate_model = False
        
        print("\n=== Starting Query Analysis ===")
        print(f"Input Query: {query}")

        # Direct string handling for classification
        query_type = str(self.classification_chain.invoke({"query": query}))
        if 'Reasoning' in query_type:
            query_type = 'Reasoning'
        elif 'greetings' in query_type.lower():
            query_type = 'greetings'
        else:
            query_type = 'Retrieval'

        print(f"Query Type: {query_type}")

        # Direct string handling for location
        location_response = str(self.location_chain.invoke({"query": query}))
        if '[' in location_response and ']' in location_response:
            try:
                locations = ast.literal_eval(location_response)
            except:
                locations = ['TSJ','TSK','TSM','TSG']
        else:
            locations = ['TSJ','TSK','TSM','TSG']
        
        division_response = self.division_chain.invoke({
            "query": query,
            "classification": locations[0] if locations else "TSJ"
        })
        divisions = ast.literal_eval(division_response.strip().replace('```python', '').replace('```', ''))
        
        month_response = self.month_chain.invoke({"query": query})
        months = ast.literal_eval(month_response.strip().replace('```python', '').replace('```', ''))
        
        current_fy = datetime.now().year + 1 if datetime.now().month >= 4 else datetime.now().year
        year_response = self.year_chain.invoke({
            "query": query,
            "default_year": current_fy
        })
        years = ast.literal_eval(year_response.strip().replace('```python', '').replace('```', ''))

        # Get data from backend
        data = self._get_bigquery_data(
            divisions=divisions if divisions else self.valid_divisions,
            time_info={
                "months": months,
                "fin_year": years
            },
            locations=locations
        )

        self.state = {
            'query': query,
            'data': data,
            'locations': locations[0] if locations else "TSJ",
            'plant_context': get_location_context(locations[0] if locations else "TSJ")
        }

        # Switch model only for final reasoning analysis
        if query_type.lower() == 'reasoning':
            print("\n=== Executing Final Reasoning Analysis ===")
            # Create a new chain specifically for final reasoning with alternate model
            final_reasoning_chain = self.create_final_reasoning_chain()
            response = final_reasoning_chain.invoke(self.state)
            return {"analysis": response, "data": data}
        else:
            print("\n=== Executing Retrieval Chain ===")
            response = self.retrieval_chain.invoke(self.state)
            return {"headlines": response}

    def create_final_reasoning_chain(self):
        # Temporarily switch to alternate model for final reasoning
        self.llm.use_alternate_model = True
        
        # Create the chain with the same template
        chain = self.create_reasoning_chain()
        
        # Reset back to default model
        self.llm.use_alternate_model = False
        
        return chain





    def _get_bigquery_data(self, divisions, time_info, locations):
        self._current_query = f"""
        SELECT * FROM BPP.T_TEST_SRM_COST
        WHERE LOCATION IN ({','.join([f"'{location}'" for location in locations])})
        AND DIVISION in ({','.join([f"'{div}'" for div in divisions])})
        AND MONTH in ({','.join(map(str, time_info['months']))})
        AND FIN_YEAR in ({','.join(map(str, time_info['fin_year']))})
        """
        
        response = requests.post(
            "http://157.0.151.132:58872/srm/get_llm_data_json",
            json={"query": self._current_query},
            headers={'Content-Type': 'application/json'}
        )
        return response.json() if response.status_code == 200 else []

    def get_current_query(self):
        return self._current_query

    def create_reasoning_chain(self):
        
        
        reasoning_template = """
        You are a Financial Analyst analyzing cost data for Tata Steel Limited locations. Reference the provided data and follow these steps:
        
        Plant Structure Context(sublocations):
        {plant_context}
        
        - **Step 1: Location-Specific Context**
        - Analyze data for {locations} focusing on its major operational areas:
            - Iron Making (IM)
            - Steel Making (SM)
            - Shared Services (SS)

        - **Step 2: Response Precision**
        - Ensure all insights are specific to {locations}
        - Maintain focus on the queried operational areas
        - If data isn't available, clearly state the limitations

        - **Step 3: Data Analysis Framework**
        - Support all insights with numerical evidence
        - Clearly state percentage comparisons and their basis
        - For time-series analysis:
            - Exclude zero values from calculations
            - Clearly identify excluded months with zero values
            - Consider both Plan and Actual values
            - Calculate trends based on non-zero values only

        - **Step 4: Cost Interpretation**
        - Negative cost differences indicate Actual > Plan
        - Lower costs are favorable
        - Higher consumption is unfavorable
        - Cost units are in Crores (Cr.)

        - **Step 5: Rate vs Cost Distinction**
        - Differentiate between rate-based and cost-based metrics
        - Provide appropriate metric based on query specification
        - Maintain clarity in unit representation

        Question: {query}
        Data Context: {data}
        
        Provide a detailed analysis with:
        - Bullet-pointed insights
        - Numerical evidence
        - Clear trends and patterns
        - Specific recommendations where applicable
        """
        
        prompt = PromptTemplate(
            template=reasoning_template,
            input_variables=["plant_context", "locations", "query", "data"]
        )
        analysis =  prompt | self.llm

        return analysis

    def create_retrieval_chain(self):
        
        retrieval_template  = """
        You are a data retrieval expert. Based on the following data and question, provide specific numerical information:
        Just retrieve the data and display results of what is asked. No extra informations or suggestions is required

        Question: {query}
        Data: {data}
        Plant Structure Context(sublocations):
        {plant_context}

        Focus on:
        - Exact values and metrics
        - Simple calculations
        - Direct comparisons
        - Specific data points requested

        **NOTE** Don' give any other informations/suggestions. Just give factual information with exact data. Don't use the 
        keyword project as we are dealing with TATA STEEL PLANT. Also while dealing with any calculation don't mention those 
        months where Actual is 0. Strictly mention the exclusion of those months in response. The unit of values is in Crores (Cr.)
        """
        prompt = PromptTemplate(
            template=retrieval_template,
            input_variables=["query", "data", "plant_context"]
        )
        
        return prompt | self.llm
    
    def handle_reasoning(self, state):
        return {
            "analysis": self.reasoning_chain.invoke({
                "plant_context": get_location_context(state['locations']),
                "locations": state['locations'],  # Changed from 'location' to 'locations'
                "query": state['query'],
                "data": state['data']
            })
        }
    def handle_retrieval(self, state):
        print("\n=== Retrieval Process Started ===")
        print(f"Processing Query: {state['query']}")
        
        response = self.retrieval_chain.invoke({
            "query": state['query'],
            "data": state['data'],
            "plant_context": get_location_context(state['locations'])
        })
        
        print("\n=== Retrieved Data ===")
        print(response)
        
        return {"headlines": response}


if __name__ == "__main__":
    analyzer = TataSteelAnalyzer()
    query = "Give  me planned Actual cost of I blast furncace TSJ in August"
    result = analyzer.analyze_query(query)
