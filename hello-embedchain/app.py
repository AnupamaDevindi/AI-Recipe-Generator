from dotenv import load_dotenv
import os
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

load_dotenv()
API_KEY = os.environ['OPENAI_API_KEY']

llm = OpenAI(openai_api_key =  API_KEY, temperature = 0.9)

prompt_template = PromptTemplate(
    template = "Give me an example of a meal could be made using the following ingredeints : {ingredients}",
    input_variables= ['ingredients']
)

gangster_template = """Re-Write the meas given below in the style of a new york mafia 
gangster:

Meals:
{meals}
"""

gangster_template_prompt = PromptTemplate(
    template = gangster_template,
    input_variables = ['meals']
)

meal_chain = LLMChain(
    llm = llm,
    prompt = prompt_template,
    output_key = "meals",
    verbose = True
)

gangster_chain = LLMChain(
    llm = llm,
    prompt = gangster_template_prompt,
    output_key = "gangster meals",
    verbose = True
)

overall_chain = SequentialChain(
    chains = [meal_chain,gangster_chain],
    input_variables = ['ingredients'],
    output_variables = ["meals", "gangster meals"]
    )

st.title("Meal Planner")
user_prompt = st.text_input("Enter a comma seperated ingredeints")

if st.button("Generate") and user_prompt:
    with st.spinner("Generating...."):
        output = overall_chain({'ingredients': user_prompt})
        
        col1,col2 =st.columns(2)
        col1.write(output['meals'])
        col2.write(output['gangster meals'])
