# ALL IMPORT STATEMENT
import streamlit as st
import os
from langchain import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain



# SET YOUR HUGGINGFACEHUB_API_TOKEN HERE
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "YOUR_HUGGINGFACEHUB_API_TOKEN"



# DEFINE LLM MODEL
llm = HuggingFaceHub( repo_id="google/gemma-1.1-7b-it",
                          model_kwargs={"temperature":0.3, "max_length" : 1024} 
                          )
    


# CREAT A ONE PROMPT TEMPLATE
template = PromptTemplate(
    input_variables=['task', 'language'],
    template="Write a code that {task} in {language} programming language"

)



# CREAT A CHAIN 
chain = LLMChain(llm=llm, prompt=template)


# FUNCTION TO GET LLM RESPONSE
def get_llm_response(task, language):
    response = chain.run({'task': task, 'language': language})
    # response = response[:]
    return response




# SET SOME PAGE CONFIG OF STREAMLIT
st.set_page_config(page_title="AI Code Generator")
st.header("AI Code Generator")



# SET INPUT BOX AND SUBMIT BUTTON
st.markdown("### Write a task : ")
task = st.text_input("",key="input")
st.markdown("### Select Language : ")
language = st.radio("", ['Python', 'JavaScript', 'java', 'kotlin', 'SQL'])
st.subheader('')
submit=st.button('Generate Code')





# GETTING RESPONSE WHEN PRESSING SUBMIT BUTTON
if submit:
    try:
        response = get_llm_response(task, language)
        st.subheader("Response: ")
        print(language)
        st.write(response)
    except Exception as e:
        print(e)