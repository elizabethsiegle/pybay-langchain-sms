import os

from langchain.llms import OpenAI, Cohere
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI(temperature=0.9) #model_name="text-davinci-003")
# name = llm("Suggest me a good name for a bike shop that is located near a beach!")
# print(name)

# llm_result = llm.generate(["Write a poem about hills", "Tell me a riddle about San Francisco"])
# print(llm_result.generations[0][0].text, llm_result.generations[1][0].text)

from langchain.prompts import PromptTemplate

# template = "Can you give me a name for a {type} shop with a {theme}?" 
# prompt = PromptTemplate(
#     template = template,
#     input_variables=["type", "theme"]
# )

# prompt = prompt.format(type="bike", theme="beach")
# print(prompt)

# from langchain.chains import LLMChain

# chain = LLMChain(llm=llm, prompt=prompt)

# print(chain.run({
#     'type': "bike",
#     'theme': "beach"
# }))

from langchain.agents import load_tools, initialize_agent, AgentType

tools = load_tools(
    ["llm-math"], llm=llm
)


a = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)
print(a.run("If I get 2 bikes and each bike costs $250, how much did I spend on bikes?"))