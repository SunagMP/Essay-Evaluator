from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END

import json
import operator
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.5-pro')

# creating the ouput evaluation schema

from pydantic import BaseModel, Field
from typing import TypedDict , Annotated, List

class outputSchema(BaseModel):
    feedback : Annotated[str, Field(description="Feedback of the written essay.")]
    score : Annotated[int, Field(description="The score for the essay out of 10.", gt=0, le=10)]

pyparser = PydanticOutputParser(pydantic_object=outputSchema)
parser = StrOutputParser()

essay = """
**The Green Revolution: A Turning Point in Agricultural History**

The Green Revolution marked a significant milestone in the global fight against hunger and poverty. Emerging during the mid-20th century, it introduced a series of agricultural innovations that dramatically increased food production, particularly in developing countries like India, Mexico, and the Philippines. This movement was not merely a technological transformation but a socio-economic shift that changed the way the world approached agriculture, food security, and rural development.

At the heart of the Green Revolution were high-yielding varieties (HYVs) of crops, especially wheat and rice, developed by agricultural scientists such as Norman Borlaug, who is often credited as the "father of the Green Revolution." These varieties were more resistant to pests and diseases and had the potential to produce much higher yields than traditional seeds. Alongside HYVs, the revolution emphasized the use of chemical fertilizers, pesticides, improved irrigation techniques, and mechanized farming tools. This holistic approach allowed countries with food shortages to become self-sufficient and even surplus producers.

India is one of the most notable examples of the Green Revolution's success. During the 1960s, the country was facing severe food shortages and depended heavily on imports and food aid. With the adoption of Green Revolution technologies, particularly in states like Punjab, Haryana, and western Uttar Pradesh, India saw a remarkable increase in agricultural productivity. Wheat production surged, transforming the country from a famine-prone nation to a food-secure one in just a few decades.

However, the Green Revolution also brought with it a set of challenges. The overuse of chemical fertilizers and pesticides led to soil degradation, water pollution, and a decline in soil fertility. The intensive use of groundwater for irrigation resulted in a sharp fall in water tables. Moreover, the benefits of the Green Revolution were not evenly distributed. Wealthier farmers with access to land, credit, and resources gained significantly, while small and marginal farmers were often left behind, increasing income inequality in rural areas.

In recent years, the focus has shifted towards a more sustainable model often referred to as the "Evergreen Revolution." This approach seeks to maintain high productivity while preserving environmental quality. It emphasizes organic farming, precision agriculture, conservation of biodiversity, and efficient water management.

In conclusion, the Green Revolution was a pivotal chapter in agricultural history that saved millions from starvation and reshaped the global food landscape. While it had its drawbacks, the lessons learned from it continue to guide modern agricultural policy and innovation. The challenge now lies in building upon its legacy to create a sustainable, inclusive, and environmentally friendly agricultural future.

"""

prompt = PromptTemplate(
    template= "For the below essay, evaluate the below essay and provide the feedback along with score out of 10 for the written essay\nessay->{essay}\n{instruction}",
    input_variables= ['essay'],
    partial_variables= {
        'instruction' : pyparser.get_format_instructions()
    }
)

chain = prompt | model | pyparser
result = chain.invoke({
    'essay' : essay
})

# step 1 : creating state

class essayState(TypedDict):
    essay_topic : str
    essay : str

    language_feedback : str
    cot_feedback : str
    doa_feedback : str
    final_feedback : str
    individual_scores : Annotated[List[int], operator.add]
    average_score : float

# step 2 : python function to perform task
def generate_topic(state : essayState):
    prompt = "generate a topic to write an essay, In order to train IPS, IAS, KPSC aspirants ask the topics that are frequently asked in this exams.make sure you just return the essay topic no need to provide options and also extra additional things"
    chain = model | parser
    result = chain.invoke(prompt)

    return {"essay_topic" : result}    

def get_language_feedback(state : essayState):
    essay = state['essay']
    prompt = PromptTemplate(
        template= "For the below essay, evaluate the below essay on the basis of {property} and provide the feedback along with score out of 10 for the written essay\nessay->{essay}\n{instruction}",
        input_variables= ['property','essay'],
        partial_variables= {
            'instruction' : pyparser.get_format_instructions()
        }
    )
    chain = prompt | model | pyparser
    result = chain.invoke({'property': "language control", 'essay' : essay})

    return {'language_feedback' : result.feedback, 'individual_scores' : [result.score]}

def get_cot_feedback(state : essayState):
    essay = state['essay']
    prompt = PromptTemplate(
        template= "For the below essay, evaluate the below essay on the basis of {property} and provide the feedback along with score out of 10 for the written essay\nessay->{essay}\n{instruction}",
        input_variables= ['property','essay'],
        partial_variables= {
            'instruction' : pyparser.get_format_instructions()
        }
    )
    chain = prompt | model | pyparser
    result = chain.invoke({'property': "Clarity of thought", 'essay' : essay})

    return {'cot_feedback' : result.feedback, 'individual_scores' : [result.score]}

def get_doa_feedback(state : essayState):
    essay = state['essay']
    prompt = PromptTemplate(
        template= "For the below essay, evaluate the below essay on the basis of {property} and provide the feedback along with score out of 10 for the written essay\nessay->{essay}\n{instruction}",
        input_variables= ['property','essay'],
        partial_variables= {
            'instruction' : pyparser.get_format_instructions()
        }
    )
    chain = prompt | model | pyparser
    result = chain.invoke({'property': "Depth of analysis", 'essay' : essay})

    return {'doa_feedback' : result.feedback, 'individual_scores' : [result.score]}

def get_final_feedback(state : essayState):
    doa = state['doa_feedback']
    lc = state['language_feedback']
    cot = state['cot_feedback']

    prompt = PromptTemplate(
        template= """generate a summary feedback on the below properties language control, depth of analysis, clarity of thought on the essay.\n
                    language control -> {lc}\n,
                    depth of analysis -> {doa}\n, 
                    clarity of thought -> {cot}\n,
        """,
        input_variables= ['lc','doa', 'cot'],
    )
    chain = prompt | model | parser
    result = chain.invoke({'lc': lc, 'doa' : doa, 'cot' : cot})

    avg_score = sum(state['individual_scores']) / len(state['individual_scores'])
    return {'final_feedback' : result , 'average_score' : avg_score}

# step 3 : define nodes
graph = StateGraph(essayState)

graph.add_node("generate_topic", generate_topic)
graph.add_node("get_language_feedback", get_language_feedback)
graph.add_node("get_cot_feedback", get_cot_feedback)
graph.add_node("get_doa_feedback", get_doa_feedback)
graph.add_node("get_final_feedback", get_final_feedback)

# step 4 : define path
graph.add_edge(START, "generate_topic")
graph.add_edge("generate_topic", "get_language_feedback")
graph.add_edge("generate_topic", "get_cot_feedback")
graph.add_edge("generate_topic", "get_doa_feedback")

graph.add_edge("get_language_feedback", "get_final_feedback")
graph.add_edge("get_cot_feedback", "get_final_feedback")
graph.add_edge("get_doa_feedback", "get_final_feedback")

graph.add_edge("get_final_feedback", END)

# step 5 : compile
workflow = graph.compile()

initial_state = {
    "essay" : essay
}

final_state = workflow.invoke(initial_state)

print(final_state)