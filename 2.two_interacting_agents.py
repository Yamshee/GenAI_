# Databricks notebook source
# PIP Install
%pip install azure-identity
%pip install openai
%pip install langchain-openai
%pip install load-dotenv
%pip install httpx


# COMMAND ----------

# Use termcolor to make it easy to colorize the outputs.
!pip install termcolor > /dev/null
!pip install langchain_experimental
!pip install tiktoken


# COMMAND ----------

!pip install faiss-cpu

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------


# Import necessary libraries.
import openai
import os
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
import httpx

# Define the authentication URL and credentials.


# Use an asynchronous client to make a POST request to the auth URL.
async with httpx.AsyncClient() as client:
   
    # Define the deployment name and project ID.
    deployment_name = "gpt-4o_2024-11-20"

    # Define the Azure OpenAI endpoint and API version.
    shared_quota_endpoint = "https://--------------/api/cloud/api-management/ai-gateway/1.0"
    azure_openai_api_version="2025-01-01-preview"

    # Initialize the OpenAI client.
    oai_client = openai.AzureOpenAI(
        azure_endpoint=shared_quota_endpoint,
        api_version=azure_openai_api_version,
        azure_deployment=deployment_name,
        azure_ad_token=access_token,
        default_headers={
            "projectId": "------------",
          
        }
    )

# COMMAND ----------

# Define the messages to be processed by the model.
messages = [{"role": "user", "content": "Hi, what is Prime number"}]

# Request the model to process the messages.
response = oai_client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
    )

# Print the response from the model.
print(response.model_dump_json(indent=2))

# COMMAND ----------

from langchain_openai.embeddings import AzureOpenAIEmbeddings

# Initialize the OpenAI client.
openai_client_embeddings = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-ada-002_2",
    model="text-embedding-ada-002",
    api_version="2025-01-01-preview",
    azure_endpoint=shared_quota_endpoint,
   tiktoken_model_name='cl100k_base',
        azure_ad_token=access_token,
        default_headers={ 
            "projectId": '---------'
        })

# Get embeddings for a query.
embeddings = openai_client_embeddings.embed_query("Hello world!")

# Print the embeddings.
print(embeddings)

# COMMAND ----------


from datetime import datetime, timedelta
from typing import List
import math
import os
import logging
logging.basicConfig(level=logging.ERROR)
from langchain.chat_models import ChatOpenAI
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers import TimeWeightedVectorStoreRetriever

from termcolor import colored
from langchain_experimental.generative_agents import (
    GenerativeAgent,
    GenerativeAgentMemory,
)

# COMMAND ----------


import faiss
from langchain.vectorstores import FAISS

# COMMAND ----------

# Initialize the OpenAI client.
from langchain_openai import AzureChatOpenAI
LLM = AzureChatOpenAI(
        azure_endpoint=shared_quota_endpoint,
        api_version=azure_openai_api_version,
        azure_deployment=deployment_name,
        azure_ad_token=access_token,
        model_name='GPT-4o',
        default_headers={
            "projectId": "--------",
            # "x-upstream-env": "nonprod" #dev
        }
    )

# COMMAND ----------

USER_NAME = "YAM"  # The name you want to use when interviewing the agent.


# COMMAND ----------

# MAGIC %md
# MAGIC ## Implementing Your First Generative Agent
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

def relevance_score_fn(score: float) -> float:
    """Return a similarity score on a scale [0, 1]."""
    # This will differ depending on a few things:
    # - the distance / similarity metric used by the VectorStore
    # - the scale of your embeddings (OpenAI's are unit norm. Many others are not!)
    # This function converts the euclidean norm of normalized embeddings
    # (0 is most similar, sqrt(2) most dissimilar)
    # to a similarity function (0 to 1)
    return 1.0 - score / math.sqrt(2)


def create_new_memory_retriever():
    """Create a new vector store retriever unique to the agent."""
    # Define your embedding model
    embeddings_model = openai_client_embeddings
    # Initialize the vectorstore as empty
    embedding_size = 1536
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = FAISS(
        embeddings_model.embed_query,
        index,
        InMemoryDocstore({}),
        {},
        relevance_score_fn=relevance_score_fn,
    )
    return TimeWeightedVectorStoreRetriever(
        vectorstore=vectorstore, other_score_keys=["importance"], k=15
    )

# COMMAND ----------

alexis_memory = GenerativeAgentMemory(
    llm=LLM,
    memory_retriever=create_new_memory_retriever(),
    verbose=False,
    reflection_threshold=8,  # we will give this a relatively low number to show how reflection works
)

# COMMAND ----------

# Defining the Generative Agent: Alexis
alexis = GenerativeAgent(
    name="Alexis",
    age=30,
    traits="curious, creative writer, world traveler",  # Persistent traits of Alexis
    status="exploring the intersection of technology and storytelling",  # Current status of Alexis
    memory_retriever=create_new_memory_retriever(),
    llm=LLM,
    memory=alexis_memory,
)

# COMMAND ----------

# The current "Summary" of a character can't be made because the agent hasn't made
# any observations yet.
print(alexis.get_summary())

# COMMAND ----------

# We can add memories directly to the memory object

alexis_observations = [
    "Alexis recalls her morning walk in the park",
    "Alexis feels excited about the new book she started reading",
    "Alexis remembers her conversation with a close friend",
    "Alexis thinks about the painting she saw at the art gallery",
    "Alexis is planning to learn a new recipe for dinner",
    "Alexis is looking forward to her weekend trip",
    "Alexis contemplates her goals for the month."
]

for observation in alexis_observations:
    alexis.memory.add_memory(observation)



# We will see how this summary updates after more observations to create a more rich description.
print(alexis.get_summary(force_refresh=True))

# COMMAND ----------

print(alexis.get_summary())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Interacting and Providing Context to Generative Characters

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pre-Interview with Character
# MAGIC
# MAGIC Before sending our character on their way, let's ask them a few questions.

# COMMAND ----------

def interview_agent(agent: GenerativeAgent, message: str) -> str:
    """Help the notebook user interact with the agent."""
    new_message = f"{USER_NAME} says {message}"
    return agent.generate_dialogue_response(new_message)[1]

# COMMAND ----------

interview_agent(alexis, "What do you like to do?")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step through the day's observations.

# COMMAND ----------

# Let's give Alexis a series of observations to reflect on her day
# Adding observations to Alexis' memory
alexis_observations_day = [
    "Alexis starts her day with a refreshing yoga session.",
    "Alexis spends time writing in her journal.",
    "Alexis experiments with a new recipe she found online.",
    "Alexis gets lost in her thoughts while gardening.",
    "Alexis decides to call her grandmother for a heartfelt chat.",
    "Alexis relaxes in the evening by playing her favorite piano pieces.",
]

for observation in alexis_observations_day:
    alexis.memory.add_memory(observation)


# COMMAND ----------

# Let's observe how Alexis's day influences her memory and character
for i, observation in enumerate(alexis_observations_day):
    _, reaction = alexis.generate_reaction(observation)
    print(colored(observation, "green"), reaction)
    if ((i + 1) % len(alexis_observations_day)) == 0:
        print("*" * 40)
        print(
            colored(
                f"After these observations, Alexis's summary is:\n{alexis.get_summary(force_refresh=True)}",
                "blue",
            )
        )
        print("*" * 40)


# COMMAND ----------

# MAGIC %md
# MAGIC ###Adding Multiple Characters

# COMMAND ----------

# Creating Jordan's Memory
jordan_memory = GenerativeAgentMemory(
    llm=LLM,
    memory_retriever=create_new_memory_retriever(),
    verbose=False,
    reflection_threshold=7,  # Set to illustrate Jordan's reflective capabilities
)

# Defining the Generative Agent: Jordan
jordan = GenerativeAgent(
    name="Jordan",
    age=28,
    traits="tech enthusiast, avid gamer, foodie",  # Persistent traits of Jordan
    status="navigating the world of tech startups",  # Current status of Jordan
    memory_retriever=create_new_memory_retriever(),
    llm=LLM,
    memory=jordan_memory,
)

# Adding observations to Jordan's memory
jordan_observations_day = [
    "Jordan finished a challenging coding project last night",
    "Jordan won a local gaming tournament over the weekend",
    "Jordan tried a new sushi restaurant and loved it",
    "Jordan read an article about the latest AI advancements",
    "Jordan is planning a meetup with tech enthusiasts",
    "Jordan discovered a bug in his latest app prototype",
    "Jordan booked tickets for a tech conference next month",
    "Jordan feels excited about a potential startup idea",
    "Jordan spent the evening playing video games to unwind",
    "Jordan is considering enrolling in a machine learning course"
]

for observation in jordan_observations_day:
    jordan.memory.add_memory(observation)

print(jordan.get_summary())


# COMMAND ----------

# MAGIC %md
# MAGIC ###Dialogue between Generative Agents

# COMMAND ----------

def run_conversation(agents: List[GenerativeAgent], initial_observation: str) -> None:
    """Runs a conversation between agents."""
    _, observation = agents[1].generate_reaction(initial_observation)
    print(observation)
    max_turns = 3
    turns = 0
    while turns<=max_turns:
        break_dialogue = False
        for agent in agents:
            stay_in_dialogue, observation = agent.generate_dialogue_response(
                observation
            )
            print(observation)
            # observation = f"{agent.name} said {reaction}"
            if not stay_in_dialogue:
                break_dialogue = True
        if break_dialogue:
            break
        turns += 1

# COMMAND ----------

agents = [alexis, jordan]
run_conversation(
    agents,
    "Alexis said: Hey Jordan, I've been exploring how technology influences creativity lately. Since you're into tech, I was wondering if you've seen any interesting intersections in your field?",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Let's interview our agents after their conversation

# COMMAND ----------

# MAGIC %md
# MAGIC Since the generative agents retain their memories from the day, we can ask them about their plans, conversations, and other memoreis.

# COMMAND ----------

interview_agent(jordan, "How was your conversation with Alexis?")

# COMMAND ----------

interview_agent(alexis, "How was your conversation with Jordan?")

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC #Trivia Night

# COMMAND ----------

# Creating Jordan's Memory
jordan_memory = GenerativeAgentMemory(
    llm=LLM,
    memory_retriever=create_new_memory_retriever(),
    verbose=False,
    reflection_threshold=7,  # Set to illustrate Jordan's reflective capabilities
)


jordan = GenerativeAgent(
    name="Jordan",
    age=28,
    traits=" only has knowledge about tech and not other topics",  # Persistent traits of Jordan
    status="navigating the world of tech startups",  # Current status of Jordan
    memory_retriever=create_new_memory_retriever(),
    llm=LLM,
    memory=jordan_memory,
)

alexis_memory = GenerativeAgentMemory(
    llm=LLM,
    memory_retriever=create_new_memory_retriever(),
    verbose=False,
    reflection_threshold=8,  # we will give this a relatively low number to show how reflection works
)

alexis = GenerativeAgent(
    name="Alexis",
    age=30,
    traits="only has knowledge geography related and not other topics",  # Persistent traits of Alexis
    status="exploring the intersection of technology and storytelling",  # Current status of Alexis
    memory_retriever=create_new_memory_retriever(),
    llm=LLM,
    memory=alexis_memory,
)


# COMMAND ----------

def run_competitive_trivia(agents: List[GenerativeAgent], questions: List[str]) -> None:
    """Runs a competitive trivia night between agents."""
    for question in questions:
        print(f"Trivia Question: {question}")

        for agent in agents:
            response = agent.generate_dialogue_response(question)[1]
            print(f"{agent.name}'s Answer: {response}")

        print("-" * 40)

# Define a list of trivia questions covering various topics
trivia_questions = [
    "What is the capital city of France?",
    "Who is known as the father of modern computing?",
    "Can you name a famous work of art by Leonardo da Vinci?",
]

agents = [alexis, jordan]
# Run the competitive trivia night
run_competitive_trivia(agents, trivia_questions)

# COMMAND ----------

