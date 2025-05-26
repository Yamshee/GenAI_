# Databricks notebook source
# PIP Install
%pip install azure-identity
%pip install openai
%pip install langchain-openai
%pip install load-dotenv
%pip install httpx
# Use termcolor to make it easy to colorize the outputs.
!pip install termcolor > /dev/null
!pip install langchain_experimental
!pip install tiktoken
!pip install faiss-cpu


# Import necessary libraries.
import openai
import os
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
import httpx

# Define the deployment name and project ID.
deployment_name = "gpt-4o_2024-11-20"

# Define the Azure OpenAI endpoint and API version.
shared_quota_endpoint = "https://............../api/cloud/api-management/ai-gateway/1.0"
azure_openai_api_version="2025-01-01-preview"

# Initialize the OpenAI client.
oai_client = openai.AzureOpenAI(
        azure_endpoint=shared_quota_endpoint,
        api_version=azure_openai_api_version,
        azure_deployment=deployment_name,
        azure_ad_token=access_token,
        default_headers={
            "projectId": "",
            # "x-upstream-env": "nonprod" #dev
        }
    )

# Define the messages to be processed by the model.
messages = [{"role": "user", "content": "Hi, what is Prime number"}]

# Request the model to process the messages.
response = oai_client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
    )

# Print the response from the model.
print(response.model_dump_json(indent=2))


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
            "projectId": ''
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
import faiss
from langchain.vectorstores import FAISS
from langchain_openai import AzureChatOpenAI
from termcolor import colored
from langchain_experimental.generative_agents import (

    GenerativeAgent,
    GenerativeAgentMemory,
)

# COMMAND ----------

# Initialize the OpenAI client.
LLM = AzureChatOpenAI(
        azure_endpoint=shared_quota_endpoint,
        api_version=azure_openai_api_version,
        azure_deployment=deployment_name,
        azure_ad_token=access_token,
        model_name='GPT-4o',
        default_headers={
            "projectId": "",
           
        }
    )

# COMMAND ----------
USER_NAME = "YAM"  # The name you want to use when interviewing the agent.


# COMMAND ----------
# MAGIC ## Implementing Your First Generative Agent


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
