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

#Chat Completion with HCP App Identity:
#Please update the values of client_id and client_secret in the below code.

# Import necessary libraries.
import openai
import os
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
import httpx

# Use an asynchronous client to make a POST request to the auth URL.
async with httpx.AsyncClient() as client:
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    resp = await client.post(auth, headers=headers, data=body, timeout=60)
    access_token = resp.json()["access_token"]

    # Define the deployment name and project ID.
    deployment_name = "gpt-4o_2024-11-20"

    # Define the Azure OpenAI endpoint and API version.
    shared_quota_endpoint = "https://-------------/api/cloud/api-management/ai-gateway/1.0"
    azure_openai_api_version="2025-01-01-preview"

    # Initialize the OpenAI client.
    oai_client = openai.AzureOpenAI(
        azure_endpoint=shared_quota_endpoint,
        api_version=azure_openai_api_version,
        azure_deployment=deployment_name,
        azure_ad_token=access_token,
        default_headers={
            "projectId": "------",
            # "x-upstream-env": "nonprod" #dev
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
            "projectId": '---------------'
        })

# Get embeddings for a query.
embeddings = openai_client_embeddings.embed_query("Hello world!")

# Print the embeddings.
print(embeddings)

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
            "projectId": "---------",
            # "x-upstream-env": "nonprod" #dev
        }
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ##DialogueAgent and DialogueSimulator classes

# COMMAND ----------

from typing import Callable, List
import tenacity
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import RegexParser
from langchain.prompts import PromptTemplate
from langchain.schema import (
    HumanMessage,
    SystemMessage,
)
import os

# COMMAND ----------

class DialogueAgent:
    def __init__(
        self,
        name: str,
        system_message: SystemMessage,
        model: ChatOpenAI,
    ) -> None:
        self.name = name
        self.system_message = system_message
        self.model = model
        self.prefix = f"{self.name}: "
        self.reset()

    def reset(self):
        self.message_history = ["Here is the conversation so far."]

    def send(self) -> str:
        """
        Applies the chatmodel to the message history and returns the message string
        """
        message = self.model(
            [
                self.system_message,
                HumanMessage(content="\n".join(self.message_history + [self.prefix])),
            ]
        )
        return message.content

    def receive(self, name: str, message: str) -> None:
        """
        Concatenates {message} spoken by {name} into message history
        """
        self.message_history.append(f"{name}: {message}")

# COMMAND ----------

class DialogueSimulator:
    def __init__(
        self,
        agents: List[DialogueAgent],
        selection_function: Callable[[int, List[DialogueAgent]], int],
    ) -> None:
        self.agents = agents
        self._step = 0
        self.select_next_speaker = selection_function

    def reset(self):
        for agent in self.agents:
            agent.reset()

    def inject(self, name: str, message: str):
        """
        Initiates the conversation with a {message} from {name}
        """
        for agent in self.agents:
            agent.receive(name, message)

        # increment time
        self._step += 1

    def step(self) -> tuple[str, str]:
        # 1. choose the next speaker
        speaker_idx = self.select_next_speaker(self._step, self.agents)
        speaker = self.agents[speaker_idx]

        # 2. next speaker sends message
        message = speaker.send()

        # 3. everyone receives message
        for receiver in self.agents:
            receiver.receive(speaker.name, message)

        # 4. increment time
        self._step += 1

        return speaker.name, message

# COMMAND ----------

# MAGIC %md
# MAGIC ##BiddingDialogueAgent class

# COMMAND ----------

# MAGIC %md
# MAGIC We define a subclass of DialogueAgent that has a bid() method that produces a bid given the message history and the most recent message.

# COMMAND ----------

class BiddingDialogueAgent(DialogueAgent):
    def __init__(
        self,
        name,
        system_message: SystemMessage,
        bidding_template: PromptTemplate,
        model: ChatOpenAI,
    ) -> None:
        super().__init__(name, system_message, model)
        self.bidding_template = bidding_template

    def bid(self) -> str:
        """
        Asks the chat model to output a bid to speak
        """
        prompt = PromptTemplate(
            input_variables=["message_history", "recent_message"],
            template=self.bidding_template,
        ).format(
            message_history="\n".join(self.message_history),
            recent_message=self.message_history[-1],
        )
        bid_string = self.model([SystemMessage(content=prompt)]).content
        return bid_string

# COMMAND ----------

# MAGIC %md
# MAGIC ##Challenge-Startup Pitch

# COMMAND ----------

# MAGIC %md
# MAGIC ###Define participants and debate topic

# COMMAND ----------

character_names = ["CTO", "CMO", "CEO", "Investor-Daniel", "Investor-Sandra"]
topic = "Startup pitch on startup focused on energy drinks with no caffeine"
word_limit = 15

# define the simulation
game_description = f"""Here is the topic for the startup pitch to investors Sandra and Daniel: {topic}.
The participants are: {', '.join(character_names)}."""

# COMMAND ----------

# @title Generate Context for Each Character (Helper Code Hidden)
player_descriptor_system_message = SystemMessage(
    content="You can add detail to the description of each participant"
)

def generate_character_description(character_name):
    character_specifier_prompt = [
        player_descriptor_system_message,
        HumanMessage(
            content=f"""{game_description}
            Please reply with a creative description of  {character_name}, in {word_limit} words or less, that emphasizes their personalities.
            Speak directly to {character_name}.
            Do not add anything else."""
        ),
    ]
    character_description = LLM(
        character_specifier_prompt
    ).content
    return character_description


def generate_character_header(character_name, character_description):
    return f"""{game_description}
Your name is {character_name}.
Your description is as follows: {character_description}
Your topic is: {topic}.
"""


def generate_character_system_message(character_name, character_header):
    return SystemMessage(
        content=(
            f"""{character_header}
You will speak in the style of {character_name}, and exaggerate their personality RESPONDING in under 450 characters.
You will come up with creative ideas related to {topic}.
Do not say the same things over and over again.
Speak in the first person from the perspective of {character_name}
ONLY SPEAK FOR YOURSELF WHO IS {character_name} AND NOT OTHER CHARACTERS FROM  {', '.join(character_names)}
For describing your own body movements, wrap your description in '*'.
Do not change roles!
Do not speak from the perspective of anyone else.
Speak only from the perspective of {character_name}.
Stop speaking the moment you finish speaking from your perspective.
Never forget to keep your response to {word_limit} words!
Do not add anything else.
    """
        )
    )


character_descriptions = [
    generate_character_description(character_name) for character_name in character_names
]
character_headers = [
    generate_character_header(character_name, character_description)
    for character_name, character_description in zip(
        character_names, character_descriptions
    )
]
character_system_messages = [
    generate_character_system_message(character_name, character_headers)
    for character_name, character_headers in zip(character_names, character_headers)
]


# COMMAND ----------

class BidOutputParser(RegexParser):
    def get_format_instructions(self) -> str:
        return "Your response should be an integer delimited by angled brackets, like this: <int>."


bid_parser = BidOutputParser(
    regex=r"<(\d+)>", output_keys=["bid"], default_output_key="bid"
)

@tenacity.retry(
    stop=tenacity.stop_after_attempt(2),
    wait=tenacity.wait_none(),  # No waiting time between retries
    retry=tenacity.retry_if_exception_type(ValueError),
    before_sleep=lambda retry_state: print(
        f"ValueError occurred: {retry_state.outcome.exception()}, retrying..."
    ),
    retry_error_callback=lambda retry_state: 0,
)  # Default value when all retries are exhausted

def ask_for_bid(agent) -> str:
    """
    Ask for agent bid and parses the bid into the correct format.
    """
    bid_string = agent.bid()
    bid = int(bid_parser.parse(bid_string)["bid"])
    return bid

def generate_character_bidding_template(character_header):
    bidding_template = f"""{character_header}

```
{{message_history}}
```

On the scale of 1 to 10, where 1 is least important to the startup pitch and 10 is extremely important and contribute, rank your recent message based on the context. Make sure to be very through in your ranking and only rank stuff that is important higher.

```
{{recent_message}}
```

{bid_parser.get_format_instructions()}
Do nothing else.
    """
    return bidding_template


character_bidding_templates = [
    generate_character_bidding_template(character_header)
    for character_header in character_headers
]

# COMMAND ----------

# MAGIC %md
# MAGIC ##Define the speaker selection function
# MAGIC Lastly define a speaker selection function select_next_speaker that takes each agent's bid and selects the agent with the highest bid (with ties broken randomly).
# MAGIC
# MAGIC Assume that you have a ask_for_bid function that takes in the agent and returns the numerical bid.

# COMMAND ----------

import numpy as np


def select_next_speaker(step: int, agents: List[DialogueAgent]) -> int:
    bids = []
    for agent in agents:
        bid = ask_for_bid(agent)
        bids.append(bid)

    # randomly select among multiple agents with the same bid
    max_value = np.max(bids)
    max_indices = np.where(bids == max_value)[0]
    idx = np.random.choice(max_indices)

    print("Bids:")
    for i, (bid, agent) in enumerate(zip(bids, agents)):
        print(f"\t{agent.name} bid: {bid}")
        if i == idx:
            selected_name = agent.name
    print(f"Selected: {selected_name}")
    print("\n")
    return idx

# COMMAND ----------

# MAGIC %md
# MAGIC ##Creating Bidding Dialogue Agents for each Character
# MAGIC Assuming that for each character we have character_name, character_system_message and bidding_template write a loop that populates the characters list with the BiddingDialogueAgent objects for each character.

# COMMAND ----------

characters = []
model=LLM


for character_name, character_system_message, bidding_template in zip(
    character_names, character_system_messages, character_bidding_templates
):
    characters.append(
        BiddingDialogueAgent(
            name=character_name,
            system_message=character_system_message,
            model=model,
            bidding_template=bidding_template,
        )
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ##Run the simulation
# MAGIC Populate the first_message field and also write the while loop to run the simulation.

# COMMAND ----------

max_iters = 10
n = 0

simulator = DialogueSimulator(agents=characters, selection_function=select_next_speaker)
simulator.reset()

first_message = "CEO, CMO, CTO You can now start pitching your ideas to our investor Sandra and Daniel"
simulator.inject("Moderator", first_message )
print(f"(Moderator): {first_message}")
print("\n")

while n < max_iters:
    name, message = simulator.step()
    print(f"({name}): {message}")
    print("\n")
    n += 1

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC
