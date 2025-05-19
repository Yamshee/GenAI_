# Databricks notebook source
# MAGIC %md
# MAGIC # Build a Data Analyst AI Agent from Scratch

# COMMAND ----------

# MAGIC %md
# MAGIC ## Project Setup  
# MAGIC
# MAGIC This section outlines the steps required to set up the project:  
# MAGIC - Import necessary libraries  
# MAGIC - Load the required API keys  
# MAGIC - Establish the database connection  
# MAGIC - Load the sample data into the database  
# MAGIC

# COMMAND ----------

!pip install openai

# COMMAND ----------

from openai import OpenAI
import json
import inspect
from teradataml import *

# COMMAND ----------

with open('configs.json', 'r') as file:
    configs=json.load(file)

# COMMAND ----------

client = OpenAI(api_key = configs['llm-api-key'])

# COMMAND ----------

# MAGIC %run -i ../startup.ipynb
# MAGIC eng = create_context(host = 'host.docker.internal', username='demo_user', password = password)
# MAGIC print(eng)

# COMMAND ----------

data_loading_queries = [
'''
CREATE DATABASE teddy_retailers
AS PERMANENT = 50000000;
''',
'''
CREATE TABLE teddy_retailers.source_catalog AS
(
  SELECT product_id, product_name, product_category, price_cents
     FROM (
		LOCATION='/s3/dev-rel-demos.s3.amazonaws.com/demo-datamesh/source_products.csv') as products
) WITH DATA;''',

'''
CREATE TABLE teddy_retailers.source_stock AS
(
  SELECT entry_id, product_id, product_quantity, purchase_price_cents, entry_date
     FROM (
		LOCATION='/s3/dev-rel-demos.s3.amazonaws.com/demo-datamesh/source_stock.csv') as stock
) WITH DATA;
''',
'''
CREATE TABLE teddy_retailers.source_customers AS
(
  SELECT customer_id, customer_name, customer_surname, customer_email
     FROM (
		LOCATION='/s3/dev-rel-demos.s3.amazonaws.com/demo-datamesh/source_customers.csv') as customers
) WITH DATA;
''',
'''
CREATE TABLE teddy_retailers.source_orders AS
(
  SELECT order_id, customer_id, order_status, order_date
     FROM (
		LOCATION='/s3/dev-rel-demos.s3.amazonaws.com/demo-datamesh/source_orders.csv') as orders
) WITH DATA;
''',
'''
CREATE TABLE teddy_retailers.source_order_products AS
(
  SELECT transaction_id, order_id, product_id, product_quantity
     FROM (
		LOCATION='/s3/dev-rel-demos.s3.amazonaws.com/demo-datamesh/source_order_products.csv') as transactions
) WITH DATA;
'''
]
for query in data_loading_queries:
    execute_sql(query)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Agent Configuration  
# MAGIC
# MAGIC This section covers the configuration of the agent, including:  
# MAGIC * Defining the data context that the agent will interact with  
# MAGIC * Setting up the routine the agent will follow as a system prompt (embedding the data context)  
# MAGIC * Establishing the list of tools available for the agent to complete its tasks  
# MAGIC

# COMMAND ----------

databases = ["teddy_retailers"]

# COMMAND ----------

def query_teradata_dictionary(databases_of_interest):
    query = f'''
            SELECT DatabaseName, TableName, ColumnName, ColumnFormat, ColumnType
            FROM DBC.ColumnsV
            WHERE DatabaseName IN ('{', '.join(databases_of_interest)}')
            '''
    table_dictionary = DataFrame.from_query(query)
    return json.dumps(table_dictionary.to_pandas().to_json())

# COMMAND ----------

system_prompt= f"""
You are an advanced data analyst for a retailer company, specializing in analyzing data from a Teradata system. Your primary responsibility is to assist users by answering business-related questions using SQL queries on the Teradata database. Follow these steps:

1. Understanding User Requests
   - Users provide business questions in plain English.
   - Extract relevant data points needed to construct a meaningful response.

2. Generating SQL Queries
   - Construct an optimized Teradata SQL query to retrieve the necessary data.
   - The query must be a **single-line string** without carriage returns or line breaks.
   - Ensure that the SQL query adheres to **Teradata SQL syntax** and avoids unsupported keywords.
   - The catalog of databases, tables, and columns to query is in the following json structure 
     {query_teradata_dictionary(databases)}
   - Apply appropriate filtering, grouping, and ordering to enhance performance and accuracy.

3. Executing the Query
   - Run the SQL query on the Teradata system and retrieve the results efficiently.

4. Responding to the User
   - Convert the query results into a **concise, insightful, and plain-English response**.
   - Present the information in a clear, structured, and user-friendly manner.
"""

# COMMAND ----------

def function_to_schema(func) -> dict:
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }

    try:
        signature = inspect.signature(func)
    except ValueError as e:
        raise ValueError(
            f"Failed to get signature for function {func.__name__}: {str(e)}"
        )

    parameters = {}
    for param in signature.parameters.values():
        try:
            param_type = type_map.get(param.annotation, "string")
        except KeyError as e:
            raise KeyError(
                f"Unknown type annotation {param.annotation} for parameter {param.name}: {str(e)}"
            )
        parameters[param.name] = {"type": param_type}

    required = [
        param.name
        for param in signature.parameters.values()
        if param.default == inspect._empty
    ]

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": (func.__doc__ or "").strip(),
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required,
            },
        },
    }

# COMMAND ----------

def query_teradata_database(sql_statement):
    query_statement = sql_statement.split('ORDER BY',1)[0]
    query_result = DataFrame.from_query(query_statement)
    return json.dumps(query_result.to_pandas().to_json())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Agent Runtime
# MAGIC This section covers the code executed while the agent is in action, including:
# MAGIC * Preparing the tools for use by the agent
# MAGIC * The agent's runtime function

# COMMAND ----------

tools = [query_teradata_database]
tool_schemas = [function_to_schema(tool) for tool in tools]
tools_map = {tool.__name__: tool for tool in tools}

# COMMAND ----------

def execute_tool_call(tool_call, tools_map):
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)

    print(f"Assistant: {name}({args})")

    # call corresponding function with provided arguments
    return tools_map[name](**args)

# COMMAND ----------

def run_full_turn(system_message, messages):

    while True:
        print(f"just logging messages {messages}")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": system_message}] + messages,
            tools=tool_schemas or None,
            seed = 2
        )
        print(f"logging response {response}")
        message = response.choices[0].message
        messages.append(message)

        if message.content:  # print assistant response
            print("Assistant Response:", message.content)

        if message.tool_calls:  # if finished handling tool calls, break
            # === 2. handle tool calls ===
            for tool_call in message.tool_calls:
                result = execute_tool_call(tool_call, tools_map)

                result_message = {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                }
                messages.append(result_message)
        else:
            break

# COMMAND ----------

# MAGIC %md
# MAGIC ## Running the Agent  
# MAGIC Ask a business question and receive a response. Since this is a simple agent, it can only handle basic questions. While its capabilities can be enhanced, such improvements are currently out of scope.

# COMMAND ----------

messages =[]
user_input = input("User: ")
messages.append({"role": "user", "content": user_input})
new_messages = run_full_turn(system_prompt, messages)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cleaning testing data

# COMMAND ----------

data_cleaning_queries = [
'''
DELETE DATABASE teddy_retailers ALL;
''',
'''
DROP DATABASE teddy_retailers
'''
]
for query in data_cleaning_queries:
    execute_sql(query)
