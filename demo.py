import openai
import snowflake.connector
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import re
import asyncio
from powerbiclient.authentication import InteractiveLoginAuthentication
from fastapi.middleware.cors import CORSMiddleware
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
import jwt
from decimal import Decimal

import uvicorn
import json
import pandas as pd

import re
import openai
import os
from typing import List, Dict, Any
from fastapi import HTTPException, FastAPI
from pydantic import BaseModel
from roster import final_filteration
# from connect_db import DataBase
import pyodbc

# Load environment variables from .env file
load_dotenv()

# FastAPI setup
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
 

# OpenAI API Setup
openai.api_type = "xyz"
openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_version = "lmn"
openai.api_key = os.getenv("OPENAI_API_KEY")


# Define the global connection object for ms-sql:
conn = None

# Database connection setup
server = 'abc'  # e.g., 'localhost' or '127.0.0.1'
database = 'chatbot_database'
username = 'abc'
password = 'def'
conn_str = f'DRIVER={{SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}'

# Path to the metadata file 
METADATA_FILE = "metadata.json"

# Function to check if metadata file exists and load it
def load_metadata():
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "r",encoding="utf-8") as file:
            try:
                return json.load(file)  # Load JSON content if available
            except json.JSONDecodeError:
                return {}  # Return an empty dict if the JSON is malformed
    return {}


def load_private_key_string(private_key_string, private_key_password):
    """
    Load the encrypted private key from a string using a password
    """
    try:
        # Convert the string key to bytes
        private_key_bytes = private_key_string.encode()
        p_key = serialization.load_pem_private_key(
            private_key_bytes,
            password=private_key_password.encode(),  # Convert password to bytes
            backend=default_backend()
        )
        print("Successful encryption of private key")
        return p_key.private_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
    except Exception as e:
        print(f"Error loading private key: {str(e)}")
        return None

# Function to save metadata to the file
def save_metadata(metadata):
    with open(METADATA_FILE, "w") as file:
        json.dump(metadata, file, indent=4)


private_key_string='''-----BEGIN ENCRYPTED PRIVATE KEY-----
MIIFHDBOBgkqhkiG9w0BBQ0wQTApBgkqhkiG9w0BBQwwHAQIcYMMkI4o24sCAggA
MAwGCCqGSIb3DQIJBQAwFAYIKoZIhvcNAwcECOr0Ka973lzfBIIEyEaFMO1Hom+F
wJHBsApn+sHraqFGPiD+bOqpIRZfDvmZAbNpBoQbZIumtnFpwfcw+IBkXLoY6lC5
C1RACOMJm7m2vj9rm2Xv4csWuTKEzc8nk+pCJtt9UA7QxyaSG70CAhz5JaHtCcDb
xhpFUrr2o0gdYYf7BBH2TQGVRd14jPbqmeU32iDEUcO6BU89E9q3vbu/XxiY3aUh
H25MBFwP7LASDSf2RLwIijdl0xQFPK/FSkwWP/noQgg+KN2vsXeIfCYwlFuzdTCQ
8dMuvzjqCn4D31cEM8sGdJPB5+Qz+0xGkna8uapoc2+b2k/vNA5uaUd1iKB2ybI8
aP/vh2b/r+si7aVQIwuOvJPsnlT918mYieTA7lZ8VLfASBz2t/bneTIBQ9VPuigK
0MsvgG49ANMs6TAZ1m4RsnDjL/wM55IvBUFWQRy3dikCQoAmog3tjjgDYf+hWp/l
ERZabC8/9CcBaHT/sSukYM3rCafr1/vez90BERuSFRCdwQSVM8GdlXKxvsgUwGct
/hSiiMHGTAkftmVl6qgkQO9sAxM0rzC7k4ThypVlRbk95eUwBitUqznnL+mCJykA
3+Pr7FPCQY4Lxwulg0ltnWXDsUT1NDPUW33YzqglIfhe1MGByygnAZtPUSrqs4Fy
qlrVz0dq6yvx1gFi3v5H+vuCMXudQGh3zxye9DIG6WBL6lXCB8jnVDJMFYGzl0WS
FoRT68ImvqULGix5LvGZnnwLSgBVfLczg7dzqa+CXmk7YfCS9SPpI8489/v5h0j0
VENtcW8Gnq6Sp6Q0rrCEQS88IAERsH467geaOnZznv6b2gqB68qIVrt9wXn0nqpt
oVg8pR79Ep6pPOBsykgF19WQJQbU3cDwLB1ff4yS7eBX4ZViEtlftRTvUGF9DctY
2Z5Al/yoL9VWFORyWJ56vsaKdUt4t/rXTGhzPPiYskazX+NPv0xXfxVSGMbfdpar
dbfoWSKBmDsx3ssyKtio5yYhrA1QhJpOWEGl1Hvi4TjnRg3zhRtDXzxWlMsSSG5X
GKPm6sMpJVTV0XL0vcjT+40VvuIMyMKCHBRDWcJm9WdIY9uIDGwlg2TcVYkXFo+o
nCQdDdZvauss16ZgCSDxqFsZYXj0J8RpwYLITNrsMuit5c5n/cEkm8JicPUA8Kyq
X2TwnVzRDITr5+IGE5N07ALkmIzif2nAnINARUOz6Z0t/jKUuUvIt2RslYc9gH7g
LSUropF4vBcwk+tfzMej/R5/0l8gxUnvffAIqFWTMCxnCb0ryA7iLAMqaLH0ZiAb
hmurU4Qz+AOMzTu+0lCA39xIHOQ+FOXUbQmF3tYkMvV1fDgR4Ob9YH9VgLuuwBm/
SuQJ2gdz7zhYxFAlBsjz9ddQabJsRr5r82fv5cbmdcrehw9EN04PzBJg/Ccyz8RJ
+jYJgR7Tk5PhaeMAdGMO7+D+7mkuOIVhk7K0p4fVcSda0+2khUK2JsTtXMKhihxJ
zuMokPksJehmbOt6XDhXsPREwWbsAy5RF0zBteovzRkmrr6MqoR5r8Iu8vDe4WdZ
nGc6P9S0MQLcsr0sk0BIfE2XuUz9CtcH/Te+U7kVXfaY7Qt/gfog5hs6QaNYem3j
lzcRWvqP3dDjER1gXeku0w==
-----END ENCRYPTED PRIVATE KEY-----'''
 

private_key = load_private_key_string(private_key_string, private_key_password)

#Define connection parameters
connection_params = {
    'account': account,
    'user': user,
    'warehouse': warehouse,
    'database':database,
    'schema':schema,
    'role':role,
    "private_key": private_key,
    'client_session_keep_alive': True   
}


# Function to establish a Snowflake connection (keeping it synchronous since Snowflake connector is not async)
def get_snowflake_connection():
    global conn
    if conn is None:
        print("Authenticating and establishing Snowflake connection...")
        conn = snowflake.connector.connect(**connection_params)
        print("Connection established.")
    else:
        print("Reusing existing Snowflake connection.")
    return conn

def get_mssql_connection():
    try:
        global conn
        if conn is None:
            print('connection  starting.......')
            conn = pyodbc.connect(conn_str)
            # cursor = conn.cursor()
            print("Connection successful!")
        else:
            print('Reusing existing MS-SQL connection.')
        return conn
    except Exception as e:
        print("Error connecting to database:", e)



# Function to query Snowflake using the global connection (now asynchronous)
async def query_snowflake(query):
    conn = get_snowflake_connection()  # Get the connection (connects if not already connected)
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _execute_query, conn, query)
    return result

async def query_mssql(query):
    conn = get_mssql_connection()
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _execute_query, conn, query)
    return result

def _execute_query(conn, query):
    # This function will run the query synchronously inside a thread pool
    with conn.cursor() as cur:
        cur.execute(query)
        return cur.fetchall()


token = None

def authenticate_user():
    global token
    # Check if the token is already authenticated
    if token is None:
        token = authenticate()  # Only authenticate if the token is not already set
    return token

# Extract email from JWT token
def extract_email_from_token(token):
    decoded_token = jwt.decode(token, options={"verify_signature": False})
    return decoded_token.get("unique_name").lower()



# Create a Pydantic model for input
class QueryRequest(BaseModel):
    userMessage: str

# Define the global interactive object
interactive_auth = None

# Authenticate user and get access token
def authenticate():
    interactive_auth = InteractiveLoginAuthentication()
    return interactive_auth.get_access_token()



async def get_table_context(database_name: str, schema_name: str, table_name: str):
    columns_query = f"""
    SELECT COLUMN_NAME, DATA_TYPE
    FROM {database_name}.INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_SCHEMA = '{schema_name}' AND TABLE_NAME = '{table_name}'
    """
    columns = await query_mssql(columns_query)

    columns_description = "\n".join([f"- **{col[0]}**: {col[1]}" for col in columns])

    table_description_query = f"""
    SELECT COMMENT 
    FROM {database_name}.INFORMATION_SCHEMA.TABLES
    WHERE TABLE_SCHEMA = '{schema_name}' AND TABLE_NAME = '{table_name}'
    """
    res = await query_mssql(table_description_query)
    table_description = res[0][0] if res else "No description available"

    # Construct the table context
    context = f"""
    Here is the table name <tableName> {table_name} </tableName>
    
    <tableDescription>{table_description}</tableDescription>
    
    Here are the columns of the {table_name}:
    
    <columns>\n\n{columns_description}\n\n</columns>
    """
    return context


# Function to get system prompt with table context
def get_system_prompt(table_context):
    GEN_SQL = """
    You will be acting as a Microsoft SQL Server (MS SQL) expert.
    Your goal is to give correct, executable SQL queries to users.
    You are given tables, the table name is in <tableName> tag,  and the columns are in <columns> tag.
    The user will ask questions; for each question, you should respond with a SQL query based on the table and answer only if the question is related to context.
    Understand the question carefully.

    During create Microsoft SQL Server (MS SQL) Query make sure the query will generate correct syntax.

    {0}

    Here are critical rules for the interaction you must abide:
    1. Remove null values in all cases anyhow and if the user doesn't specify restrict to 20 rows.  
    2. If there are duplicate values in data write query in such a way the duplicates are removed use 'DISTINCT' function to remove those values.
    3. Always add column name, which is present in "order by" clause, in *select clause also* 
    4. When TIME_PERIOD is not mentioned in the question take *'Year to Date' as default always*.
    5. When number associate with Highest then provide as it is. 
    6. When number not associate with Highest then provide highest only.
    7. Whenever a question is asked with some time period mentioned in it, be sure to reflect it in the response as well.
    8. use offset and rows fetch next statement to limit records. Dont use top or limit.
    9. TIME_PERIOD are varchar and are not to be converted to int.
    """
    return GEN_SQL.format(table_context)



# Function to load and filter admin data based on email
def load_and_filter_master_data(email: str, admin_file_path: str, admin_email_column: str, national_manager_email: str):
    df_admin = pd.read_csv(admin_file_path,encoding="utf-8")
    df_admin.columns = df_admin.columns.str.lower()  # Normalize column names to lowercase

    # Normalize the user's email to lowercase and strip any extra spaces
    email = email.strip().lower()
    df_admin[admin_email_column.lower()] = df_admin[admin_email_column.lower()].str.strip().str.lower()

    if email in df_admin[admin_email_column.lower()].values:
        return national_manager_email
    else:
        return email



# Function to filter tables based on user query and schema using LLM
async def filter_top_tables_based_on_query(user_query: str) -> list:
    # Build the prompt for the LLM model to select top 3 tables based on user query and schema
    try:
        prompt = f"""
        
        Based on the user query provide only the relevant table name.

        Table names and descriptions:

            drug sales: This table contains sales transaction details of pharmaceutical products across various cities, channels, and sales teams. This table can be used for analyzing plan-level access and utilization metrics for Brands 1, 2, 3, 4, and 5 across various PBMs and payers. It includes data relevant for understanding payer dynamics, formulary positioning, claims performance (TRx/NBRx), and rejection trends. State and Territory can be used to derive geographic levels (e.g., National, Regional, etc.), where “National” refers to the overall U.S. average, and “Regional” refers to the corresponding Territory average. This table is suitable for access analytics, formulary tracking, and payer engagement insights. It should not be used for Goal attainment or Calls per Day calculations.
            exporting: This table contains export-related sales data of pharmaceutical products, including pricing, destination, agents, and transaction dates. This table can be used for data source and the date of last refresh. It provides details related to the source of data and the time of its most recent update.
            segmentation: This table expresses data segmented based on id, sex, martial status, age, education, income. This table can be used for analyzing the performance of all Brands across various segmentss within different regions and territories. This table is essential for analyzing brand performance at the regional and territory level.
        
        Return the table name in json format
        Output must be raw JSON only — no code blocks, no triple backticks, no explanations.
        
        example :
         table_name_key :[table name 1, table name 2]
         .....etc..
        """
        # PS: If list of ECPs is requested then always recommend *MY_ECPS* alongwith other relevant table
        # PS: If anything related to *visit* is asked then always recommend *RECOMMENDATIONS* alongwith other relevant table
        # PS: Do NOT use FILTER command

        # """
        response = azure_openai_call(user_query, prompt)
        print('------------')
        print(response)
        return response
    except Exception as e:
        print(f'error is coming from filter top tables based on query :: {e}')


def azure_openai_call(system_prompt,user_message):

    completion = openai.ChatCompletion.create(
            engine="gpt-4",
            messages=[ 
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=2000,
            temperature=0,
        )
    
    return completion.choices[0].message['content']

schema = None

print("---------------------------done")
# Function to handle the query and process the request
@app.post("/query")
async def query_mssql_api(request: QueryRequest):
    try:
        
        user_message = request.userMessage
        if len(list(user_message.split())) <= 3:
            resp = 'Sorry I could not understand can you provide more details.'
            return [0, resp]
        # token = authenticate_user()
        # user_email = extract_email_from_token(token).lower()
        # print(user_email)

        # Step 1: Filter the user email (determine national manager or user email)
        # final_email = load_and_filter_master_data(user_email, admin_file_path, admin_email_column, national_manager_email)
        # user_message += f" Filter by mail id - {final_email}"
        print(user_message)

        # Step 2: Prepare schema and database for querying

        schema_name = os.getenv("MSSQL_SCHEMA")
        database_name = os.getenv("MSSQL_DATABASE")
        tables_to_query = ['drug sales', 'exporting', 'segementation']

        # Load metadata from file
        metadata = load_metadata()

        # Query Snowflake to get relevant tables
        tables_query = f"""
        SELECT TABLE_CATALOG, TABLE_SCHEMA, TABLE_NAME 
        FROM {database_name}.INFORMATION_SCHEMA.TABLES
        WHERE TABLE_SCHEMA = '{schema_name}'   
        AND TABLE_NAME IN ({','.join([f"'{table}'" for table in tables_to_query])})
        """

        tables = await query_mssql(tables_query)

        # Step 4: Generate table contexts for the filtered tables
        table_contexts = []
        

        if len(metadata) == 0:
            # If metadata is empty, create it based on tables_to_query
            for table in tables:
                global schema
                database_name, schema, table_name = table[0], table[1], table[2]
                # Fetch the table context and save it to the metadata
                table_context = await get_table_context(database_name, schema, table_name)
                metadata[f"{schema}.{table_name}"] = table_context
                save_metadata(metadata)  # Save the updated metadata to the file
        
        # If metadata exists, work with filtered table names
        # print('----------------+++++++++++++++++++++++++------------')
        filtered_tables = await filter_top_tables_based_on_query(user_query=user_message)
        print('Filtered tables:', filtered_tables)      
        if not filtered_tables:
            raise HTTPException(status_code=400, detail="No relevant tables found for the query.")

        # For each filtered table, check if it has metadata, if so, append it to table_contexts
        # for filtered_table in filtered_tables:

        filtered_tables_json = json.loads(filtered_tables)
        print('after loading json object')
        print(filtered_tables_json)


        for key, table in filtered_tables_json.items():
            final_table =filtered_tables_json[key]
            check_val = f'{schema_name}.{final_table}'
            if check_val in metadata.keys():
                table_context = metadata[f'{schema_name}.{final_table}']
                table_contexts.append(table_context)

        for key, val in filtered_tables_json.items():
            if len(val) == 1:
                tabel_ = val[0]
                check_val = f'{schema_name}.{tabel_}'
                if check_val in metadata.keys():
                    table_context = metadata[f'{schema_name}.{tabel_}']
                    table_contexts.append(table_context)
            else:
                for table in val:
                    check_val = f'{schema_name}.{table}'
                    if check_val in metadata.keys():
                        table_context = metadata[f'{schema_name}.{table}']
                        table_contexts.append(table_context)
    
        # Combine table contexts into one string for the system prompt
        print(table_contexts)

        # add additional information of rep and manager:
        table_name_s = filtered_tables_json.get('table_name_key')
        # print('-------------table-name-s---------------')

        get_mail_data, name_s = final_filteration(email=user_email, table_name=table_name_s)

        print('get-mail-data', get_mail_data)

        # print('------------get-mail-data-check')
        # print(get_mail_data, name_s)
        #table_context = '\n'.join(table_contexts)

        system_prompt = get_system_prompt(table_contexts)
        
        # Step 5: Call OpenAI API to process the query and generate SQL
        print(system_prompt)
        response = azure_openai_call(system_prompt, user_message)
        # Try to extract SQL from the response
        sql_match = re.search(r"sql\n([\s\S]*?)\n", response)
        
        if sql_match:
    
            sql_query = sql_match.group(1)
            print("SQL Query Generated:", sql_query)

            # Execute the SQL query on Snowflake
            try:
                result = await query_mssql(sql_query)
            except Exception as e:
                text_resp = f'please try again :: {e}'
                return [0, text_resp]
            # print('before rounding off 2 decimal')
            # print(result)

            #update the result means round of 2 decimal only.
            f_result = round_numeric_values(data = result)

            # print('after rounding off 2 decimal')
            print(f_result)

            # implement modification of ECPs name: in bold form:
            # Step 6: Call OpenAI to generate human-readable response from the SQL result
            text_res = azure_openai_call(
                f"Give human touch to the answer always and give space between each pointer. "
                # f"Here is the output from the SQL query for the question-{user_message}, "
                f"give the answer in natural language for the question and refrain from stating the mail id that is being used to filter. "
                f"Say 'NO data found' if the SQL query returns nothing."
                f"Whenever a question is asked with some time period mentioned in it, be sure to reflect it in the response as well."
                f"If a question is asked related to growth, do not generate response with words like rate as we are only using numbers to specify growth and not percentage."
                f"When you get time period Wrap it inside the <b>"
                f"When you get Name and Decimal('') wrap it inside the <b>"
                f"don't mention email ID of the user in response",
                # f"Give output in bullet points whenever more than one value is in output",
                
                str(f_result)
            )
                #f"whenever asked about IC attainment, NBEx adherence, % calls to target, convert the numbers to percentage and add the symbol as well.",
                # f"When you get ECPs name Wrap it inside the <b>"

            # Dumping the data into the paas database:
            # DataBase().insert_chat_data(UserId=user_email, user_question=user_message, sql_query=sql_query, sql_answer=f_result)
            # if len(f_result) == 0:
            #     return HTTPException(status_code=200, detail='Sorry I could not found data.')
            if len(f_result)== 0:
                text_rep = 'no data found.'
                return [0, text_rep]
            else:
                return [f_result, text_res]

        else:
            
            # raise HTTPException(status_code=400, detail="No SQL query found in the response.")
            txt_mg ='Sorry, I do not have the info you requested'
            return [0, txt_mg]

    except Exception as e:

        raise HTTPException(status_code=500, detail=str(e))
    

def get_goals_table_prompt(manager_name):

   try:
       if manager_name == 'National Manager':
           return "ROSTER.MANAGERS_NATION_BAUSCH_EMAIL and GOALS.BAUSCH_EMAIL"
       elif manager_name == 'Regional Manager':
           return "ROSTER.MANAGERS_REGION_BAUSCH_EMAIL and GOALS.BAUSCH_EMAIL"
       else:
           return "ROSTER.MANAGERS_DISTRICT_BAUSCH_EMAIL and GOALS.BAUSCH_EMAIL"
        
   except Exception as e:
       print(f'error is coming from get goals table prompt :: {e}')



from decimal import Decimal

def round_numeric_values(data):
    updated_data = []
    
    for tuple_item in data:
        updated_tuple = tuple(
            round(value, 2) if isinstance(value, (int, float, Decimal)) else value
            for value in tuple_item
        )
        
        updated_data.append(updated_tuple)
    return updated_data