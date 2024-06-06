import os
import asyncio
import pandas as pd
import duckdb
from dotenv import load_dotenv, find_dotenv
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
dotenv_path = r'C:\Users\Arief\Videos\database\groq.env' 
if not os.path.exists(dotenv_path):
    raise FileNotFoundError(f"No .env file found at {dotenv_path}.")
load_dotenv(dotenv_path)

# Define the category mapping
category_mapping = {
    1: 'Uang Keluar',
    2: 'Tabungan & Investasi',
    3: 'Pinjaman',
    4: 'Tagihan',
    5: 'Hadiah & Amal',
    6: 'Transportasi',
    7: 'Belanja',
    8: 'Top Up',
    9: 'Hiburan',
    10: 'Makanan & Minuman',
    11: 'Biaya & Lainnya',
    12: 'Hobi & Gaya Hidup',
    13: 'Perawatan Diri',
    14: 'Kesehatan',
    15: 'Pendidikan',
    16: 'Uang Masuk',
    17: 'Gaji',
    18: 'Pencairan Investasi',
    19: 'Bunga',
    20: 'Refund',
    21: 'Pencairan Pinjaman',
    22: 'Cashback'
}

# Define the prompt template
def create_prompt(question, context):
    return f"""
    You are a powerful text-to-SQL model. Your job is to answer questions about a database. You are given a question and context regarding one or more tables. Don't add newline characters.

    You must output the SQL query that answers the question in a single line.

    ### Input:
    {question}

    ### Context:
    {context}

    ### Response:
    """

# Predefined mapping of email addresses to account numbers
email_to_account = {
    "hjgiadw@gmail.com": "1000291185",
    "example2@gmail.com": "1000693586",
    "example3@gmail.com": "1002623599"
}

def get_account_number(email):
    """Retrieves the account number based on the email address."""
    return email_to_account.get(email)

async def groq_infer(question, api_key):
    """Gets an SQL query from the Groq model."""
    # Provide detailed schema information
    context = """
    The 'hakathon' table includes the following fields:
    - id (integer): The unique identifier for each transaction.
    - account_number (string): The account number associated with the transaction.
    - type (string): The type of transaction (e.g., "Pembayaran", "Transfer").
    - transaction (string): Details about the transaction.
    - amount (numeric): The amount of the transaction.
    - debit_credit (string): Indicates if the transaction is a debit or credit.
    - merchant_code (string): The merchant code associated with the transaction.
    - subheader (string): Additional information about the transaction.
    - detail_information (string): Detailed information about the transaction.
    - trx_date (date): The date of the transaction.
    - created_at (timestamp): The timestamp when the record was created.
    - updated_at (timestamp): The timestamp when the record was last updated.
    - persona (string): The persona of the account holder.
    - name (string): The name associated with the transaction.
    - sub_name (string): Additional name information.
    - logo (string): The logo associated with the transaction.
    - website (string): The website associated with the transaction.
    - latitude (numeric): The latitude of the transaction location.
    - longitude (numeric): The longitude of the transaction location.
    - address (string): The address associated with the transaction.
    - category_id (integer): The category ID associated with the transaction.
    """
    prompt_text = create_prompt(question, context)
    chat = ChatGroq(api_key=api_key, model_name="mixtral-8x7b-32768", temperature=0)
    prompt = ChatPromptTemplate.from_messages([("human", prompt_text)])
    chain = LLMChain(llm=chat, prompt=prompt)
    response = await chain.ainvoke({"input": ""})

    # Extract the text from the response
    if "text" in response:
        query = response["text"]
    else:
        raise ValueError("Unexpected response format: 'text' key not found.")
    return query

def clean_sql_query(query):
    """Extracts the SQL query from the model's response and cleans it."""
    sql_start = query.find("SELECT")
    if sql_start == -1:
        raise ValueError("No valid SQL query found in the response.")
    cleaned_query = query[sql_start:].strip()
    # Remove any trailing semicolon
    if cleaned_query.endswith(";"):
        cleaned_query = cleaned_query[:-1]
    return cleaned_query

def load_duckdb(file_name):
    """Loads the DuckDB database."""
    con = duckdb.connect(database=file_name, read_only=True)
    return con

def main():
    st.set_page_config(page_title="Transaction Data Viewer", page_icon="📊", layout="wide")
    st.title("Transaction Data Viewer")

    # Check API key and manage login
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("API key not found. Please check your environment variables.")
        st.stop()

    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False

    if not st.session_state.logged_in:
        email = st.text_input("Enter your email to login")
        if st.button("Login"):
            account_number = get_account_number(email)
            if account_number:
                st.session_state['logged_in'] = True
                st.session_state['account_number'] = account_number
                st.success(f"Logged in as {email} with account number {account_number}")
            else:
                st.error("No account found for this email.")

    if st.session_state.logged_in:
        file_name = st.text_input("Enter name of a DB file")
        if st.button("Load DB"):
            try:
                con = load_duckdb(file_name)
                st.session_state['con'] = con
                st.success(f"Database {file_name} loaded successfully.")
                tables = con.execute("SHOW TABLES").fetchdf()
                st.subheader("Tables in Database")
                st.write(tables)
            except Exception as e:
                st.error(f"Error loading database: {e}")

        if 'con' in st.session_state:
            question = st.text_input("Ask a question about your transactions")
            if st.button("Get Answer"):
                try:
                    account_number = st.session_state['account_number']
                    raw_query = asyncio.run(groq_infer(question, api_key))
                    sql_query = clean_sql_query(raw_query)

                    # Ensure the query includes a WHERE clause for account_number
                    if "WHERE" not in sql_query.upper():
                        sql_query += f" WHERE account_number = '{account_number}'"
                    else:
                        sql_query += f" AND account_number = '{account_number}'"

                    # Execute query with DuckDB
                    result = st.session_state['con'].execute(sql_query).fetchdf()
                    st.dataframe(result)

                    # Map category_id to category name
                    if 'category_id' in result.columns:
                        result['category'] = result['category_id'].map(category_mapping)

                    # Display insights
                    if not result.empty:
                        st.subheader("Insights")
                        if 'category' in result.columns and 'amount' in result.columns:
                            st.write(f"The highest spending category is: {result['category'].iloc[0]} with an amount of {result['amount'].iloc[0]}")
                        else:
                            st.write("No data found for the given query.")
                    else:
                        st.write("No data found for the given query.")

                except Exception as e:
                    st.error(f"Error executing query: {e}")

        # PDF Upload
        st.subheader("Upload PDF")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        if uploaded_file is not None:
            save_path = f"./uploaded_pdfs/{uploaded_file.name}"
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"File uploaded and saved to {save_path}")

if __name__ == "__main__":
    main()
