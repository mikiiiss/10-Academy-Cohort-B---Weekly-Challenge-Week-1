# my_package/db_connection.py
import os
from dotenv import load_dotenv

def connect_to_database():
    load_dotenv()  # Load environment variables from .env file
    
    # Get database connection parameters from environment variables
    db_host = os.getenv("DB_HOST")
    db_port = os.getenv("DB_PORT")
    db_name = os.getenv("DB_NAME")
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    
    # Connect to the PostgreSQL database (example using psycopg2)
    import psycopg2
    conn = psycopg2.connect(
        host=db_host,
        port=db_port,
        dbname=db_name,
        user=db_user,
        password=db_password
    )
    
    return conn
