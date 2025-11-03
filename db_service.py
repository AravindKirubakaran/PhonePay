import pandas as pd
from sqlalchemy import create_engine
import json

with open("config.json", 'r') as f:
    config_data = json.load(f)

MYSQL_USER = config_data["db_username"]
MYSQL_PASS = config_data["db_password"]
MYSQL_HOST = config_data["db_host"]
MYSQL_PORT = config_data["db_port"]
MYSQL_DB = config_data["db_name"]


def GetData(TableName):

    DATABASE_URL = f"mysql+mysqlconnector://{MYSQL_USER}:{MYSQL_PASS}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"

    # Create the SQLAlchemy Engine
    try:
        # **CHANGE 2: Create the engine using the new MySQL URL**
        engine = create_engine(DATABASE_URL)
    except Exception as e:
        print(f"❌ Could not create the database engine: {e}")
        exit()

    try:
        get_data = pd.read_sql(f"SELECT * FROM {TableName}", con=engine)
        print("\nData retrieved from database for verification:")
        print(get_data)

    except Exception as e:
        print(f"❌ An error occurred during database operation: {e}")

    return get_data

