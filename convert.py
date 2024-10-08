#convert csv to sql file and create DB
from sqlalchemy import create_engine, Table, Column, Integer, String, MetaData

# Create an SQLite database in memory
engine = create_engine('sqlite:///:database.db:')

# Create a metadata instance
metadata = MetaData()

import glob

csvfiles = []
for file in glob.glob("/content/drive/MyDrive/Python assignment/Dataset2/*.csv"): #Path to csv's
    csvfiles.append(file)
    csv_path = file
    df = pd.read_csv(csv_path)

    # Replace with your desired database URL and table name
    database_url = "sqlite:///database.db"
    table_name = csv_path.split("/")[-1].split(".")[0]+"_table"
    print(table_name)

    # Create an SQLAlchemy engine
    engine = create_engine(database_url)
    df.to_sql(table_name, engine, index=False, if_exists="replace")
