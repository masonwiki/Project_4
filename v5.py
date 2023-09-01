# Import dependencies 
import psycopg2
import json
from datetime import datetime

# Database connection parameters
db_params = {
    "dbname": "Project_4",
    "user": "postgres",
    "password": "postgres",
    "host": "localhost"
}

# Timeframe of the data we are working with
timeframe = '2015-01'

# Empty list that adds data and injects it into sql all at once
sql_transaction = []

# Subreddits to include
desired_subreddits = [
    "stocks", "dividends", "Entrepreneur", "investing",
    "Superstonk", "options", "personalfinance", "Economics",
    "Forex", "StockMarket", "CryptoCurrency", "Trading",
    "Investor", "algotrading", "Daytrading", "Etoro",
    "stocktraders", "Economics", "ValueInvesting"
]

# Establish a connection
connection = psycopg2.connect(**db_params)
c = connection.cursor()

# Prepare the SQL query template
def create_table():
    c.execute("CREATE TABLE IF NOT EXISTS public.parent_reply (parent_id TEXT PRIMARY KEY, comment_id TEXT UNIQUE, parent TEXT, comment TEXT, subreddit TEXT, unix INT, score INT);")

# Remove user and reply tahs
def clean_and_escape(data):
    data = data.replace("\n", " newlinechar ").replace("\r", " newlinechar ").replace('"', "'")
    return data

# Here we are making sure the body of the replies are not too short/long or don't consist of deleted or removed characters.
def is_valid_comment(data):
    if len(data.split(' ')) > 50 or len(data) < 1:
        return False
    elif len(data) > 1000:
        return False
    elif data == '[deleted]' or data == '[removed]':
        return False
    else: 
        return True

# Here we are defining a function that will find a comment_id's parent_id    
def find_parent(pid):
    try:
        sql = "SELECT comment FROM parent_reply WHERE comment_id = %s LIMIT 1"
        c.execute(sql, (pid,))
        result = c.fetchone()
        if result is not None:
            return result[0]
        else:
            return False
    except psycopg2.Error as e:
        print("Error executing SQL query:", str(e))
        return False
    
def find_existing_score(pid):
    try:
        sql = "SELECT score FROM parent_reply WHERE parent_id = %s LIMIT 1"
        c.execute(sql, (pid,))
        result = c.fetchone()
        if result is not None:
            return result[0]
        else: return False
    except Exception as e:
        return False
    


def transaction_bldr(sql, values=None):
    global sql_transaction
    if values is None:
        sql_transaction.append(sql)
    else:
        # Check if the parent_id already exists in the transaction
        parent_id = values[0] if values else None
        if any(p[0] == parent_id for p in sql_transaction):
            return  # Skip inserting if parent_id already exists

        sql_transaction.append((sql, values))
    
    if len(sql_transaction) > 1000:
        try:
            c.execute("BEGIN TRANSACTION")
            for s in sql_transaction:
                if isinstance(s, tuple):
                    c.execute(s[0], s[1])
                else:
                    c.execute(s)
            connection.commit()
            sql_transaction = []
        except Exception as e:
            print("Error starting or committing transaction:", str(e))
            connection.rollback()  # Roll back the transaction if an error occurred
        finally:
            sql_transaction = []  # Clear the transaction list in case of success or failure


def sql_insert_replace_comment(commentid, parentid, parent, comment, subreddit, time, score):
    try:
        sql = """UPDATE parent_reply 
                 SET comment_id = %s, parent = %s, comment = %s, subreddit = %s, unix = %s, score = %s 
                 WHERE parent_id = %s AND comment_id = %s;"""
        values = (commentid, parent, comment, subreddit, time, score, parentid, commentid)
        transaction_bldr(sql, values)
    except Exception as e:
        print('s-UPDATE insertion', str(e))

def sql_insert_has_parent(commentid, parentid, parent, comment, subreddit, time, score):
    try:
        sql = """INSERT INTO parent_reply (parent_id, comment_id, parent, comment, subreddit, unix, score) 
                 VALUES (%s, %s, %s, %s, %s, %s, %s);"""
        values = (parentid, commentid, parent, comment, subreddit, time, score)
        transaction_bldr(sql, values)
    except Exception as e:
        print('s-PARENT insertion', str(e))

def sql_insert_no_parent(commentid, parentid, comment, subreddit, time, score):
    try:
        sql = """INSERT INTO parent_reply (parent_id, comment_id, comment, subreddit, unix, score) 
                 VALUES (%s, %s, %s, %s, %s, %s);"""
        values = (parentid, commentid, comment, subreddit, time, score)
        transaction_bldr(sql, values)
    except Exception as e:
        print('s-NO_PARENT insertion', str(e))

if __name__ == "__main__":
    create_table()
    row_counter = 0
    paired_rows = 0

    # Load in data
    with open(f"D:/Users/Mason/Reddit_Comments/RC_{timeframe}.txt", buffering=1000) as file:
        for row in file:
            row_counter += 1
            row = json.loads(row)
            parent_id = row['parent_id']
            body = clean_and_escape(row['body'])
            created_utc = row['created_utc']
            score = row['score']
            comment_id = row['name']
            subreddit = row['subreddit']
            parent_data = find_parent(parent_id)
            if score >= 2:
                if subreddit in desired_subreddits:               
                    existing_comment_score = find_existing_score(parent_id)
                    if existing_comment_score:
                        if score > existing_comment_score:
                            if is_valid_comment(body):
                                sql_insert_replace_comment(comment_id,parent_id,parent_data,body,subreddit,created_utc,score)
                            
                    else:
                        if is_valid_comment(body):
                            if parent_data:
                                sql_insert_has_parent(comment_id,parent_id,parent_data,body,subreddit,created_utc,score)
                                paired_rows += 1
                            else:
                                sql_insert_no_parent(comment_id,parent_id,body,subreddit,created_utc,score)

                if row_counter % 100000 == 0:
                    print(f"Total rows read: {row_counter}, Paired rows: {paired_rows}, Time: {str(datetime.now())}")

    c.close()
    connection.commit()  # Commit any remaining transactions
    connection.close()