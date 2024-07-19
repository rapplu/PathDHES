import sqlite3
import os

def initialize_database(db_file):
    """
    Initialize a database db_file.
    """
    # Check if the database file exists.
    if os.path.exists(db_file):
        try:
            # Delete the existing database file.
            os.remove(db_file)
            print(f"Existing database '{db_file}' deleted successfully.")
        except OSError as e:
            print(f"Error deleting existing database '{db_file}': {e}")

    # Connect to SQLite database (creates a new one if the file doesn't exist).
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Create the 'byte_data' table.
    cursor.execute('''CREATE TABLE IF NOT EXISTS byte_data (
                        key TEXT,
                        bytes BLOB
                      )''')
    
    print("Table 'byte_data' created successfully.")

    conn.commit()
    return conn


def write_dict_to_sqlite(data_dict, conn):
    """
    Write dictionary to a SQLite database.
    """
    cursor = conn.cursor()
    for label, byte_list in data_dict.items():
        for byte_data in byte_list:
            # Insert the new label-value pair.
            cursor.execute("INSERT INTO byte_data (key, bytes) VALUES (?, ?)", (sqlite3.Binary(label), sqlite3.Binary(byte_data)))

    # Commit changes. 
    conn.commit()


def get_values_for_label(conn, label, chunk_size=1000):
    """
    Lookup all values associated with label.
    """
    cursor = conn.cursor()
    cursor.execute("SELECT bytes FROM byte_data WHERE key = ?", (sqlite3.Binary(label),))
    
    while True:

        rows = cursor.fetchmany(chunk_size)
        if not rows:
            break
        for row in rows:
            yield row[0]
            if row[0] == label:
                return


def read_data_streaming(conn_read, chunk_size):
    """
    Read database in chunks of size chunk_size in a streaming manner.
    """
    cursor_read = conn_read.cursor()
    cursor_read.execute("SELECT * FROM byte_data")

    while True:
        rows = cursor_read.fetchmany(chunk_size)
        if not rows:
            break
        yield rows


def get_row_count(conn_read):
    """
    Return number of rows in database.
    """
    cursor_read = conn_read.cursor()
    cursor_read.execute("SELECT COUNT (*) FROM byte_data")
    total_chunks = cursor_read.fetchone()[0]
    return total_chunks