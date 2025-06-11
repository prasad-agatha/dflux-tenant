import pyodbc


class CursorByName:
    """
    This method will allow loop the cursor objects results.
    """

    def __init__(self, cursor):
        self._cursor = cursor

    def __iter__(self):
        return self

    def __next__(self):
        row = self._cursor.__next__()

        return {
            description[0]: row[col]
            for col, description in enumerate(self._cursor.description)
        }


async def query_invoke(cursor, sql_string):
    """
    This method will allow execute the raw sql query and returns the results in the list of objects.
    * required params:
    ------------------
    - cursor
    - SQL Query
    """

    cursor = cursor
    cursor.execute(sql_string)
    rows = []
    for row in CursorByName(cursor):
        rows.append(row)
    return rows


drivers = {
    "postgres": "{Devart ODBC Driver for PostgreSQL}",
    "mysql": "{MySQL ODBC 8.0 Unicode Driver}",
    "mssql": "{ODBC Driver 17 for SQL Server}",
    "oracle": "{Devart ODBC Driver for Oracle}",
}


# print("Python ODBC Drivers:", pyodbc.drivers())


def create_connection_string(driver, database, username, password, server, port):
    """
    This method will allows create database connection string with required details.
    * required fields:
    ----------------
    - driver
    - database
    - username
    - password
    - server
    - port
    """
    conn_str = (
        f"DRIVER={driver};"
        f"DATABASE={database};"
        f"UID={username};"
        f"PWD={password};"
        f"SERVER={server};"
        f"PORT={port};"
    )
    return conn_str


def connection_establishment(db):
    """
    This method will allows return the connection object for given database object.
    """
    # print("Type of DB object:", type(db))
    if type(db) == dict:
        driver = drivers.get(db.get("engine", None))
        conn_str = create_connection_string(
            drivers.get(db.get("engine", None)),
            db.get("dbname", None),
            db.get("username", None),
            db.get("password", None),
            db.get("host", None),
            db.get("port", None),
        )
        conn = pyodbc.connect(
            conn_str,
            Direct=True if driver == "{Devart ODBC Driver for Oracle}" else None,
            sslmode="allow"
            if driver == "{Devart ODBC Driver for PostgreSQL}"
            else None,
        )
        # cursor = conn.cursor()
        return conn

    else:
        driver = drivers.get(db.engine)

        conn_str = create_connection_string(
            driver, db.dbname, db.username, db.password, db.host, db.port
        )
        # print(conn_str, "conn_str")
        conn = pyodbc.connect(
            conn_str,
            Direct=True if driver == "{Devart ODBC Driver for Oracle}" else None,
            sslmode="allow"
            if driver == "{Devart ODBC Driver for PostgreSQL}"
            else None,
        )
        # print(conn, "conn")
        # cursor = conn.cursor()
        return conn
