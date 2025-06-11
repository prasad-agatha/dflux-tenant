import snowflake.connector


def test_snowflake_connection(request):
    """
    This method will allows return the snowflake connection object for given db.
    """
    conn = snowflake.connector.connect(
        user=request.data.get("username"),
        password=request.data.get("password"),
        account=request.data.get("account"),
        warehouse=request.data.get("warehouse"),
        database=request.data.get("dbname"),
        schema=request.data.get("schema"),
    )
    return conn


def execute_query_in_snowflake(
    user=None,
    password=None,
    account=None,
    warehouse=None,
    database=None,
    schema=None,
    query=None,
):
    """
    This method will allow execute the raw sql query and returns the results in the list of objects.
    """

    conn = snowflake.connector.connect(
        user=user,
        password=password,
        account=account,
        warehouse=warehouse,
        database=database,
        schema=schema,
    )

    cursor = conn.cursor()
    query = query
    data = cursor.execute(query)

    columns = [column[0] for column in cursor.description]
    results = []
    for row in cursor.fetchall():
        results.append(dict(zip(columns, row)))

    return results
