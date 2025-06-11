from google.cloud import bigquery
from google.oauth2 import service_account


def check_big_query_connection(credential_path):
    """
    Establish the google bigquery connection
    """

    import requests
    import json

    # user have already credentials file
    content = requests.get(credential_path)
    dict_data = json.loads(content.content)
    credentials = service_account.Credentials.from_service_account_info(dict_data)
    client = bigquery.Client(credentials=credentials)
    return client
