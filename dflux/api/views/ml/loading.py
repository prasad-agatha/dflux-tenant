import pandas as pd

from rest_framework import status
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated

from django.db import connection as db_connection

from dflux.db.models import Connection

from dflux.api.views.base import BaseAPIView
from dflux.api.views.query import connection_establishment
from dflux.api.views.ml.auto_ml import read_data, read_data_from_db


class LoadCSVData(BaseAPIView):
    """
    API endpoint that allows parse the csv file returns data into the Json format.

    * Authentication not required.
    * This endpoint will allows only GET method.
    """

    permission_classes = (IsAuthenticated,)

    def post(self, request):
        csv_file = request.FILES.get("csv_file")
        # convert csv file data into df
        df = read_data(csv_file)
        # print(df)
        return Response(df)


class LoadDBData(BaseAPIView):
    """
    API endpoint that allows return the json based user raws SQL query.
    -If user try to execute wrong SQL query it will raise an error.

    * Requires JWT authentication.
    * This endpoint will allows only POST method.
    """

    permission_classes = (IsAuthenticated,)

    def post(self, request):

        connection_id = request.data.get("connection_id")
        try:
            db = Connection.objects.get(id=connection_id)

        except:
            return Response({"error": "Please submit a valid connection_id"})

        if db.connection_type == "INTERNAL":
            conn = db_connection.cursor()
            try:
                cursor = db_connection.cursor()
                query = request.data.get("sql_query")
                cursor.execute(query)
                columns = [column[0] for column in cursor.description]
                results = []
                for row in cursor.fetchall():
                    results.append(dict(zip(columns, row)))
                df = pd.DataFrame(results)
                return Response(df)
            except Exception as e:
                return Response(
                    {"error": str(e)},
                    status=status.HTTP_400_BAD_REQUEST,
                )
        else:
            conn = connection_establishment(db)
            # database connection establishment
            try:
                query = request.data.get("sql_query")
                # converting sql query data into df
                df = read_data_from_db(query, conn)
                # print(df)
                return Response(df)
            except Exception as e:
                return Response(
                    {"error": str(e)},
                    status=status.HTTP_400_BAD_REQUEST,
                )
