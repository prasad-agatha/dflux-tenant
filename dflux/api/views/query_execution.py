import asyncio

from rest_framework import status
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated

from django.db import connection as db_connection
from django.shortcuts import Http404, get_object_or_404

from .filters import QueryFilter
from .big_query import check_big_query_connection
from .permissions import (
    ProjectOwnerOrCollaborator,
    ProjectQueryAccess,
    ProjectExecuteQueryAccess,
    ProjectSchemaAccess,
    ProjectModuleAccess,
)

from dflux.db.models import (
    Connection,
    Query,
    Project,
    ExcelData,
)
from dflux.api.serializers import (
    QuerySerializer,
    QueryPostSerializer,
    QueryDetailsSerializer,
)
from dflux.api.views.base import BaseAPIView
from dflux.db.models.google_big_query import GoogleBigQueryCredential
from dflux.api.views.query import connection_establishment, query_invoke

from .utils import internal_connection_schema, external_connection_schema


class ExecuteQuery(BaseAPIView):
    """
    API endpoint that allows run the raw sql query based on the connection id.

    * Requires JWT authentication.
    * Only collaborator or owner of the project can access.
    * This endpoint will allows only POST method.
    """

    permission_classes = (IsAuthenticated, ProjectExecuteQueryAccess)

    def post(self, request):

        connection_id = request.data.get("connection_id")
        connection = Connection.objects.get(pk=connection_id)
        get_object_or_404(Project, id=connection.project.id)

        try:
            db = Connection.objects.get(id=connection_id)

        except:
            return Response({"error": "Please submit a valid connection_id"})

        # database connection establishment
        try:
            if db.connection_type == "INTERNAL":
                cursor = db_connection.cursor()
                query = request.data.get("sql_query")
                cursor.execute(query)
                columns = [column[0] for column in cursor.description]
                # data = cursor.fetchall()
                results = []
                for row in cursor.fetchall():
                    results.append(dict(zip(columns, row)))
                return Response({"data": results})
            if db.connection_type == "snowflake":
                from .snowflake_query_execution import execute_query_in_snowflake

                query = request.data.get("sql_query")
                result = execute_query_in_snowflake(
                    user=db.username,
                    password=db.password,
                    account=db.account,
                    warehouse=db.warehouse,
                    database=db.dbname,
                    schema=db.schema,
                    query=query,
                )
                return Response({"data": result})

            if db.connection_type == "bigquery":
                gcp = GoogleBigQueryCredential.objects.filter(connection=db).first()
                client = check_big_query_connection(gcp.credential_path)
                # Perform a query.
                query = request.data.get("sql_query")
                query_job = client.query(query)  # API request
                result = query_job.result()  # Waits for query to finish
                df = result.to_dataframe()
                return Response(df)
            else:
                conn = connection_establishment(db)
                cursor = conn.cursor()
                query = request.data.get("sql_query")
                loop = asyncio.new_event_loop()
                result = loop.run_until_complete(query_invoke(cursor, query))
                loop.close()
                # print(result, "result")
                # if result:
                #     Query.objects.create(raw_sql=query, verify=True, accepted=True)
                return Response({"data": result})
        except Exception as e:
            return Response(
                {"error": str(e)},
                status=status.HTTP_400_BAD_REQUEST,
            )


class SQLQuery(BaseAPIView):
    """
    API endpoint that allows view list of all the queries or add new query.

    * Requires JWT authentication.
    * Only collaborator or owner of the project can access.
    * This endpoint will allows only GET, POST methods.
    """

    permission_classes = (IsAuthenticated, ProjectModuleAccess)

    def get(self, request, pk):
        """
        This method will allows view list of all the queries corresponding project.
        """
        queryset = QueryFilter(
            request.GET,
            queryset=Query.objects.select_related(
                "user", "project", "connection", "excel", "json"
            ).filter(project__id=pk),
        ).qs
        serializer = QuerySerializer(queryset, many=True)
        return Response(serializer.data)

    def post(self, request, pk):
        """
        This method will allows add new queries corresponding project.
        """

        request.data["project"] = pk
        request.data["user"] = request.user.id
        serializer = QueryPostSerializer(data=request.data)
        if serializer.is_valid(raise_exception=True):
            serializer.save(accepted=True, verify=True)
            return Response(serializer.data, status=status.HTTP_201_CREATED)


class QueryDetail(BaseAPIView):
    """
    API endpoint that allows view, update, delete individual query details.

    * Requires JWT authentication.
    * Only collaborator or owner of the project can access.
    * This endpoint will allows only GET, PUT, DELETE methods.
    """

    permission_classes = (IsAuthenticated, ProjectQueryAccess)

    def get_object(self, pk):
        """
        This method will return Query object give pk value.
        """
        try:
            return Query.objects.select_related(
                "user", "project", "connection", "excel", "json"
            ).get(pk=pk)
        except Query.DoesNotExist:
            raise Http404

    def get(self, request, pk):
        """
        This method will allows get the individual query details.
        """
        query = self.get_object(pk)
        get_object_or_404(Project, id=query.project.id)
        serializer = QueryDetailsSerializer(query)
        return Response(serializer.data)

    def put(self, request, pk):
        """
        This method will allows update the individual query details.
        """
        query = self.get_object(pk)
        get_object_or_404(Project, id=query.project.id)
        request.data["user"] = request.user.id
        serializer = QueryPostSerializer(query, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, pk):
        """
        This method will allows delete the individual query details.
        """
        query = self.get_object(pk)
        get_object_or_404(Project, id=query.project.id)
        query.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


class Schematable(BaseAPIView):
    """
    API endpoint that allows view the schema of given connection.

    * Requires JWT authentication.
    * Only collaborator or owner of the project can access.
    * This endpoint will allows only GET, PUT, DELETE methods.
    """

    permission_classes = (IsAuthenticated, ProjectModuleAccess)

    def get(self, request, pk):
        try:
            db = Connection.objects.get(id=pk)
            get_object_or_404(Project, id=db.project.id)
        except:
            return Response(
                {"error": "Please give valid connection id"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            if db.connection_type == "INTERNAL":
                results = internal_connection_schema(request, db)
                return Response({"data": results})
            else:
                results = external_connection_schema(db)
                return Response({"data": results})

        except Exception as e:
            return Response({"msg": str(e)}, status=status.HTTP_400_BAD_REQUEST)
