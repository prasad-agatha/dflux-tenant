from django.http import Http404
from django.shortcuts import get_object_or_404

from rest_framework import status
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework.serializers import ValidationError

from .filters import ConnectionFilter
from .query import connection_establishment
from .big_query import check_big_query_connection
from .snowflake_query_execution import test_snowflake_connection
from .permissions import (
    ProjectOwnerOrCollaborator,
    ProjectConnectionAccess,
    ProjectModuleAccess,
)

from dflux.api.views.utils import create_database_string
from dflux.api.serializers import ConnectionSerializer
from dflux.db.models import Connection, Project
from dflux.api.views.base import BaseAPIView


class ConnectionView(BaseAPIView):
    """
    API endpoint that allows view list of all the connections or create new connection.
    * Requires JWT authentication.
    * Only collaborator or owner of the project can access.
    * This endpoint will allows only GET, POST methods.
    """

    permission_classes = (IsAuthenticated, ProjectModuleAccess)

    def get_project(self, pk):
        try:
            project = Project.objects.get(id=pk)
            return project
        except:
            return ValidationError({"error": "please provide valid project id"})

    def get(self, request, pk):
        """
        View list of all the connections.
        """
        queryset = ConnectionFilter(
            request.GET,
            queryset=Connection.objects.select_related("project").filter(
                project__id=pk
            ),
        ).qs
        serializer = ConnectionSerializer(queryset, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def post(self, request, pk):
        """
        Create new database connection.
        """
        project = self.get_project(pk)
        request.data["project"] = project.id
        serializer = ConnectionSerializer(data=request.data)
        if serializer.is_valid(raise_exception=True):
            serializer.save()
            return Response(serializer.data)


class TestConnectionView(BaseAPIView):
    """
    API endpoint that allows check the given db connection details correct or wrong.
    - If connection details correct establish the connection else raise an error.
    * Authentication not required.
    * This endpoint will allows only GET, POST methods.
    """

    def post(self, request):

        db = create_database_string(request)
        try:
            conn = connection_establishment(db)
            return Response(
                {"message": "Database Connection Established successfully"},
                status=status.HTTP_200_OK,
            )
        except Exception as e:
            return Response(
                {"error": str(e)},
                status=status.HTTP_400_BAD_REQUEST,
            )


class ConnectionDetailView(BaseAPIView):
    """
    API endpoint that allows view, update, delete individual connection details.
    * Requires JWT authentication.
    * This endpoint will allows only GET, PUT, DELETE methods.
    """

    permission_classes = (IsAuthenticated, ProjectModuleAccess)

    def get_object(self, pk):
        """
        Return connection object if pk value present.
        """
        try:
            return Connection.objects.select_related("project").get(pk=pk)
        except Connection.DoesNotExist:
            raise Http404

    def get(self, request, pk, format=None):
        """
        Get individual connection details.
        """
        connection = self.get_object(pk)

        serializer = ConnectionSerializer(connection)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def put(self, request, pk, format=None):
        """
        Update individual connection details.
        """
        connection = self.get_object(pk)
        get_object_or_404(Project, id=connection.project.id)
        serializer = ConnectionSerializer(connection, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, pk, format=None):
        """
        Delete individual connection details.
        """
        connection = self.get_object(pk)
        connection.delete()
        return Response({"message": "Delete Success"}, status=status.HTTP_200_OK)


class TestSnowflakeConnectionView(BaseAPIView):
    """
    API endpoint that allows given db connection details correct or wrong.
    - If db credentials correct establish the snowflake connection else raise an error
    * Authentication not required.
    * This endpoint will allows only GET method.
    """

    def post(self, request):
        """
        Check the the given db connection details correct or wrong.
        """
        try:
            conn = test_snowflake_connection(request)
            return Response(
                {"message": "Database Connection Established successfully"},
                status=status.HTTP_200_OK,
            )
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)


class TestBigQueryConnectionView(BaseAPIView):
    """
    API endpoint that allows given db connection details correct or wrong.
    - If db credentials correct establish the BigQuery connection else raise an error
    * Authentication not required.
    * This endpoint will allows only GET method.
    """

    def post(self, request):
        """
        Check the the given bigquery connection details correct or wrong.
        """
        try:
            check_big_query_connection(request.data.get("credentials_path"))
            return Response(
                {"message": "Database Connection Established successfully"},
                status=status.HTTP_200_OK,
            )
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
