from rest_framework import status
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated

from django.shortcuts import get_object_or_404

from dflux.db.models import Project, DataModel
from dflux.api.views.base import BaseAPIView
from dflux.api.serializers import DataModelSerializer, RequiredFieldsDataModelSerializer

from .filters import DataModelFilter
from .permissions import (
    ProjectOwnerOrCollaborator,
    ProjectDataModelAccess,
    ProjectModuleAccess,
)


class DataModelLimitedView(BaseAPIView):
    """Create new data model and get all the data models"""

    permission_classes = [ProjectModuleAccess]

    def get(self, request, pk):
        data_models = DataModelFilter(
            request.GET,
            queryset=DataModel.objects.select_related("project", "meta_data").filter(
                project__id=pk
            ),
        ).qs
        serializer = RequiredFieldsDataModelSerializer(data_models, many=True)
        return Response(serializer.data)


class DataModelView(BaseAPIView):
    """
    API endpoint that allows view list of all the data models or create new data model.
    - Data models stores data science models(like classification, regression etc) data.

    * Requires JWT authentication.
    * Only collaborator or owner of the project can access.
    * This endpoint will allows only GET, POST methods.
    """

    permission_classes = [IsAuthenticated, ProjectModuleAccess]

    def get(self, request, pk):
        """
        View list of all the data models.
        """
        data_models = DataModelFilter(
            request.GET,
            queryset=DataModel.objects.select_related("project", "meta_data").filter(
                project__id=pk
            ),
        ).qs
        serializer = DataModelSerializer(data_models, many=True)
        return Response(serializer.data)

    def post(self, request, pk):
        """
        Create new data model.
        """
        project = get_object_or_404(Project, id=pk)
        request.data["project"] = project.id
        serializer = DataModelSerializer(data=request.data)
        if serializer.is_valid(raise_exception=True):
            serializer.save()
            return Response(serializer.data)


class DataModeDetailView(BaseAPIView):
    """
    API endpoint that allows view, update, delete individual data model details.

    * Requires JWT authentication.
    * Only collaborator or owner of the project can access.
    * This endpoint will allows only GET, PUT, DELETE methods.
    """

    permission_classes = (IsAuthenticated, ProjectDataModelAccess)

    def get(self, request, project_id, pk):
        """
        Get individual data model details.
        """
        data_model = get_object_or_404(DataModel, project_id=project_id, id=pk)
        serializer = DataModelSerializer(data_model)
        return Response(serializer.data)

    def put(self, request, project_id, pk):
        """
        Update individual data model details.
        """
        data_model = get_object_or_404(DataModel, id=pk)
        serializer = DataModelSerializer(data_model, data=request.data, partial=True)
        if serializer.is_valid(raise_exception=True):
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.data)

    def delete(self, request, project_id, pk):
        """
        Delete individual data model details.
        """
        data_model = get_object_or_404(DataModel, id=pk)
        data_model.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)
