from rest_framework import status
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated

from django.shortcuts import Http404
from django.shortcuts import get_object_or_404


from dflux.api.serializers import (
    ChartSerializer,
    SaveChartSerializer,
    ShareChartSerializer,
    LimitedFieldsChartSerializer,
)

from dflux.api.views.base import BaseAPIView
from dflux.utils.emails import emails
from dflux.db.models import Charts, Project, MediaAsset, ShareCharts

from decouple import config

from .filters import ChartsFilter
from .permissions import (
    ProjectOwnerOrCollaborator,
    ProjectChartAccess,
    ProjectModuleAccess,
)


class ChartsLimitedView(BaseAPIView):
    permission_classes = (IsAuthenticated, ProjectModuleAccess)

    def get(self, request, pk):
        """
        Return a list of all Charts.
        """
        query_charts = ChartsFilter(
            request.GET,
            queryset=Charts.objects.select_related(
                "project", "user", "query", "data_model"
            ).filter(project__id=pk),
        ).qs
        serializer = LimitedFieldsChartSerializer(query_charts, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)


class ChartsView(BaseAPIView):
    """
    API endpoint that allows view list of all the charts or create new chart.

    * Requires JWT authentication.
    * Only collaborator or owner of the project can access.
    * This endpoint will allows only GET, POST methods.
    """

    permission_classes = (IsAuthenticated, ProjectModuleAccess)

    def get(self, request, pk):
        """
        Return a list of all Charts.
        """
        query_charts = ChartsFilter(
            request.GET,
            queryset=Charts.objects.select_related(
                "project", "user", "query", "data_model"
            ).filter(project__id=pk),
        ).qs
        serializer = ChartSerializer(query_charts, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def post(self, request, pk):
        """
        Create a new Chart.
        """
        project = Project.objects.get(id=pk)
        request.data["project"] = project.id
        request.data["user"] = request.user.id
        serializer = SaveChartSerializer(data=request.data)
        if serializer.is_valid(raise_exception=True):
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)


class ChartsDetailView(BaseAPIView):
    """
    API endpoint that allows view, update, delete individual chart details.

    * Requires JWT authentication.
    * Only collaborator or owner of the project can access.
    * This endpoint will allows only GET, PUT, DELETE methods.
    """

    permission_classes = (IsAuthenticated, ProjectChartAccess)

    def get_object(self, pk):
        """
        Return Chart object if pk value present.
        """
        try:
            return Charts.objects.select_related(
                "project", "user", "query", "data_model"
            ).get(pk=pk)
        except Charts.DoesNotExist:
            raise Http404

    def get(self, request, pk, format=None):
        """
        Return Chart.
        """
        chart = self.get_object(pk)
        serializer = ChartSerializer(chart)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def put(self, request, pk, format=None):
        """
        Update the chart details
        """
        chart = self.get_object(pk)
        if request.data.get("extra") is not None:
            media = MediaAsset.objects.filter(
                id=request.data.get("extra").get("thumbnail").get("previous_id")
            ).first()
            if media is not None:
                media.delete()
        serializer = SaveChartSerializer(chart, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, pk, format=None):
        """
        Delete chart.
        """
        chart = self.get_object(pk)
        chart.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


class ShareChartView(BaseAPIView):
    """
    API endpoint that allows share the chart to users.

    * Authentication not required.
    * This endpoint will allows only GET method.
    """

    permission_classes = (IsAuthenticated,)

    def get(self, request, pk):
        import uuid

        token = uuid.uuid1().hex
        chart = get_object_or_404(Charts, id=pk)
        request.data["token"] = token
        request.data["charts"] = chart.id
        serializer = ShareChartSerializer(data=request.data)
        if serializer.is_valid(raise_exception=True):
            serializer.save()
            return Response({"shared_token": token})


class LookSharedChart(BaseAPIView):
    """
    API endpoint that allows view the shared chart to users.

    * Authentication not required.
    * This endpoint will allows only GET method.
    """

    def get(self, request):
        chart_token = request.query_params.get("token")
        chart = get_object_or_404(ShareCharts, token=chart_token)

        serializer = ChartSerializer(chart.charts)
        return Response(serializer.data, status=status.HTTP_200_OK)


class SendChartEmail(BaseAPIView):
    """
    API endpoint that allows send shared chart email to users.

    * Requires JWT authentication.
    * This endpoint will allows only POST method.
    """

    permission_classes = (IsAuthenticated,)

    def post(self, request):

        url = request.data.get("url")
        emails.send_chart_email(request, url)
        return Response({"message": "email sent"})
