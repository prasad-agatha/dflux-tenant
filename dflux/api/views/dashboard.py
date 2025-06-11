from rest_framework import status
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated


from django.shortcuts import Http404, get_object_or_404

from dflux.db.models import (
    DashBoard,
    ShareDashBoard,
    DashBoardCharts,
    MediaAsset,
)
from dflux.api.serializers import (
    SaveDashBoardSerializer,
    SaveDashBoardChartsSerializer,
    DashBoardSerializer,
    DashBoardDetailSerializer,
    ShareDashBoardSerializer,
    RequiredFieldsDashBoardSerializer,
)
from dflux.utils.emails import emails
from dflux.api.views.base import BaseAPIView

from decouple import config

from .filters import DashboardFilter
from .permissions import (
    ProjectOwnerOrCollaborator,
    ProjectDashboardAccess,
    ProjectModuleAccess,
)


class LimitedDashBoardView(BaseAPIView):
    """Create New dash board in project and get all the dashboards in projects"""

    permission_classes = [ProjectModuleAccess]

    def get(self, request, pk):
        dashboards = DashboardFilter(
            request.GET,
            queryset=DashBoard.objects.select_related("project").filter(project=pk),
        ).qs
        serializer = RequiredFieldsDashBoardSerializer(dashboards, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)


class DashBoardView(BaseAPIView):
    """
    API endpoint that allows view list of all the dashboards or create new dashboard.

    * Requires JWT authentication.
    * Only collaborator or owner of the project can access.
    * This endpoint will allows only GET, POST methods.
    """

    permission_classes = [IsAuthenticated, ProjectModuleAccess]

    def get(self, request, pk):
        """
        View list of all the dashboards.
        """
        dashboards = DashboardFilter(
            request.GET,
            queryset=DashBoard.objects.select_related("project").filter(project=pk),
        ).qs
        serializer = DashBoardSerializer(dashboards, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def post(self, request, pk):
        """
        Create new dashboard.
        """
        request.data["project"] = pk
        dashboard_serializer = SaveDashBoardSerializer(data=request.data)
        if dashboard_serializer.is_valid(raise_exception=True):
            dashboard = dashboard_serializer.save()
            charts = [
                {"dashboard": dashboard.id, "chart": chart}
                for chart in request.data.get("charts")
            ]
            serializer = SaveDashBoardChartsSerializer(data=charts, many=True)
            if serializer.is_valid(raise_exception=True):
                serializer.save()
            dashboard_details_serializer = DashBoardDetailSerializer(dashboard)
            return Response(dashboard_details_serializer.data)


class DashBoardDetailView(BaseAPIView):
    """
    API endpoint that allows view, update, delete individual dashboard details.

    * Requires JWT authentication.
    * Only collaborator or owner of the project can access.
    * This endpoint will allows only GET, PUT, DELETE methods.
    """

    permission_classes = [IsAuthenticated, ProjectDashboardAccess]

    def get_object(self, pk):
        """
        Get dashboard object using the pk value.
        """
        try:
            return DashBoard.objects.select_related("project").get(id=pk)
        except DashBoard.DoesNotExist:
            raise Http404

    def get(self, request, pk):
        """
        View individual dashboard details.
        """
        dashboard = self.get_object(pk)
        serializer = DashBoardDetailSerializer(dashboard)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def put(self, request, pk):
        """
        Update individual dashboard details.
        """
        dashboard = self.get_object(pk)
        if request.data.get("extra") is not None:
            media = MediaAsset.objects.filter(
                id=request.data.get("extra").get("thumbnail").get("previous_id")
            ).first()
            if media is not None:
                media.delete()
        dashboard_charts = DashBoardCharts.objects.filter(dashboard=dashboard)
        dashboard_charts.delete()

        # updated dashboard
        dashboard_serializer = SaveDashBoardSerializer(
            dashboard, data=request.data, partial=True
        )
        if dashboard_serializer.is_valid(raise_exception=True):
            dashboard_serializer.save()

        # update dashboard charts
        if request.data.get("charts") is not None:
            for chart in request.data.get("charts"):
                data = {"dashboard": dashboard.id, "chart": chart}
                serializer = SaveDashBoardChartsSerializer(data=data)
                if serializer.is_valid(raise_exception=True):
                    serializer.save()
        serializer = DashBoardDetailSerializer(dashboard)
        return Response(serializer.data)

    def delete(self, request, pk):
        """
        Delete individual dashboard details.
        """
        dashboard = self.get_object(pk)
        dashboard.delete()
        return Response({"message": "Dashboard Deleted."}, status=status.HTTP_200_OK)


class UpdateDashboardChartDetails(BaseAPIView):
    """
    API endpoint that allows update individual chart details in the dashboard.

    * Requires JWT authentication.
    * This endpoint will allows only PUT method.
    """

    permission_classes = (IsAuthenticated,)

    def put(self, request, pk):
        dashboard_chart = get_object_or_404(DashBoardCharts, id=pk)
        serializer = SaveDashBoardChartsSerializer(
            dashboard_chart, data=request.data, partial=True
        )
        if serializer.is_valid(raise_exception=True):
            serializer.save()
            return Response(serializer.data)


class ShareDashBoardView(BaseAPIView):
    """
    API endpoint that allows share the dashboards to users.

    * Authentication not required.
    * This endpoint will allows only GET method.
    """

    permission_classes = (IsAuthenticated,)

    def get(self, request, pk):
        import uuid

        token = uuid.uuid1().hex
        dashboard = get_object_or_404(DashBoard, id=pk)
        request.data["token"] = token
        request.data["dashboard"] = dashboard.id
        serializer = ShareDashBoardSerializer(data=request.data)
        if serializer.is_valid(raise_exception=True):
            serializer.save()
            return Response({"shared_token": token})


class LookSharedDashBoard(BaseAPIView):
    """
    API endpoint that allows view the shared dashboards details to users.

    * Authentication not required.
    * This endpoint will allows only GET method.
    """

    def get(self, request):
        dashboard_token = request.query_params.get("token")
        dashboard = get_object_or_404(ShareDashBoard, token=dashboard_token)
        serializer = DashBoardDetailSerializer(dashboard.dashboard)
        return Response(serializer.data, status=status.HTTP_200_OK)


class SendDashboardEmail(BaseAPIView):
    """
    API endpoint that allows send shared dashboard email to users.

    * Requires JWT authentication.
    * This endpoint will allows only POST method.
    """

    permission_classes = (IsAuthenticated,)

    def post(self, request):

        url = request.data.get("url")
        emails.send_dashboard_email(request, url)
        return Response({"message": "email sent"})
