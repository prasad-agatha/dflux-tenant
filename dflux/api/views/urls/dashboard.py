from django.urls import path

from dflux.api.views.dashboard import (
    DashBoardView,
    DashBoardDetailView,
    ShareDashBoardView,
    LookSharedDashBoard,
    UpdateDashboardChartDetails,
    SendDashboardEmail,
    LimitedDashBoardView,
)


urlpatterns = [
    path("projects/<int:pk>/dashboards/", DashBoardView.as_view(), name="dashboard"),
    path(
        "dashboards/<int:pk>/", DashBoardDetailView.as_view(), name="dashboard-details"
    ),
    # share DashBoard
    path(
        "dashboards/<int:pk>/share/",
        ShareDashBoardView.as_view(),
        name="share-dashboard",
    ),
    path(
        "shared-dashboards/",
        LookSharedDashBoard.as_view(),
        name="view-shared-dashboard",
    ),
    path(
        "dashboards/charts/<int:pk>/",
        UpdateDashboardChartDetails.as_view(),
        name="updata-dashboards-charts-details",
    ),
    path(
        "send-dashboard-emails/",
        SendDashboardEmail.as_view(),
        name="send-dashboard-emails",
    ),
    path(
        "projects/<int:pk>/limiteddashboards/",
        LimitedDashBoardView.as_view(),
        name="limited-dashboard",
    ),
]
