from django.urls import path

from dflux.api.views import (
    ChartsView,
    ChartsDetailView,
    LookSharedChart,
    ShareChartView,
    SendChartEmail,
    ChartsLimitedView,
)


urlpatterns = [
    # get list of saved charts
    path("projects/<int:pk>/charts/", ChartsView.as_view(), name="charts"),
    # get  particular saved chart details
    path("charts/<int:pk>/", ChartsDetailView.as_view(), name="chart-details"),
    path(
        "shared-charts/",
        LookSharedChart.as_view(),
        name="view-shared-dashboard",
    ),
    path(
        "charts/<int:pk>/share/",
        ShareChartView.as_view(),
        name="share-chart",
    ),
    path(
        "send-chart-emails/",
        SendChartEmail.as_view(),
        name="send-chart-emails",
    ),
    path(
        "projects/<int:pk>/limitedcharts/",
        ChartsLimitedView.as_view(),
        name="limited-charts",
    ),
]
