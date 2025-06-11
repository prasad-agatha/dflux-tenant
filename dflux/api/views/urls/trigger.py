from django.urls import path

from dflux.api.views.trigger import (
    # TriggerView,
    # TriggerDetailView,
    TriggerOutputViews,
    ChartTriggerView,
    ChartTriggerDetailView,
)


urlpatterns = [
    # path("projects/<int:pk>/triggers/", TriggerView.as_view(), name="triggers"),
    # path(
    #     "projects/triggers/<int:pk>/",
    #     TriggerDetailView.as_view(),
    #     name="trigger-details",
    # ),
    path(
        "queries/<int:pk>/triggers/output/",
        TriggerOutputViews.as_view(),
        name="trigger-output",
    ),
    path(
        "projects/<int:pk>/charts/triggers/",
        ChartTriggerView.as_view(),
        name="chart-trigger",
    ),
    path(
        "projects/charts/triggers/<int:pk>/",
        ChartTriggerDetailView.as_view(),
        name="trigger-details",
    ),
]
