from django.urls import path

from dflux.api.views import (
    ConnectionView,
    ConnectionDetailView,
    TestConnectionView,
    TestSnowflakeConnectionView,
    TestBigQueryConnectionView,
)


urlpatterns = [
    # get list of connections
    path("projects/<int:pk>/connections/", ConnectionView.as_view(), name="connection"),
    # get particular connection details
    path(
        "connections/<int:pk>/",
        ConnectionDetailView.as_view(),
        name="connection_detail",
    ),
    # check given db details are correct or not
    path("test/connection/", TestConnectionView.as_view(), name="test-connection"),
    # test snowflake connection
    path(
        "test/snowflake/connection/",
        TestSnowflakeConnectionView.as_view(),
        name="test-snowflake-connection",
    ),
    # big query endpoints
    path(
        "test/bigquery/connection/",
        TestBigQueryConnectionView.as_view(),
        name="test-bigquery-connection",
    ),
]
