from django.urls import path

from dflux.api.views import (
    ExecuteQuery,
    SQLQuery,
    QueryDetail,
    Schematable,
)

urlpatterns = [
    # execute given sql raw query
    path("queries/", ExecuteQuery.as_view(), name="query"),
    # save project related query
    path("project/<int:pk>/queries/", SQLQuery.as_view(), name="save-sql-query"),
    # get saved query datails
    path("queries/<int:pk>/", QueryDetail.as_view(), name="query-details"),
    # get particular database tables
    path("connections/<int:pk>/schema/", Schematable.as_view(), name="table-query"),
]
