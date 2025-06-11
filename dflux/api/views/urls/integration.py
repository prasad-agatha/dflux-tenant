from django.urls import path


from dflux.api.views.dump_excel import (
    DumpExcelData,
    DumpExcelDataDetail,
    GoogleSheetParserEndpoint,
)
from dflux.api.views.dump_json import DumpJsonDataEndpoint, DumpJsonDataDetail
from dflux.api.views.google_sheet import DumpGoogleSheetEndpoint, DumpGoogleSheetDetail
from dflux.api.views.data_model import (
    DataModelView,
    DataModeDetailView,
    DataModelLimitedView,
)


urlpatterns = [
    path("projects/<int:pk>/excel/", DumpExcelData.as_view(), name="excel"),
    path("excel/<int:pk>/", DumpExcelDataDetail.as_view(), name="excel-detail"),
    path(
        "google-sheet/parser/",
        GoogleSheetParserEndpoint.as_view(),
        name="google-sheet-parser",
    ),
    path("projects/<int:pk>/json/", DumpJsonDataEndpoint.as_view(), name="json-dump"),
    path("json/<int:pk>/", DumpJsonDataDetail.as_view(), name="json-detail"),
    path(
        "project/<int:pk>/googlesheet",
        DumpGoogleSheetEndpoint.as_view(),
        name="googlesheet",
    ),
    path(
        "googlesheet/<int:pk>/",
        DumpGoogleSheetDetail.as_view(),
        name="googlesheet-detail",
    ),
    # data models
    path("projects/<int:pk>/models/", DataModelView.as_view(), name="data-models"),
    path(
        "projects/<int:pk>/limitedmodels/",
        DataModelLimitedView.as_view(),
        name="data-models-limited",
    ),
    path(
        "projects/<int:project_id>/models/<int:pk>/",
        DataModeDetailView.as_view(),
        name="data-model-detail",
    ),
]
