from django.urls import path, include

urlpatterns = [
    path("api/", include("dflux.api.views.urls.asset")),
    path("api/", include("dflux.api.views.urls.auth")),
    path("api/", include("dflux.api.views.urls.project")),
    path("api/", include("dflux.api.views.urls.chart")),
    path("api/", include("dflux.api.views.urls.connection")),
    path("api/", include("dflux.api.views.urls.contact")),
    path("api/", include("dflux.api.views.urls.dashboard")),
    path("api/", include("dflux.api.views.urls.integration")),
    path("api/", include("dflux.api.views.urls.ml")),
    path("api/", include("dflux.api.views.urls.python")),
    path("api/", include("dflux.api.views.urls.query")),
    path("api/", include("dflux.api.views.urls.team")),
    path("api/", include("dflux.api.views.urls.trigger")),
]
