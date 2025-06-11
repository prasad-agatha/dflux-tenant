from django.urls import path

from dflux.api.views.asset import MediaAssetView, MediaAssetDetailView


urlpatterns = [
    # media asset endpoints
    path("assets/", MediaAssetView.as_view(), name="assets"),
    path("assets/<int:pk>/", MediaAssetDetailView.as_view(), name="asset-detail"),
]
