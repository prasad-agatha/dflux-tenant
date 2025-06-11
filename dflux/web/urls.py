from django.urls import path
from django.views.generic import TemplateView
from dflux.web.views import TenantView, TenantDetailsView, TenantUpdateView

urlpatterns = [
    path("about/", TemplateView.as_view(template_name="about.html")),
    path("create/tenant/", TenantView.as_view()),
    # path("tenants/", TenantDetailsView.as_view()),
    # path("tenants/<int:pk>/edit/", TenantUpdateView.as_view()),
]
