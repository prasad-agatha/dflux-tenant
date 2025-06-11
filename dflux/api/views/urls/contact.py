from django.urls import path

from dflux.api.views.contact import ContactSaleView, ContactSaleDetailView
from dflux.api.views.support import SupportView


urlpatterns = [
    # contact sale endpoint
    path(
        "contact/sales/",
        ContactSaleView.as_view(),
        name="contact-sale",
    ),
    path(
        "contact/sales/<int:pk>/",
        ContactSaleDetailView.as_view(),
        name="contact-sales-detail",
    ),
    # support endpoints
    path(
        "support/",
        SupportView.as_view(),
        name="support-endpoint",
    ),
]
