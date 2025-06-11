from dflux.db.models import ContactSale

from .base import BaseModelSerializer


class ContactSaleSerializer(BaseModelSerializer):
    """
    This serializer will allows serialize the ContactSale model data.
    """

    class Meta:
        model = ContactSale
        exclude = ("tenant",)
