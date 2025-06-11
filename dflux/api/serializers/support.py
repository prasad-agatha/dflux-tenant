from dflux.db.models import Support

from .base import BaseModelSerializer


class SupportSerializer(BaseModelSerializer):
    """
    This serializer will allows serialize the Support model data.
    """

    class Meta:
        model = Support
        exclude = ("tenant",)
