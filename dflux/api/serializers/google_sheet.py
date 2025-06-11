from dflux.db.models import GoogleSheet

from .base import BaseModelSerializer


class GoogleSheetSerializer(BaseModelSerializer):
    """
    This serializer will allows serialize the GoogleSheet model data.
    """

    class Meta:
        model = GoogleSheet
        fields = "__all__"
