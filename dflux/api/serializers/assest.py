from dflux.db.models import MediaAsset

from .base import BaseModelSerializer


class MediaAssetSerializer(BaseModelSerializer):
    """
    This serializer will allows serialize the MediaAsset model data.
    """

    class Meta:
        model = MediaAsset
        fields = ("id", "name", "asset")
