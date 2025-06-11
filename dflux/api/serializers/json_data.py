from rest_framework import serializers

from dflux.db.models import JsonData

from .base import BaseModelSerializer


class JsonDataSerializer(BaseModelSerializer):
    """
    This serializer will allows serialize the JsonData model data.
    """

    engine = serializers.ReadOnlyField()

    class Meta:
        model = JsonData
        fields = "__all__"
