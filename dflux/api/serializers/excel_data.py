from rest_framework import serializers

from dflux.db.models import ExcelData

from .base import BaseModelSerializer


class ExcelDataSerializer(BaseModelSerializer):
    """
    This serializer will allows serialize the ExcelData model data.
    """

    engine = serializers.ReadOnlyField()

    class Meta:
        model = ExcelData
        fields = "__all__"
