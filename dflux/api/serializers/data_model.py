from rest_framework import serializers

from dflux.db.models import DataModel, DataModelMetaData

from .base import BaseModelSerializer


class DataModelSerializer(BaseModelSerializer):
    """
    This serializer will allows serialize the DataModel model data.
    """

    created_by = serializers.ReadOnlyField(source="created_by.email")

    class Meta:
        model = DataModel
        fields = [
            "id",
            "project",
            "name",
            "model_type",
            "data",
            "other_params",
            "extra",
            "created_at",
            "created_by",
            "updated_at",
            "meta_data",
            "pickle_url",
            "scaler_url",
        ]


class RequiredFieldsDataModelSerializer(BaseModelSerializer):
    """
    This serializer will allows serialize the DataModel model data.
    """

    created_by = serializers.ReadOnlyField(source="created_by.email")

    class Meta:
        model = DataModel
        fields = [
            "id",
            "project",
            "name",
            "model_type",
            "data",
            "other_params",
            "extra",
            "created_by",
            "created_at",
            "updated_at",
            "meta_data",
            "pickle_url",
            "scaler_url",
        ]


class DataModelLimitedFieldsSerializer(BaseModelSerializer):
    """
    This serializer will allows serialize the DataModel model data.
    """

    class Meta:
        model = DataModel
        fields = ["id", "project", "name", "model_type", "created_by"]


class DataModelMetaDataSerializer(BaseModelSerializer):
    """
    This serializer will allows serialize the DataModelMetaData model data.
    """

    class Meta:
        model = DataModelMetaData
        exclude = ("tenant",)


class NewDataModelSerializer(BaseModelSerializer):
    """
    This serializer will allows serialize the DataModel model data.
    """

    class Meta:
        model = DataModel
        fields = [
            "name",
            "created_at",
        ]
