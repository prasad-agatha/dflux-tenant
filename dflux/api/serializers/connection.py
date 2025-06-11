from rest_framework import serializers

from dflux.db.models import Connection

from .base import BaseModelSerializer


class ConnectionSerializer(BaseModelSerializer):
    """
    This serializer will allows serialize the Connection model data.
    """

    created_by = serializers.ReadOnlyField(source="project.user.email")

    class Meta:
        model = Connection
        fields = [
            "id",
            "project",
            "name",
            "engine",
            "dbname",
            "username",
            "password",
            "port",
            "host",
            "extra",
            "connection_type",
            "account",
            "warehouse",
            "schema",
            "created_by",
            "created_at",
            "updated_at",
        ]
