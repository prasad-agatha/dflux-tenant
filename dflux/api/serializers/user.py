from rest_framework import serializers

from dflux.db.models import TenantUser
from .base import BaseModelSerializer
from dflux.api.serializers import TenantSerializer

# class PasswordResetTokenSerializer(BaseModelSerializer):
#     """
#     This serializer will allows serialize the UserPasswordResetTokens model data.
#     """

#     class Meta:
#         model = UserPasswordResetTokens
#         fields = "__all__"


class UserSerializer(BaseModelSerializer):
    """
    This serializer will allows serialize the User model data.
    """

    class Meta:
        model = TenantUser
        exclude = ("password",)


class UserDetailsSerializer(BaseModelSerializer):
    """
    This serializer will allows serialize the User model data.
    """

    tenant = TenantSerializer()

    class Meta:
        model = TenantUser
        fields = "__all__"
