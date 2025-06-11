from rest_framework import serializers

from dflux.db.models import Team, TeamMembers, TeamInvitation

from .base import BaseModelSerializer


class TeamSerializer(BaseModelSerializer):
    """
    This serializer will allows serialize the Team model data.
    """

    class Meta:
        model = Team
        fields = "__all__"


class TeamMembersSerializer(BaseModelSerializer):
    """
    This serializer will allows serialize the TeamMembers model data.
    """

    user = serializers.StringRelatedField()
    team = serializers.StringRelatedField()

    class Meta:
        model = TeamMembers
        fields = ["id", "user", "team", "extra"]


class AddTeamMembersSerializer(BaseModelSerializer):
    """
    This serializer will allows serialize the TeamMembers model data.
    """

    class Meta:
        model = TeamMembers
        fields = "__all__"


class TeamInvitationSerializer(BaseModelSerializer):
    """
    This serializer will allows serialize the TeamInvitation model data.
    """

    class Meta:
        model = TeamInvitation
        fields = ["user", "team", "token", "status", "extra"]
