from rest_framework import serializers

from .user import UserSerializer

from .query import NewQuerySerializer
from .base import BaseModelSerializer

from .dashboard import NewDashBoardSerializer
from .data_model import NewDataModelSerializer
from .charts import ChartLimitedFieldsSerializer
from .trigger import TriggrSerializer, NewChartTriggrSerializer

from dflux.db.models import Project, ProjectTeam, ProjectMembers, ProjectInvitation


class ProjectSerializer(BaseModelSerializer):
    """
    This serializer will allows serialize the Project model data.
    """

    class Meta:
        model = Project
        exclude = ("tenant", "created_by")


class ProjectDetailSerializer(BaseModelSerializer):
    """
    This serializer will allows serialize the Project model data.
    """

    data_sources = serializers.ReadOnlyField()
    queries = NewQuerySerializer(many=True)
    triggers = TriggrSerializer(many=True)
    chart_triggers = NewChartTriggrSerializer(many=True)
    charts = ChartLimitedFieldsSerializer(many=True)
    models = NewDataModelSerializer(many=True)
    dashboards = NewDashBoardSerializer(many=True)
    users_count = serializers.ReadOnlyField()

    class Meta:
        model = Project
        fields = [
            "id",
            "user",
            "users_count",
            "name",
            "token",
            "created_at",
            "description",
            "data_sources",
            "queries",
            "triggers",
            "chart_triggers",
            "charts",
            "models",
            "dashboards",
            "extra",
        ]


class ProjectTeamSerializer(BaseModelSerializer):
    """
    This serializer will allows serialize the ProjectTeam model data.
    """

    project = serializers.StringRelatedField()
    team = serializers.StringRelatedField()

    class Meta:
        model = ProjectTeam
        fields = ["id", "project", "team", "extra"]


class AddProjectTeamSerializer(BaseModelSerializer):
    """
    This serializer will allows serialize the ProjectTeam model data.
    """

    class Meta:
        model = ProjectTeam
        fields = "__all__"


class ProjectMemberSerializer(BaseModelSerializer):
    """
    This serializer will allows serialize the ProjectMembers model data.
    """

    project = serializers.StringRelatedField()
    user = UserSerializer()
    role = serializers.ReadOnlyField()
    access = serializers.ReadOnlyField()
    module_access = serializers.ReadOnlyField()

    class Meta:
        model = ProjectMembers
        fields = ["id", "project", "user", "role", "access", "module_access", "extra"]


class AddProjectMembersSerializer(BaseModelSerializer):
    """
    This serializer will allows serialize the ProjectMembers model data.
    """

    class Meta:
        model = ProjectMembers
        exclude = ("tenant",)


class ProjectInvitationSerializer(BaseModelSerializer):
    """
    This serializer will allows serialize the ProjectInvitation model data.
    """

    class Meta:
        model = ProjectInvitation
        fields = [
            "invitee",
            "project",
            "access",
            "module_access",
            "token",
            "status",
            "extra",
        ]
