import uuid
from .team import Team
from django.db import models

from dflux.db.models import TenantUser, TenantBaseModel


ACCESS_CHOICES = [
    ("READ", "read"),
    ("WRITE", "write"),
]


class Project(TenantBaseModel):
    """
    This model will allows store all the Project data in Project table.

    * This model contains FK(one to many) relation with User model.
    """

    user = models.ForeignKey(
        TenantUser, related_name="user_projects", on_delete=models.CASCADE
    )
    name = models.CharField(max_length=256)
    token = models.CharField(max_length=256, null=True, blank=True)
    description = models.TextField(null=True, blank=True)
    extra = models.JSONField(null=True, blank=True)

    class Meta:

        verbose_name = "Project"
        verbose_name_plural = "Projects"
        db_table = "projects"
        unique_together = ("created_by", "name")
        ordering = ("name",)

    def __str__(self):
        return self.name

    def save(self, *args, **kwargs):
        self.token = uuid.uuid4().hex + uuid.uuid4().hex
        super(Project, self).save(*args, **kwargs)

    @property
    def queries(self):
        from .query import Query

        queyset = Query.objects.filter(project__id=self.id).order_by("name")[:5]
        return queyset

    @property
    def triggers(self):
        from .trigger import Trigger

        triggers = Trigger.objects.filter(project__id=self.id).order_by("name")[:5]
        return triggers

    @property
    def chart_triggers(self):
        from .trigger import ChartTrigger

        triggers = ChartTrigger.objects.filter(project__id=self.id).order_by("name")[:5]
        return triggers

    @property
    def charts(self):
        from .trigger import Charts

        charts = Charts.objects.filter(project__id=self.id).order_by("name")[:5]
        return charts

    @property
    def models(self):
        from .data_model import DataModel

        models = DataModel.objects.filter(project__id=self.id).order_by("name")[:5]
        return models

    @property
    def dashboards(self):
        from .dashboard import DashBoard

        dashboards = DashBoard.objects.filter(project__id=self.id).order_by("name")[:5]
        return dashboards

    @property
    def users_count(self):
        count = ProjectMembers.objects.filter(project=self.id).count()
        return count

    @property
    def data_sources(self):
        from .connection import Connection
        from .excel_data import ExcelData
        from .json_data import JsonData
        from dflux.api.serializers import (
            ConnectionSerializer,
            ExcelDataSerializer,
            JsonDataSerializer,
        )

        response_data = []
        queryset = (
            Connection.objects.filter(project__id=self.id)
            .exclude(connection_type="INTERNAL")
            .order_by("-updated_at")[:5]
        )
        excel_queryset = ExcelData.objects.filter(project__id=self.id).order_by(
            "-updated_at"
        )[:5]
        json_queryset = JsonData.objects.filter(project__id=self.id).order_by(
            "-updated_at"
        )[:5]
        connectios_serializer = ConnectionSerializer(queryset, many=True)
        excel_serializer = ExcelDataSerializer(excel_queryset, many=True)
        json_serializer = JsonDataSerializer(json_queryset, many=True)
        response_data.extend(connectios_serializer.data)
        response_data.extend(excel_serializer.data)
        response_data.extend(json_serializer.data)
        return response_data


class ProjectTeam(TenantBaseModel):
    """
    This model will allows store all the ProjectTeam data in ProjectTeam table.

    * This model contains FK(one to many) relation with Project, Team models.
    """

    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    team = models.ForeignKey(Team, on_delete=models.CASCADE)
    extra = models.JSONField(null=True, blank=True)

    class Meta:
        verbose_name = "Project Team"
        verbose_name_plural = "Project Teams"
        db_table = "project_team"

    def __str__(self):
        return f"{self.project.name} -- {self.team.name}"


class ProjectInvitation(TenantBaseModel):
    """
    This model will allows store all the ProjectInvitation data in ProjectInvitation table.

    * This model contains FK(one to many) relation with Project model.
    """

    invitee = models.CharField(max_length=256)
    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    token = models.CharField(max_length=1000)
    status = models.BooleanField(default=True)
    access = models.CharField(max_length=256, default="READ", choices=ACCESS_CHOICES)
    module_access = models.JSONField(default=list)
    extra = models.JSONField(null=True, blank=True)

    class Meta:
        verbose_name = "Project Invitation"
        verbose_name_plural = "Project Invitations"
        db_table = "project_invitations"

    def __str__(self):
        return self.invitee


class ProjectMembers(TenantBaseModel):
    """
    This model will allows store all the ProjectMembers data in ProjectMembers table.

    * This model contains FK(one to many) relation with User, Project models.
    """

    user = models.ForeignKey(
        TenantUser, related_name="project_member", on_delete=models.CASCADE
    )
    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    extra = models.JSONField(null=True, blank=True)

    class Meta:
        verbose_name = "Project Members"
        verbose_name_plural = "Project Members"
        db_table = "project_members"
        unique_together = ("user", "project")

    def __str__(self):
        return f"{self.user.username}"

    @property
    def role(self):
        if self.project.user == self.user:
            return "Owner"
        else:
            a = ProjectInvitation.objects.filter(
                project__id=self.project.id, invitee=self.user.username
            )
            for members in a:
                if members.access == "WRITE":
                    return "Collaborator"
            return "Viewer"

    @property
    def access(self):
        if self.project.user == self.user:
            return "WRITE"

    @property
    def module_access(self):
        a = ProjectInvitation.objects.filter(
            project__id=self.project.id, invitee=self.user.email
        )
        for members in a:
            return members.module_access
