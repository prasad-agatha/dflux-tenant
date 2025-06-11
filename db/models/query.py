from django.db import models
from django.conf import settings

from dflux.db.models.json_data import JsonData
from dflux.db.models.excel_data import ExcelData
from dflux.db.models import Connection, Project, TenantBaseModel, TenantUser


class Query(TenantBaseModel):
    """
    This model will allows store all the Query data in Query table.

    * This model contains FK(one to many) relation with User, Project, Connection, ExcelData, JsonData models.
    """

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, related_name="user_queries", on_delete=models.CASCADE
    )

    project = models.ForeignKey(
        Project, related_name="project_queries", on_delete=models.CASCADE
    )

    connection = models.ForeignKey(
        Connection, related_name="connection_queries", on_delete=models.CASCADE
    )

    engine_type = models.CharField(max_length=256, null=True, blank=True)

    excel = models.ForeignKey(
        ExcelData,
        related_name="excel_queries",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
    )

    json = models.ForeignKey(
        JsonData,
        related_name="json_queries",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
    )

    name = models.CharField(max_length=2000)

    description = models.TextField(null=True, blank=True)

    vars = models.JSONField(null=True, blank=True)

    raw_sql = models.TextField()

    verify = models.BooleanField(default=False)

    accepted = models.BooleanField(default=False)

    extra = models.JSONField(null=True, blank=True)

    # created_by = models.CharField(max_length=256, null=True, blank=True)

    class Meta:
        db_table = "query"

    def __str__(self):
        return self.raw_sql
