from django.db import models
from dflux.db.models import Project, Connection, TenantUser, TenantBaseModel


class GoogleSheet(TenantBaseModel):
    """
    This model will allows store all the GoogleSheet data in GoogleSheet table.

    * This model contains FK(one to many) relation with User, Project, Connection models.
    """

    tablename = models.CharField(max_length=256)
    project = models.ForeignKey(
        Project, on_delete=models.CASCADE, null=True, blank=True
    )
    connection = models.ForeignKey(
        Connection, on_delete=models.CASCADE, null=True, blank=True
    )

    class Meta:
        verbose_name = "Google Sheet"
        verbose_name_plural = "Google Sheets"
        db_table = "google_sheets"
        unique_together = ("created_by", "tablename")

    def __str__(self):
        return self.tablename
