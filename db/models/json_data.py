from django.db import models
from dflux.db.models import Project, Connection, TenantUser, TenantBaseModel


class JsonData(TenantBaseModel):
    """
    This model will allows store all the JsonData data in JsonData table.

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
        verbose_name = "Json Data"
        verbose_name_plural = "Json Data"
        db_table = "json_data"
        # unique_together = ("created_by", "tablename")

    def __str__(self):
        return self.tablename
