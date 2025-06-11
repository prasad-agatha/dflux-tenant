from django.db import models
from dflux.db.models import Project, Connection, TenantUser, TenantBaseModel


class ExcelData(TenantBaseModel):
    """
    This model will allows store all the ExcelData data in ExcelData table.

    * This model contains FK(one to many) relation with User, Project, Connection models.
    """

    tablename = models.CharField(max_length=256)
    project = models.ForeignKey(
        Project, on_delete=models.CASCADE, null=True, blank=True
    )
    file_type = models.CharField(max_length=256, null=True, blank=True)
    connection = models.ForeignKey(
        Connection, on_delete=models.CASCADE, null=True, blank=True
    )

    class Meta:
        verbose_name = "Excel Data"
        verbose_name_plural = "Excel Data"
        db_table = "excel_data"
        unique_together = ("created_by", "tablename", "file_type")

    def __str__(self):
        return self.tablename
