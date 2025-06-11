from django.db import models
from dflux.db.models import Project, TenantBaseModel

CONNECTION_CHOICES = [
    ("INTERNAL", "internal"),
    ("EXTERNAL", "external"),
    ("SNOWFLAKE", "snowflake"),
]


class Connection(TenantBaseModel):
    """
    This model will allows store all the connections data in connections table.

    * This model contains FK(one to many) relation with Project model.
    """

    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    name = models.CharField(max_length=256)
    engine = models.CharField(max_length=256, null=True, blank=True)
    dbname = models.CharField(max_length=256, null=True, blank=True)
    username = models.CharField(max_length=256, null=True, blank=True)
    password = models.CharField(max_length=256, null=True, blank=True)
    port = models.CharField(max_length=256, null=True, blank=True)
    host = models.CharField(max_length=256, null=True, blank=True)
    extra = models.JSONField(null=True, blank=True)
    connection_type = models.CharField(
        max_length=256, choices=CONNECTION_CHOICES, null=True, blank=True
    )
    # snowflake related fields
    account = models.CharField(max_length=256, null=True, blank=True)
    warehouse = models.CharField(max_length=256, null=True, blank=True)
    schema = models.CharField(max_length=256, null=True, blank=True)

    class Meta:
        verbose_name = "Connection"
        verbose_name_plural = "Connections"
        db_table = "connections"
        # unique_together = ("project", "name")
        ordering = ("name",)

    def __str__(self):
        return self.name
