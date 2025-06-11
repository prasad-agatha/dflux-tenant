from django.db import models
from dflux.db.models import Query, Project, TenantUser, TenantBaseModel, DataModel


CHART_TYPES = [("query", "query"), ("data_model", "data_model")]


class Charts(TenantBaseModel):
    """
    This model will allows store all the charts data in charts table.

    * This model contains  FK(one to many) relation with User, Query, DataModel models.
    """

    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    user = models.ForeignKey(
        TenantUser, related_name="chart_created_by", on_delete=models.CASCADE
    )
    query = models.ForeignKey(Query, on_delete=models.CASCADE, null=True, blank=True)
    data_model = models.ForeignKey(
        DataModel, on_delete=models.CASCADE, null=True, blank=True
    )
    name = models.CharField(max_length=256)
    chart_type = models.CharField(max_length=256)
    save_from = models.CharField(
        max_length=256, choices=CHART_TYPES, null=True, blank=True
    )
    data = models.JSONField()
    extra = models.JSONField(null=True, blank=True)

    class Meta:
        verbose_name = "Charts"
        verbose_name_plural = "Charts"
        db_table = "charts"
        unique_together = ("project", "name")

    def __str__(self):
        return self.name


class ShareCharts(TenantBaseModel):
    """
    This model will allows store all the shared charts data in shared charts table.
    """

    charts = models.ForeignKey(Charts, on_delete=models.CASCADE)
    token = models.CharField(max_length=256)
    extra = models.JSONField(null=True, blank=True)

    class Meta:
        verbose_name = "Share Chart"
        verbose_name_plural = "Share Charts"
        db_table = "share_charts"

    def __str__(self):
        return self.charts.name
