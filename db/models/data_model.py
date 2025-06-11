from django.db import models

from dflux.db.models import Project, Query, TenantUser, TenantBaseModel


class DataModelMetaData(TenantBaseModel):
    """
    This model will allows store all the DataModelMetaData data in datamodelmetadata table.

    * This model contains FK(one to many) relation with Query model.
    """

    MODELING_TYPES = (
        ("classification", "classification"),
        ("regression", "regression"),
        ("timeseries", "timeseries"),
    )
    name = models.CharField(max_length=256)
    query = models.ForeignKey(Query, on_delete=models.CASCADE)
    skipped = models.BooleanField(default=False)
    pre_processing = models.JSONField(default=dict)
    modeling_type = models.CharField(max_length=256, choices=MODELING_TYPES)
    algorithms = models.JSONField(default=list)
    target_variable = models.CharField(max_length=256)

    class Meta:
        verbose_name = "Datamodel Metadata"
        verbose_name_plural = "Datamodel Metadata"
        db_table = "datamodel_metadata"

    def __str__(self):
        return self.name


class DataModel(TenantBaseModel):
    """
    This model will allows store all the DataModel data in datamodel table.

    * This model contains FK(one to many) relation with Query model.
    """

    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    name = models.CharField(max_length=256)
    model_type = models.CharField(max_length=256)
    data = models.JSONField()
    other_params = models.JSONField()
    extra = models.JSONField(null=True, blank=True)
    meta_data = models.ForeignKey(DataModelMetaData, on_delete=models.CASCADE)
    pickle_url = models.URLField(null=True, blank=True)
    scaler_url = models.URLField(null=True, blank=True)

    class Meta:
        verbose_name = "Data Model"
        verbose_name_plural = "Data Models"
        db_table = "data_models"

    def __str__(self):
        return self.name
