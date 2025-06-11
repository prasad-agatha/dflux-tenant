from django.db import models
from dflux.db.models import Connection, TenantBaseModel


class GoogleBigQueryCredential(TenantBaseModel):
    """
    This model will allows store all the GoogleBigQueryCredential data in GoogleBigQueryCredential table.

    * This model contains FK(one to many) relation with Connection model.
    """

    connection = models.ForeignKey(
        Connection, on_delete=models.CASCADE, null=True, blank=True
    )
    credential_path = models.URLField()

    class Meta:
        verbose_name = "Google Bigquery Credential"
        verbose_name_plural = "Google Bigquery Credentials"
        db_table = "google_bigquery_credentials"

    def __str__(self):
        return self.connection.name
