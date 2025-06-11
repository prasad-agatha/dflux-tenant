from statistics import mode
from django.db import models
from dflux.db.models import TenantUser, TenantBaseModel


class Support(TenantBaseModel):
    """
    This model will allows store all the Support data in Support table.

    * This model contains FK(one to many) relation with User model.
    """

    subject = models.CharField(max_length=256)
    description = models.TextField()
    attachment = models.URLField(null=True, blank=True)

    class Meta:
        verbose_name = "Support"
        verbose_name_plural = "Supports"
        db_table = "supports"

    def __str__(self):
        return self.subject
