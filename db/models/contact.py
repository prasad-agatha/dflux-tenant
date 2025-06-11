from django.db import models
from dflux.db.models import TenantUser, TenantBaseModel


class ContactSale(TenantBaseModel):
    """
    This model will allows store all the ContactSale data in ContactSale table.

    * This model contains FK(one to many) relation with User model.
    """

    subject = models.CharField(max_length=256)
    message = models.TextField()

    class Meta:
        verbose_name = "Contact Sale"
        verbose_name_plural = "Contact Sales"
        db_table = "contact_sales"

    def __str__(self):
        return self.user.username
