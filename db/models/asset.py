from django.db import models
from dflux.db.mixins import TimeAuditModel


class MediaAsset(TimeAuditModel):
    """
    This model will allows upload files into to default file storage(like s3 bucket etc).
    """

    name = models.CharField(max_length=255)
    asset = models.FileField(upload_to="assets")

    def __str__(self):
        return self.name
