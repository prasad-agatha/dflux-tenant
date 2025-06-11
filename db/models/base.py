import uuid
from django import conf
from django.db import models
from django.conf import settings
from django.utils import timezone
from crum import get_current_user
from django.contrib.auth.models import AbstractBaseUser
from django.core.management.base import CommandError


class Tenant(models.Model):
    name = models.CharField(max_length=255)
    subdomain_prefix = models.CharField(max_length=255, unique=True)
    created_at = models.DateTimeField(
        auto_now_add=True,
        verbose_name="Created At",
    )
    updated_at = models.DateTimeField(auto_now=True, verbose_name="Last Modified At")

    def __str__(self):
        return self.name


class TenantUserManager(models.Manager):
    # def get_queryset(self):
    #     user = get_current_user()
    #     if user is None or user.is_anonymous:
    #         return super(TenantUserManager, self).get_queryset()
    #     return (
    #         super(TenantUserManager, self).get_queryset().filter(tenant=user.tenant_id)
    #     )

    def create_user(self, email, password, tenant):
        user = self.model(email=email, tenant=tenant)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(
        self, username, email, password, tenant, first_name, last_name
    ):
        try:
            tenant = Tenant.objects.get(pk=int(tenant))
        except Tenant.DoesNotExist:
            raise CommandError("Tenant doesn't exist!")
        user = self.create_user(email, password, tenant)
        user.is_superuser = True
        user.tenant_superuser = True
        user.first_name = first_name
        user.last_name = last_name
        user.save(using=self._db)
        return user

    def get_by_natural_key(self, email_):
        return self.get(email=email_)


class TenantUser(AbstractBaseUser):
    id = models.BigAutoField(unique=True, primary_key=True)

    # on platform stuff

    # mark it unique?
    username = models.CharField(max_length=128, unique=True)
    tenant = models.ForeignKey(Tenant, on_delete=models.CASCADE)

    # user fields
    mobile_number = models.CharField(max_length=255, blank=True, null=True)
    email = models.CharField(max_length=255, unique=True)
    first_name = models.CharField(max_length=255, blank=True)
    last_name = models.CharField(max_length=255, blank=True)

    # tracking metrics
    date_joined = models.DateTimeField(auto_now_add=True, verbose_name="Created At")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="Created At")
    updated_at = models.DateTimeField(auto_now=True, verbose_name="Last Modified At")
    last_location = models.CharField(max_length=255, blank=True)
    created_location = models.CharField(max_length=255, blank=True)

    # the is' es
    is_superuser = models.BooleanField(default=False)
    tenant_superuser = models.BooleanField(default=False)
    is_managed = models.BooleanField(default=False)
    is_bot_app = models.BooleanField(default=False)
    is_password_expired = models.BooleanField(default=False)
    is_active = models.BooleanField(default=True)
    is_staff = models.BooleanField(default=False)
    is_email_verified = models.BooleanField(default=False)
    is_password_autoset = models.BooleanField(default=False)
    objects = TenantUserManager()

    token = models.CharField(max_length=64, blank=True)
    token_status = models.BooleanField(default=False)

    role = models.CharField(max_length=256, null=True, blank=True)
    industry = models.CharField(max_length=256, null=True, blank=True)
    profile_pic = models.URLField(null=True, blank=True)
    extended_date = models.DateTimeField(null=True, blank=True)

    user_role = models.BigIntegerField(null=True)
    billing_address_country = models.CharField(max_length=255, default="INDIA")
    billing_address = models.JSONField(null=True)
    has_billing_address = models.BooleanField(default=False)
    has_onboard = models.BooleanField(default=False)

    # # onboarding fields
    company = models.CharField(max_length=256, null=True, blank=True)
    team_name = models.CharField(max_length=256, null=True, blank=True)
    job_title = models.CharField(max_length=256, null=True, blank=True)
    dflux_using = models.JSONField(null=True, blank=True)

    user_timezone = models.CharField(max_length=255, default="Asia/Kolkata")

    last_active = models.DateTimeField(default=timezone.now, null=True)
    last_login_time = models.DateTimeField(null=True)
    last_logout_time = models.DateTimeField(null=True)
    last_login_ip = models.CharField(max_length=255, blank=True)
    last_logout_ip = models.CharField(max_length=255, blank=True)
    last_login_medium = models.CharField(
        max_length=7,
        default="email",
    )
    last_login_uagent = models.TextField(blank=True)
    token_updated_at = models.DateTimeField(null=True)
    token_expired_at = models.DateTimeField(default=timezone.now)

    USERNAME_FIELD = "username"
    REQUIRED_FIELDS = ["email", "tenant", "first_name", "last_name"]

    def __str__(self):
        return self.username

    # class Meta:
    #     unique_together = [["mobile_number", "tenant"], ["email", "tenant"]]

    def save(self, *args, **kwargs):
        self.email = self.email.lower().strip()
        self.username = uuid.uuid4().hex
        self.token = uuid.uuid4().hex + uuid.uuid4().hex
        if self.mobile_number is not None:
            self.mobile_number = self.mobile_number.lower().strip()

        if self.user_role is not None:
            self.is_staff = True
        else:
            self.is_staff = False

        if self.is_superuser:
            self.is_staff = True

        super(TenantUser, self).save(*args, **kwargs)

    def set_token(self):
        # print("setting token")
        if self.token_expired_at < timezone.now():
            self.token = uuid.uuid4().hex + uuid.uuid4().hex
            self.token_updated_at = timezone.now()
            self.token_expired_at = timezone.now() + timezone.timedelta(days=1)

    def reset_token(self):
        self.token = uuid.uuid4().hex + uuid.uuid4().hex
        self.token_updated_at = timezone.now()
        self.token_expired_at = timezone.now()


class TenantBaseManager(models.Manager):
    # def all(self):
    #     print(..., "v")
    #     user = get_current_user()
    #     if user is None:
    #         return []
    #     print(user)
    #     return self.get_queryset().filter(tenant=user.tenant_id)

    def get_queryset(self):
        user = get_current_user()
        if user is None or user.is_anonymous:
            return super(TenantBaseManager, self).get_queryset()
        return (
            super(TenantBaseManager, self).get_queryset().filter(tenant=user.tenant_id)
        )


class TenantBaseModel(models.Model):
    id = models.BigAutoField(unique=True, primary_key=True)

    tenant = models.ForeignKey(Tenant, on_delete=models.CASCADE)

    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
    )

    # updated_by = models.ForeignKey(
    #     settings.AUTH_USER_MODEL,
    #     on_delete=models.SET_NULL,
    #     null=True,
    # )

    created_at = models.DateTimeField(
        auto_now_add=True,
        verbose_name="Created At",
    )
    updated_at = models.DateTimeField(auto_now=True, verbose_name="Last Modified At")
    objects = TenantBaseManager()

    class Meta:
        abstract = True

    def save(self, *args, **kwargs):
        user = get_current_user()
        # Launch pad
        if user is None or user.is_anonymous:
            # Fired through signals apis and bots
            super(TenantBaseModel, self).save(*args, **kwargs)
        else:
            # logged in user
            self.created_by = user
            # self.updated_by = updated_by
            self.tenant = user.tenant
            super(TenantBaseModel, self).save(*args, **kwargs)

    def __str__(self):
        return self.name


class TenantInvitation(TenantBaseModel):
    token = models.CharField(max_length=256)
    email = models.EmailField()

    class Meta:
        verbose_name = "Tenant Invitation"
        verbose_name_plural = "Tenant Invitations"
        db_table = "tenant_invitation"

    def __str__(self):
        return self.email


class TenantIntegration(TenantBaseModel):
    tenant = models.OneToOneField(
        Tenant, on_delete=models.CASCADE, related_name="tenant_integrations", unique=True
    )
    name = models.CharField(max_length=255)
    props = models.JSONField()
    extra = models.JSONField()
    is_public = models.BooleanField(default=True)
    are_keys_safe = models.BooleanField(default=False)

    def __str__(self):
        return self.name


class TenantUserLoginConnection(TenantBaseModel):
    tenant_integration = models.ForeignKey(
        TenantIntegration,
        on_delete=models.CASCADE,
        null=True,
        related_name="user_login_connections",
    )
    medium = models.CharField(
        max_length=20,
        choices=(("Google", "google"), ("Microsoft", "microsoft")),
        default=None,
    )
    last_login_at = models.DateTimeField(default=timezone.now, null=True)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="user_login_connections",
    )
    token_data = models.JSONField(null=True)
    extra_data = models.JSONField(null=True)
