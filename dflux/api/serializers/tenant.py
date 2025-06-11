from dflux.db.models import Tenant, TenantInvitation, TenantIntegration, TenantUser
from rest_framework import serializers


class TenantSerializer(serializers.ModelSerializer):
    class Meta:
        model = Tenant
        fields = "__all__"


class TenantInvitationSerializer(serializers.ModelSerializer):
    class Meta:
        model = TenantInvitation
        exclude = ("tenant",)


class TenantIntegrationSerializer(serializers.ModelSerializer):
    class Meta:
        model = TenantIntegration
        fields = "__all__"


class TenantUsersSerializer(serializers.Serializer):
    tenant_user = serializers.SerializerMethodField("get_user")

    def get_user(self, obj):
        try:
            user = TenantUser.objects.get(
                tenant=obj, tenant_superuser=True, is_superuser=False
            )
            return {
                "tenant_id": user.tenant.id,
                "name": user.tenant.name,
                "subdomain_prefix": user.tenant.subdomain_prefix,
                "tenant_created_at": user.tenant.created_at.strftime("%m/%d/%Y %H:%M"),
                "company": user.company,
                "email": user.email,
                "first_name": user.first_name,
                "last_name": user.last_name,
                "size": TenantUser.objects.filter(tenant=obj).count(),
                "status": user.is_email_verified,
                "interested_in": user.dflux_using,
            }
        except TenantUser.DoesNotExist:
            # return None
            return {
                "name": obj.name,
                "subdomain_prefix": obj.subdomain_prefix,
                "tenant_created_at": obj.created_at.strftime("%m/%d/%Y %H:%M"),
                "tenant_id": obj.id,
            }
