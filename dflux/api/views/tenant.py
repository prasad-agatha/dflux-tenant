from django.db import IntegrityError
from rest_framework import viewsets
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import (
    BasePermission,
    SAFE_METHODS,
    IsAuthenticated,
    AllowAny,
)
from dflux.db.models import Tenant, TenantInvitation, TenantIntegration, TenantUser
from dflux.api.serializers import (
    TenantSerializer,
    TenantInvitationSerializer,
    TenantIntegrationSerializer,
    TenantUsersSerializer,
)

import uuid
from rest_framework import status
from dflux.utils.emails import emails


class IsAdminOrReadOnly(BasePermission):
    def has_permission(self, request, view):
        if request.method in SAFE_METHODS:
            return True

        return bool(request.user and request.user.is_superuser)


class TenantAvailabilityEndpoint(APIView):
    def get(self, request):
        name = request.query_params.get("q")
        if Tenant.objects.filter(subdomain_prefix__icontains=name).exists():
            return Response({"available": True})
        else:
            return Response({"available": False})


class TenantEndpoint(viewsets.ModelViewSet):
    """
    API endpoint that allows users to be viewed or edited.
    """

    queryset = Tenant.objects.all()
    serializer_class = TenantSerializer
    lookup_field = "subdomain_prefix"
    permission_classes = [IsAuthenticated]

    def list(self, request):
        queryset = Tenant.objects.all()
        serializer = TenantUsersSerializer(queryset, many=True)
        return Response(serializer.data)

    def post(self, request):
        tenant = request.data.get("name")
        subdomain_prefix = request.data.get("subdomain_prefix")
        email = request.data.get("email")
        first_name = request.data.get("admin_name")
        company = request.data.get("company")

        missing_fields = {}

        if not tenant:
            missing_fields.update({"name": "Required Field"})
        if not subdomain_prefix:
            missing_fields.update({"subdomain_prefix": "Required Field"})
        if not email:
            missing_fields.update({"email": "Required Field"})
        if not first_name:
            missing_fields.update({"admin_name": "Required Field"})
        if not company:
            missing_fields.update({"company": "Required Field"})

        if missing_fields:
            return Response(missing_fields, status=400)

        try:
            tenant = Tenant.objects.create(
                name=tenant, subdomain_prefix=subdomain_prefix
            )
        except IntegrityError:
            return Response(
                {"subdomain_prefix": "Prefix is already taken!"}, status=400
            )

        user = TenantUser.objects.create(
            email=email,
            tenant=tenant,
            first_name=first_name,
            company=company,
            tenant_superuser=True,
        )
        emails.send_tenant_registration_email(user)

        return Response({"message": "Tenant Registration Successfull!"})

    def get_permissions(self, *args, **kwargs):
        if self.request.method in ["PUT", "GET"]:
            return [AllowAny()]
        else:
            return super().get_permissions()


class TenantInvitationView(APIView):
    permission_classes = (IsAuthenticated,)

    def get(self, request):
        queryset = TenantInvitation.objects.all()
        serializer = TenantInvitationSerializer(queryset, many=True)
        return Response(serializer.data)

    def post(self, request):
        token = str(uuid.uuid4().hex + uuid.uuid4().hex)
        request.data["token"] = token
        to_mail = request.data.get("email")
        serializer = TenantInvitationSerializer(data=request.data)
        if serializer.is_valid(raise_exception=True):
            emails.tenant_invitation_mail(to_mail=to_mail, token=token, request=request)
            serializer.save()
            return Response({"message": "Inviataion email sent."})


class TenantIntegrationEndpoint(APIView):
    def get(self, request, pk):
        queryset = TenantIntegration.objects.filter(tenant__id=pk)
        serializer = TenantIntegrationSerializer(queryset, many=True)
        return Response(serializer.data)

    def post(self, request, pk):
        request.data["tenant"] = pk
        serializer = TenantIntegrationSerializer(data=request.data)
        if serializer.is_valid(raise_exception=True):
            serializer.save()
            return Response(serializer.data)


class TenantIntegrationDetailsEndpoint(APIView):
    def get(self, request, pk):
        queryset = TenantIntegration.objects.get(id=pk)
        serializer = TenantIntegrationSerializer(queryset)
        return Response(serializer.data)

    def put(self, request, pk):
        queryset = TenantIntegration.objects.get(id=pk)
        serializer = TenantIntegrationSerializer(
            queryset, data=request.data, partial=True
        )
        if serializer.is_valid(raise_exception=True):
            serializer.save()
            return Response(serializer.data)

    def delete(self, request, pk):
        queryset = TenantIntegration.objects.get(id=pk)
        queryset.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


class GetUserTenantEndpoint(APIView):
    def post(self, request):
        queryset = TenantUser.objects.filter(email=request.data.get("email")).first()
        if queryset is not None:
            integratons_queryset = TenantIntegration.objects.filter(tenant=queryset.tenant).first()
            integrations = None
            if integratons_queryset is not None:
                integrations_serializer =  TenantIntegrationSerializer(integratons_queryset)
                integrations = integrations_serializer.data
            return Response(
                {
                    "id": queryset.tenant.id,
                    "name": queryset.tenant.name,
                    "subdomain_prefix": queryset.tenant.subdomain_prefix,
                    "last_login_medium": queryset.last_login_medium,
                    "integrations": integrations,
                }
            )
        else:
            return Response(
                {"msg": "user not found"}, status=status.HTTP_400_BAD_REQUEST
            )
