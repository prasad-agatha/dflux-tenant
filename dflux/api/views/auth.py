from rest_framework import status
from rest_framework import serializers
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated, IsAdminUser
from django.shortcuts import get_object_or_404


from django.utils import timezone

from dflux.db.models import Tenant, TenantUser, TenantInvitation

from dflux.utils.emails import emails
from dflux.api.views.base import BaseAPIView
from dflux.api.serializers import UserSerializer, UserDetailsSerializer


from .filters import UserFilter
from .utils import get_tokens_for_user
from .permissions import DomainValidationPermissions

import uuid
from rest_framework import exceptions
from rest_framework.views import APIView
from rest_framework.permissions import AllowAny
from sentry_sdk import capture_exception, capture_message

# sso authentication
from google.oauth2 import id_token
from google.auth.transport import requests as google_auth_request
from dflux.db.models import TenantUserLoginConnection, TenantIntegration


class UserSignUpView(BaseAPIView):
    """
    API endpoint that allows register new user.

    * Authentication not required.
    * This endpoint will allows only POST method.
    """

    def post(self, request, pk):
        import uuid

        try:
            tenant = Tenant.objects.get(id=pk)
        except Tenant.DoesNotExist:
            raise serializers.ValidationError(
                {"message": f" {pk} is not valid tenant id."}
            )

        try:
            email = request.data.get("email")
        except KeyError:
            return Response({"error": "missing required fields"})
        # check email is exists or not
        if TenantUser.objects.filter(email=email, tenant=tenant).exists():
            raise serializers.ValidationError({"message": f"{email} already taken."})
        token = request.data.get("token", None)
        first_name = request.data.get("first_name", None)
        last_name = request.data.get("last_name", None)
        if token:
            if TenantInvitation.objects.filter(token=token, email=email).exists():
                user = TenantUser.objects.create(
                    email=email,
                    tenant=tenant,
                    first_name=first_name,
                    last_name=last_name,
                )
                user.set_password(request.data.get("password"))
                user.save()
                # save user registration token
                emails.send_registration_email(request, user.token, tenant)
                jwt_token = get_tokens_for_user(user)
                jwt_token["email"] = request.data.get("email")
                return Response(jwt_token)
            else:
                return Response(
                    {"message": "token not valid"}, status=status.HTTP_400_BAD_REQUEST
                )
        else:
            user = TenantUser.objects.create(
                email=email, tenant=tenant, first_name=first_name, last_name=last_name
            )
            user.set_password(request.data.get("password"))
            user.save()
            # save user registration token
            emails.send_registration_email(request, user.token, tenant)
            jwt_token = get_tokens_for_user(user)
            jwt_token["email"] = request.data.get("email")
            return Response(jwt_token)


class UserSignInView(BaseAPIView):
    """
    API endpoint that allows to login user.
    - Once user is logged in this will generate new access , refresh tokens.

    * Authentication not required.
    * This endpoint will allows only POST method.
    """

    def post(self, request, pk):
        try:
            email = request.data.get("email")
            password = request.data.get("password")
        except KeyError:
            return Response(
                {"error": "email, password required"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        try:
            user = TenantUser.objects.get(email=email, tenant_id=pk)
            verified = user.check_password(password)
            if verified:
                jwt_token = get_tokens_for_user(user)
                return Response(jwt_token)
            else:
                return Response(
                    {"error": "please provide valid password"},
                    status=status.HTTP_401_UNAUTHORIZED,
                )
        except:
            return Response(
                {"error": "Please check your email and password."},
                status=status.HTTP_401_UNAUTHORIZED,
            )


class UserDetailView(BaseAPIView):
    """
    API endpoint that allows view, update, delete individual user details.

    * Requires JWT authentication.
    * Only collaborator or owner of the project can access.
    * This endpoint will allows only GET, PUT, DELETE methods.
    """

    permission_classes = (IsAuthenticated,)

    def get(self, request):
        """
        This method allows view individual user details.
        """
        serializer = UserDetailsSerializer(request.user)
        return Response(serializer.data)

    def put(self, request):
        """
        This method allows update individual user details.
        """
        user = request.user
        serializer = UserSerializer(user, data=request.data, partial=True)
        if serializer.is_valid(raise_exception=True):
            serializer.save()
        return Response(serializer.data)

    # def delete(self, request):
    #     """
    #     This method allows delete individual user details.
    #     """
    #     user_delete = request.user
    #     user_delete.delete()
    #     return Response({"message": "Delete Success"}, status=status.HTTP_200_OK)


class DeleteTenantUserView(BaseAPIView):
    """
    API endpoint that allows view, update, delete individual user details.

    * Requires JWT authentication.
    * Only collaborator or owner of the project can access.
    * This endpoint will allows only GET, PUT, DELETE methods.
    """

    permission_classes = (IsAuthenticated,)

    def delete(self, request, pk):
        """
        This method allows delete individual user details.
        """
        if request.user.tenant_superuser:
            user = get_object_or_404(TenantUser, id=pk)
            user.delete()
            return Response({"message": "Delete Success"}, status=status.HTTP_200_OK)
        else:
            return Response(
                {"msg": "don't have permission to delete user"},
                status=status.HTTP_400_BAD_REQUEST,
            )


class UpdatedTenantUserDetailView(BaseAPIView):
    """
    API endpoint that allows view, update, delete individual user details.

    * Requires JWT authentication.
    * Only collaborator or owner of the project can access.
    * This endpoint will allows only GET, PUT, DELETE methods.
    """

    # permission_classes = (IsAuthenticated,)

    def put(self, request, pk):
        """
        This method allows update individual user details.
        """
        user = get_object_or_404(TenantUser, id=pk)
        serializer = UserSerializer(user, data=request.data, partial=True)
        if serializer.is_valid(raise_exception=True):
            serializer.save()
        return Response(serializer.data)


class ActivateUserView(BaseAPIView):
    """
    API endpoint that allows user is active or not.
    - If user is active it will return True else it will return False

    * This endpoint will allows only GET method.
    """

    permission_classes = (IsAdminUser,)

    def get(self, request, id):
        user = TenantUser.objects.filter(id=id).first()
        user.is_active = True
        if user is not None:
            user.extended_date = timezone.now()
            user.save()
        return Response({"message": "user activated"})


class TenantUsersView(BaseAPIView):
    """
    API endpoint that allows view list of all users in the databases.

    * Requires JWT authentication.
    * This endpoint will allows only GET method.
    """

    permission_classes = (IsAuthenticated,)
    # permission_classes = (IsAuthenticated, DomainValidationPermissions)

    def get(self, request):
        queryset = UserFilter(
            request.GET, queryset=TenantUser.objects.filter(tenant=request.user.tenant)
        ).qs
        serializer = UserSerializer(queryset, many=True)
        return Response(serializer.data)


def validate_google_token(token, client_id):
    try:
        id_info = id_token.verify_oauth2_token(
            token, google_auth_request.Request(), client_id
        )
        email = id_info.get("email")
        first_name = id_info.get("given_name")
        last_name = id_info.get("family_name", "")
        data = {
            "email": email,
            "first_name": first_name,
            "last_name": last_name,
        }
        return data
    except ValueError:
        raise exceptions.AuthenticationFailed("Error with Google connection.")


class OauthEndpoint(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        try:
            medium = request.data.get("medium", False)
            tenant_name = request.data.get("tenant", False)
            id_token = request.data.get("token", False)
            if not medium or not tenant_name or not id_token:
                return Response(
                    {
                        "error": "Something went wrong. Please try again later or contact the support team."
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                )

            tenant = Tenant.objects.get(subdomain_prefix=tenant_name)
            if medium == "google":
                try:
                    tenant_integration_google = TenantIntegration.objects.get(
                        tenant=tenant, name="google"
                    )
                    if (
                        tenant_integration_google.props["enabled"] == 1
                        and type(tenant_integration_google.props["key"]["clientId"])
                        == str
                    ):
                        client_id = tenant_integration_google.props["key"]["clientId"]
                    else:
                        return Response(
                            {
                                "error": "Google integration is not enabled. Please contact your site administrator."
                            },
                            status=status.HTTP_400_BAD_REQUEST,
                        )
                except Exception as e:
                    capture_exception(e)
                    return Response(
                        {
                            "error": "Something went wrong. Please try again later or contact the support team."
                        },
                        status=status.HTTP_400_BAD_REQUEST,
                    )

                data = validate_google_token(id_token, client_id)
                print(data)

            email = data.get("email", None)
            if email == None:
                capture_message("Google did not return email")
                return Response(
                    {
                        "error": "Something went wrong. Please try again later or contact the support team."
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                )

            if "@" in email:
                user = TenantUser.objects.get(
                    email=email, tenant__subdomain_prefix=tenant_name
                )
                email = data["email"]
                channel = "email"
                mobile_number = uuid.uuid4().hex
                email_verified = True
            else:
                capture_message("Google returned invalid email")
                return Response(
                    {
                        "error": "Something went wrong. Please try again later or contact the support team."
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                )

            ## Login Case

            if not user.is_active:
                return Response(
                    {
                        "error": "Your account has been deactivated. Please contact your site administrator."
                    },
                    status=status.HTTP_403_FORBIDDEN,
                )

            user.last_active = timezone.now()
            user.last_login_time = timezone.now()
            user.last_login_ip = request.META.get("REMOTE_ADDR")
            user.last_login_medium = f"oauth"
            user.last_login_uagent = request.META.get("HTTP_USER_AGENT")
            user.is_email_verified = email_verified
            user.set_token()
            user.save()

            serialized_user = UserSerializer(user).data

            token_data = get_tokens_for_user(user)

            data = {
                "access": token_data.get("access"),
                "refresh": token_data.get("refresh"),
                "user": serialized_user,
            }

            TenantUserLoginConnection.objects.update_or_create(
                medium=medium,
                extra_data={},
                tenant=tenant,
                user=user,
                defaults={
                    "token_data": {"id_token": id_token},
                    "last_login_at": timezone.now(),
                },
            )

            return Response(data, status=status.HTTP_200_OK)

        except TenantUser.DoesNotExist:
            ## Signup Case

            username = uuid.uuid4().hex

            if "@" in email:
                email = data["email"]
                mobile_number = uuid.uuid4().hex
                channel = "email"
                email_verified = True
            else:
                capture_message("Google returned invalid email")
                return Response(
                    {
                        "error": "Something went wrong. Please try again later or contact the support team."
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                )

            user = TenantUser(
                username=username,
                email=email,
                mobile_number=mobile_number,
                tenant=tenant,
                first_name=data["first_name"],
                last_name=data["last_name"],
                is_email_verified=email_verified,
                is_password_autoset=True,
            )

            user.set_password(uuid.uuid4().hex)
            user.is_password_autoset = True
            user.last_active = timezone.now()
            user.last_login_time = timezone.now()
            user.last_login_ip = request.META.get("REMOTE_ADDR")
            user.last_login_medium = "oauth"
            user.last_login_uagent = request.META.get("HTTP_USER_AGENT")
            user.token_updated_at = timezone.now()
            user.set_token()
            user.save()
            serialized_user = UserSerializer(user).data

            access_token, refresh_token = get_tokens_for_user(user)
            data = {
                "access_token": access_token,
                "refresh_token": refresh_token,
                "user": serialized_user,
                "permissions": [],
            }

            TenantUserLoginConnection.objects.update_or_create(
                medium=medium,
                extra_data={},
                tenant=tenant,
                user=user,
                defaults={
                    "token_data": {"id_token": id_token},
                    "last_login_at": timezone.now(),
                },
            )
            return Response(data, status=status.HTTP_201_CREATED)
        except Exception as e:
            capture_exception(e)
            return Response(
                {
                    "error": "Something went wrong. Please try again later or contact the support team."
                },
                status=status.HTTP_400_BAD_REQUEST,
            )


class SetUserPassword(APIView):
    def post(self, request, subdomain_prefix, username):
        token = request.data.get("token")
        password = request.data.get("password")
        confirm_password = request.data.get("confirm_password")

        missing_fields = {}
        if not token:
            missing_fields.update({"token": "required field"})
        if not password:
            missing_fields.update({"password": "required field"})
        if not confirm_password:
            missing_fields.update({"confirm_password": "required field"})

        if missing_fields:
            return Response(missing_fields, status=400)

        if password != confirm_password:
            return Response({"message": "Passwords do not match"}, status=400)
        try:
            user = TenantUser.objects.get(
                username=username,
                token=token,
                tenant__subdomain_prefix=subdomain_prefix,
            )
            user.set_password(password)
            user.token_updated_at = timezone.now()
            user.reset_token()
            user.is_email_verified = True
            user.save()
        except TenantUser.DoesNotExist:
            return Response({"message": "User not found!"}, status=400)
        return Response(
            {
                "message": "Password Updated",
                "data": {
                    "id": user.id,
                    "username": user.username,
                    "tanant_name": user.tenant.subdomain_prefix,
                },
            },
            status=200,
        )


class UserOnBoardAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, subdomain_prefix):
        queryset = get_object_or_404(
            TenantUser, pk=request.user.id, tenant__subdomain_prefix=subdomain_prefix
        )
        data = request.data
        data["has_onboard"] = True
        serializer = UserSerializer(queryset, data=data, partial=True)
        if serializer.is_valid(raise_exception=True):
            serializer.save()
        return Response(serializer.data, status=200)


class ValidateUserByEmail(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        queryset = get_object_or_404(TenantUser, email=request.data.get("email"))
        return Response(
            {
                "tenant_id": queryset.tenant.id,
                "tenant": queryset.tenant.name,
                "subdomain_prefix": queryset.tenant.subdomain_prefix,
            },
            status=200,
        )


class CheckEmailAvailableEndpoint(APIView):
    def post(self, request):
        if TenantUser.objects.filter(email=request.data.get("email")).exists():
            return Response({"available": True})
        else:
            return Response({"available": False})
