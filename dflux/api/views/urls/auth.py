from django.urls import path

from rest_framework_simplejwt.views import TokenRefreshView

from dflux.api.views import TenantEndpoint, TenantInvitationView
from dflux.api.views import (
    UserDetailView,
    UserSignInView,
    UserSignUpView,
    TenantUsersView,
    ActivateUserView,
    OauthEndpoint,
    TenantAvailabilityEndpoint,
    TenantIntegrationEndpoint,
    TenantIntegrationDetailsEndpoint,
    SetUserPassword,
    UserOnBoardAPIView,
    ValidateUserByEmail,
    UpdatedTenantUserDetailView,
    GetUserTenantEndpoint,
    CheckEmailAvailableEndpoint,
    DeleteTenantUserView,
)

from dflux.api.views.verify_email import VerifyEmailEndpoint

from dflux.api.views.password import (
    PasswordResetView,
    PasswordResetConfirmView,
    ChangePasswordView,
)

tenant_urlpatterns = [
    path("oauth/", OauthEndpoint.as_view(), name="oauth"),
    path("tenants-availability/", TenantAvailabilityEndpoint.as_view()),
    path(
        "tenants/",
        TenantEndpoint.as_view({"get": "list", "post": "post"}),
    ),
    path(
        "tenants/<str:subdomain_prefix>/",
        TenantEndpoint.as_view(
            {"get": "retrieve", "put": "partial_update", "delete": "destroy"}
        ),
    ),
    path("tenants/<int:pk>/integrations/", TenantIntegrationEndpoint.as_view()),
    path("tenants/integrations/<int:pk>/", TenantIntegrationDetailsEndpoint.as_view()),
    path("tenant-invitation/", TenantInvitationView.as_view()),
    path("user-tenant/", GetUserTenantEndpoint.as_view()),
]
auth_urlpatterns = [
    path("tenants/<int:pk>/users/", UserSignUpView.as_view(), name="create"),
    path(
        "tenants/<str:subdomain_prefix>/users/<str:username>/set-password/",
        SetUserPassword.as_view(),
        name="set_user_password",
    ),
    path(
        "tenants/<str:subdomain_prefix>/on-board/",
        UserOnBoardAPIView.as_view(),
        name="user_onboard",
    ),
    # user login
    path(
        "tenants/<int:pk>/signin/", UserSignInView.as_view(), name="token_obtain_pair"
    ),
    # user access token refresh
    path("token-refresh/", TokenRefreshView.as_view(), name="token_refresh"),
    # get list of users
    path("userslist/", TenantUsersView.as_view(), name="tenant-users"),
    # get  particular user details
    path("users/me/", UserDetailView.as_view(), name="retreive user"),
    path(
        "users/<int:pk>/delete/",
        DeleteTenantUserView.as_view(),
        name="delete user",
    ),
    path(
        "users/<int:pk>/details/",
        UpdatedTenantUserDetailView.as_view(),
        name="retreive user",
    ),
    path(
        "users/<int:id>/activate/",
        ActivateUserView.as_view(),
        name="active-user",
    ),
    # verify email
    path(
        "tenants/<int:pk>/verify/email/",
        VerifyEmailEndpoint.as_view(),
        name="verify-email",
    ),
    # password
    path(
        "tenants/<int:pk>/password-reset/",
        PasswordResetView.as_view(),
        name="password-reset",
    ),
    path(
        "tenants/<int:pk>/password-reset/confirm/",
        PasswordResetConfirmView.as_view(),
        name="password-reset-confirm",
    ),
    path(
        "tenants/<int:tenant_id>/change-password/<int:pk>/",
        ChangePasswordView.as_view(),
        name="auth_change_password",
    ),
    path(
        "validate-user/", ValidateUserByEmail.as_view(), name="validate_user_by_email"
    ),
    path(
        "check-email/", CheckEmailAvailableEndpoint.as_view(), name="email-avalability"
    ),
]

urlpatterns = tenant_urlpatterns + auth_urlpatterns
