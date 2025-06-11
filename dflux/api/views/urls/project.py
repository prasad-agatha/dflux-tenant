from django.urls import path

from dflux.api.views import (
    ProjectView,
    ProjectDetailView,
    ProjectTeamView,
    ProjectTeamDetailView,
    ProjectInvitations,
    ProjectMembersView,
    ProjectMemberDetailView,
    UserProjectRoleView,
)

urlpatterns = [
    # get list of projects
    path("projects/", ProjectView.as_view(), name="project"),
    # # get particular project details
    path("projects/<int:pk>/", ProjectDetailView.as_view(), name="project_detail"),
    # get particular project teams
    path("projects/<int:pk>/teams/", ProjectTeamView.as_view(), name="project-teams"),
    # project invitefrom .employee import employee_status
    path(
        "projects/<int:pk>/invite/",
        ProjectInvitations.as_view(),
        name="project-invitation",
    ),
    # project members
    path(
        "projects/<int:pk>/members/",
        ProjectMembersView.as_view(),
        name="project-members",
    ),
    path(
        "projects/<int:pk>/members/<int:pk1>",
        ProjectMemberDetailView.as_view(),
        name="project-members-details",
    ),
    path(
        "projects/<int:pk>/role/",
        UserProjectRoleView.as_view(),
        name="user-project-role",
    ),
]
