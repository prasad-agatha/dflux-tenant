from django.urls import path

from dflux.api.views import (
    TeamView,
    TeamMembersView,
    TeamDetailView,
    SendTeamInvitationToUsers,
)

urlpatterns = [
    # list of all the teams
    path("teams/", TeamView.as_view(), name="teams"),
    # get particular team members
    path("teams/<int:pk>/members/", TeamMembersView.as_view(), name="team-members"),
    # get particular team details
    path("teams/<int:pk>/", TeamDetailView.as_view(), name="team-details"),
    # send team invite to users
    path(
        "teams/<int:pk>/invite/",
        SendTeamInvitationToUsers.as_view(),
        name="team-invitation",
    ),
]
