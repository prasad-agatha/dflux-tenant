from rest_framework import status
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated

from django.http import Http404
from django.contrib.auth.models import User

from dflux.db.models import Team, TeamMembers
from dflux.api.views.base import BaseAPIView
from dflux.utils.emails import emails
from dflux.api.serializers import (
    TeamSerializer,
    TeamMembersSerializer,
    AddTeamMembersSerializer,
    TeamInvitationSerializer,
)


class TeamView(BaseAPIView):
    """
    API endpoint that allows view list of all the teams or create new team.

    * Requires JWT authentication.
    * This endpoint will allows only GET, POST methods.
    """

    permission_classes = (IsAuthenticated,)

    def get(self, request):
        """
        This method allows view list of all the teams
        """
        teams = Team.objects.all()
        serializer = TeamSerializer(teams, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def post(self, request):
        """
        This method allows us to create new team.
        """
        serializer = TeamSerializer(data=request.data)
        if serializer.is_valid(raise_exception=True):
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)


class TeamDetailView(BaseAPIView):
    """
    API endpoint that allows view, update, delete individual team details.

    * Requires JWT authentication.
    * Only collaborator or owner of the project can access.
    * This endpoint will allows only GET, PUT, DELETE methods.
    """

    permission_classes = (IsAuthenticated,)

    def get_object(self, pk):
        """
        This method will return team object given pk value.
        """
        try:
            return Team.objects.get(pk=pk)
        except Team.DoesNotExist:
            raise Http404

    def get(self, request, pk):
        """
        This method will allows view individual team details.
        """
        team = self.get_object(pk)
        serializer = TeamSerializer(team)
        return Response(serializer.data)

    def put(self, request, pk):
        """
        This method will allows update individual team details.
        """
        team = self.get_object(pk)
        serializer = TeamSerializer(team, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, pk):
        """
        This method will allows delete individual team details.
        """
        team = self.get_object(pk)
        team.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


def check_user_status(email):
    """
    This method will allows check the user email exits or not in database.
    - if user email exits it will return True else it will return False
    """
    if User.objects.filter(email=email).exists():
        return True
    return False


def genarate_token(user, team):
    """
    This method will allows generate JWT token using the user object, team.
    """
    import jwt

    JWT_SECRET = "soulpage"
    JWT_ALGORITHM = "HS256"
    payload = {"user": user, "team": team, "user_status": check_user_status(user)}
    jwt_token = jwt.encode(payload, JWT_SECRET, JWT_ALGORITHM, {"exp": "24hr"})
    return str(jwt_token).strip("b").strip("'")


class SendTeamInvitationToUsers(BaseAPIView):
    """
    API endpoint that allows send team invitation emails to users.

    * Authentication not required.
    * This endpoint will allows only POST method.
    """

    def post(self, request, pk):
        team = Team.objects.get(id=pk)
        users = [
            {"user": email, "team": team.id, "token": genarate_token(email, team.id)}
            for email in request.data.get("emails")
        ]
        serializer = TeamInvitationSerializer(data=users, many=True)
        if serializer.is_valid(raise_exception=True):
            serializer.save()
            emails.send_team_invitation_mail_to_users(users, request)
            return Response({"message": "Invitation send to given users"})


class TeamMembersView(BaseAPIView):
    """
    API endpoint that allows view list of all the team members or add new member to team.

    * Requires JWT authentication.
    * This endpoint will allows only GET, POST methods.
    """

    permission_classes = (IsAuthenticated,)

    def get(self, request, pk):
        """
        This method allows view list of all the team members.
        """
        team_members = TeamMembers.objects.filter(team__id=pk)
        serializer = TeamMembersSerializer(team_members, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def post(self, request, pk):
        """
        This method allows add new member to team.
        """
        data = {"user": request.user.id, "team": pk}
        serializer = AddTeamMembersSerializer(data=data)
        if serializer.is_valid(raise_exception=True):
            serializer.save()
            return Response(serializer.data, status=status.HTTP_200_OK)
