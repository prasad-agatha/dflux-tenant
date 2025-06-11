from django.db.models import Q
from django.http import Http404
from django.shortcuts import get_object_or_404

from rest_framework import status
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated

from dflux.api.serializers import (
    ProjectSerializer,
    ProjectDetailSerializer,
    ProjectTeamSerializer,
    AddProjectTeamSerializer,
    ProjectInvitationSerializer,
    ProjectMemberSerializer,
    AddProjectMembersSerializer,
)
from dflux.api.views.base import BaseAPIView
from dflux.api.views.utils import generate_token
from dflux.utils.emails import emails
from dflux.db.models import Project, ProjectTeam, ProjectMembers, ProjectInvitation

from .permissions import (
    ProjectOwnerOrCollaborator,
    ProjectMembersAccess,
    ProjectModuleAccess,
)
from django.utils.decorators import method_decorator
from django.views.decorators.cache import cache_page
from .filters import ProjectFilter, ProjectMemberFilter


class ProjectView(BaseAPIView):
    """
    API endpoint that allows view list of all the projects or create new project.
    * Requires JWT authentication.
    * This endpoint will allows only GET, POST methods.
    """

    permission_classes = [IsAuthenticated]

    def get(self, request):
        """
        This method will allows view list of all the projects.
        """

        queryset = ProjectFilter(
            request.GET,
            queryset=Project.objects.filter(
                Q(user=request.user) | Q(projectmembers__user=request.user)
            ).distinct(),
        ).qs
        serializer = ProjectSerializer(queryset, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def post(self, request):
        """
        This method will allows create a new project.
        """
        request.data["user"] = request.user.id
        project_serializer = ProjectSerializer(data=request.data)
        if project_serializer.is_valid(raise_exception=True):
            project = project_serializer.save()
            # add the project owner into the project members
            data = {"user": project.user.id, "project": project.id}
            project_member_serializer = AddProjectMembersSerializer(data=data)
            if project_member_serializer.is_valid(raise_exception=True):
                project_member_serializer.save()
                return Response(project_serializer.data)


class ProjectDetailView(BaseAPIView):
    """
    API endpoint that allows view, update, delete individual project details.
    * Requires JWT authentication.
    * Only collaborator or owner of the project can access.
    * This endpoint will allows only GET, PUT, DELETE methods.
    """

    permission_classes = (IsAuthenticated, ProjectModuleAccess)

    def get_object(self, request, pk):
        """
        This method will return the project object given pk value
        """

        try:
            project = Project.objects.get(id=pk)
            return project
        except Project.DoesNotExist:
            raise Http404

    def get(self, request, pk, format=None):
        """
        This method will allows view the individual project details.
        """
        project = self.get_object(request, pk)
        self.check_object_permissions(request, Project)
        serializer = ProjectDetailSerializer(project)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def put(self, request, pk, format=None):
        """
        This method will allows update the individual project details.
        """
        project = self.get_object(request, pk)
        serializer = ProjectSerializer(project, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, pk, format=None):
        """
        This method will allows delete the individual project details.
        """
        project = self.get_object(request, pk)
        project.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


class ProjectTeamView(BaseAPIView):
    """
    API endpoint that allows view list of all the projects teams or add new team to project.
    * Requires JWT authentication.
    * This endpoint will allows only GET, POST methods.
    """

    permission_classes = (IsAuthenticated,)

    def get(self, request, pk):
        """
        This method allows view list of all the projects teams.
        """
        project_teams = ProjectTeam.objects.select_related("project", "team").filter(
            project__id=pk
        )
        serializer = ProjectTeamSerializer(project_teams, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def post(self, request, pk):
        """
        This method allows add team in to the projects teams list.
        """
        request.data["project"] = pk
        serializer = AddProjectTeamSerializer(data=request.data)
        if serializer.is_valid(raise_exception=True):
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)


class ProjectTeamDetailView(BaseAPIView):
    """
    API endpoint that allows view, update, delete individual project team details.
    * Requires JWT authentication.
    * This endpoint will allows only GET, PUT, DELETE methods.
    """

    permission_classes = (IsAuthenticated,)

    def get_object(self, pk):
        """
        This method will allows return project team object given pk value.
        """
        try:
            return ProjectTeam.objects.select_related("project", "team").get(pk=pk)
        except ProjectTeam.DoesNotExist:
            raise Http404

    def get(self, request, pk):
        """
        This method will allows get the individual project team details.
        """
        project_team = self.get_object(pk)
        serializer = ProjectTeamSerializer(project_team)
        return Response(serializer.data)

    def put(self, request, pk):
        """
        This method will allows update the individual project team details.
        """
        project_team = self.get_object(pk)
        serializer = AddProjectTeamSerializer(project_team, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, pk):
        """
        This method will allows delete the individual project team details.
        """
        project_team = self.get_object(pk)
        project_team.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


class ProjectInvitations(BaseAPIView):
    """
    API endpoint that allows send project invitation emails to users.
    * Authentication not required.
    * This endpoint will allows only POST method.
    """

    def post(self, request, pk):
        project = get_object_or_404(Project, id=pk)
        access = request.data.get("access")
        module_access = request.data.get("module_access")
        users = [
            {
                "invitee": email,
                "project": project.id,
                "project_name": project.name,
                "token": generate_token(email, project.id),
                "access": access,
                "module_access": module_access,
            }
            for email in request.data.get("emails")
        ]
        for user in users:
            a = ProjectInvitation.objects.filter(
                project__id=project.id, invitee=user["invitee"]
            )
            a.delete()
        serializer = ProjectInvitationSerializer(data=users, many=True)
        if serializer.is_valid(raise_exception=True):
            serializer.save()
            try:
                emails.send_project_invitation_mail_to_users(users, request)
            except Exception as e:
                return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
            return Response({"message": "Invitation send to given users"})


class ProjectMembersView(BaseAPIView):
    """
    API endpoint that allows view list of all the projects members or add new member to project list.
    * Requires JWT authentication.
    * This endpoint will allows only GET, POST methods.
    """

    permission_classes = (ProjectMembersAccess,)

    def get(self, request, pk):
        """
        This method allows view list of all the projects members.
        """
        project = get_object_or_404(Project, id=pk)
        members = ProjectMemberFilter(
            request.GET, queryset=ProjectMembers.objects.filter(project__id=project.id)
        ).qs
        # self.check_object_permissions(request, Project)
        serializer = ProjectMemberSerializer(members, many=True)
        data = serializer.data[:]
        for list_item in data:
            invitation = ProjectInvitation.objects.filter(
                project__id=pk, invitee=list_item["user"]["email"]
            ).first()
            if invitation is not None:
                list_item.update(
                    {
                        "access": invitation.access,
                        "model_access": invitation.module_access,
                    }
                )
        return Response(serializer.data)

    def post(self, request, pk):
        """
        This method allows add new members to project list.
        """
        # add member to project
        data = {"user": request.data.get("user"), "project": pk}
        serializer = AddProjectMembersSerializer(data=data)
        if serializer.is_valid(raise_exception=True):
            serializer.save()
            return Response(serializer.data, status=status.HTTP_200_OK)


class ProjectMemberDetailView(BaseAPIView):
    """
    API endpoint that allows view, update, delete individual project member details.
    * Requires JWT authentication.
    * Only collaborator or owner of the project can access.
    * This endpoint will allows only GET, PUT, DELETE methods.
    """

    permission_classes = (IsAuthenticated, ProjectOwnerOrCollaborator)

    def get_object(self, pk):
        """
        This method allow get the project member object given pk value.
        """
        try:
            return ProjectMembers.objects.get(id=pk)
        except ProjectMembers.DoesNotExist:
            raise Http404

    def get(self, request, pk, pk1):
        """
        This method will allows get the individual project member details.
        """
        project_members = self.get_object(pk1)
        serializer = ProjectMemberSerializer(project_members)
        return Response(serializer.data)

    def put(self, request, pk, pk1):
        """
        This method will allows update the individual project member details.
        """
        project_members = self.get_object(pk1)
        a = ProjectInvitation.objects.filter(
            project__id=pk, invitee=project_members.user.email
        )
        for members in a:
            members.access = request.data.get("access")
            members.module_access = request.data.get("module_access")
            members.save()

        return Response({"msg": "access Updated"})

    def delete(self, request, pk, pk1):
        """
        This method will allows delete the individual project member details.
        """
        project_members = self.get_object(pk1)
        project_members.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


class UserProjectRoleView(BaseAPIView):
    """
    API endpoint that allows get the user role(like owner, collaborator).
    * Requires JWT authentication.
    * This endpoint will allows only GET method.
    """

    permission_classes = (IsAuthenticated,)

    def get(self, request, pk):
        print(request.user)
        members = ProjectMembers.objects.filter(
            project__id=pk, user=request.user
        ).first()
        serializer = ProjectMemberSerializer(members)
        return Response(
            {
                "user_email": serializer.data.get("user").get("email", None),
                "user_role": serializer.data.get("role", None),
                "user_access": serializer.data.get("access", None),
                "user_module_access": serializer.data.get("module_access", None),
            },
            status=status.HTTP_200_OK,
        )
