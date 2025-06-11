from rest_framework import permissions, status
from rest_framework.exceptions import APIException
from rest_framework.serializers import ValidationError

from dflux.db.models import (
    Project,
    Query,
    Charts,
    JsonData,
    ExcelData,
    GoogleSheet,
    DashBoard,
    Connection,
    ChartTrigger,
    ProjectMembers,
    ProjectInvitation,
)


class DomainNotValidException(APIException):
    """
    - Custom exception class
    - If domain name not in ["@intellectdata.com", "@soulpageit.com"] it will raise an exception
    """

    status_code = status.HTTP_403_FORBIDDEN
    default_detail = "Your domain must ends with @intellectdata.com or @soulpageit.com"


class DomainValidationPermissions(permissions.BasePermission):
    """
    - Custom permission class
    - If user email not in [@intellectdata.com or @soulpageit.com"] it will raise an exception
    """

    def has_permission(self, request, view):
        email = request.user.email
        if email.endswith("@intellectdata.com") or email.endswith("@soulpageit.com"):
            return True
        raise DomainNotValidException


class ProjectOwnerOrCollaboratorException(APIException):
    """
    - Custom exception class
    - If user is not collaborator or owner of the project it will raise an exception
    """

    status_code = status.HTTP_403_FORBIDDEN
    default_detail = "user must be owner or collaborator of the project"


class ReadWriteException(APIException):
    """
    - Custom exception class
    - If user don't have WRITE it will raise an exception
    """

    status_code = status.HTTP_200_OK
    default_detail = "user must have WRITE access."


def is_project_member(request, project, project_id):
    project_member = ProjectMembers.objects.filter(
        project__id=project_id, user=request.user.id
    ).first()
    # if request user is project owner
    if project.user == request.user:
        return True
    if project_member:
        invitation = ProjectInvitation.objects.filter(
            project__id=project_id, invitee=project_member.user.email
        ).first()
        if invitation.access == "WRITE":
            return True
        else:
            raise ReadWriteException
    else:
        raise ProjectOwnerOrCollaboratorException


class ProjectOwnerOrCollaborator(permissions.BasePermission):
    """
    Checking current user is project owner or collaborator
    """

    def has_permission(self, request, view):
        # value getting from url parameter
        project_id = view.kwargs.get("pk")
        try:
            project = Project.objects.get(id=project_id)
        except:
            raise ValidationError({"error": "please provide valid project id"})
        return is_project_member(request, project, project_id)

    # project object level permission
    def has_object_permission(self, request, view, obj):
        """if user is project owner or member then provide permissions else not"""
        if obj.user == request.user:
            return True
        if ProjectMembers.objects.filter(
            project__id=view.kwargs.get("pk"), user=request.user
        ).exists():
            return True
        raise ProjectOwnerOrCollaboratorException


class ProjectChartAccess(permissions.BasePermission):
    def has_permission(self, request, view):
        chart_id = view.kwargs.get("pk")
        try:
            chart = Charts.objects.get(id=chart_id)
        except:
            raise ValidationError({"error": "please provide valid chart id"})
        return is_project_member(request, chart.project, chart.project.id)


class ProjectChartTriggerAccess(permissions.BasePermission):
    def has_permission(self, request, view):
        chart_trigger_id = view.kwargs.get("pk")
        try:
            chart_trigger = ChartTrigger.objects.get(id=chart_trigger_id)
        except:
            raise ValidationError({"error": "please provide valid chart id"})
        return is_project_member(
            request, chart_trigger.project, chart_trigger.project.id
        )


class ProjectQueryAccess(permissions.BasePermission):
    def has_permission(self, request, view):
        query_id = view.kwargs.get("pk")
        try:
            query = Query.objects.get(id=query_id)
        except:
            raise ValidationError({"error": "please provide valid query id"})
        return is_project_member(request, query.project, query.project.id)


class ProjectExecuteQueryAccess(permissions.BasePermission):
    def has_permission(self, request, view):
        connection_id = request.data.get("connection_id")
        try:
            connection = Connection.objects.get(pk=connection_id)
        except:
            raise ValidationError({"error": "please provide valid connection id"})
        return is_project_member(request, connection.project, connection.project.id)


class ProjectSchemaAccess(permissions.BasePermission):
    def has_permission(self, request, view):
        connection_id = view.kwargs.get("pk")
        try:
            connection = Connection.objects.get(pk=connection_id)
        except:
            raise ValidationError({"error": "please provide valid connection id"})
        return is_project_member(request, connection.project, connection.project.id)


class ProjectGooglesheetAccess(permissions.BasePermission):
    def has_permission(self, request, view):
        google_sheet_id = view.kwargs.get("pk")
        try:
            google_sheet = GoogleSheet.objects.get(id=google_sheet_id)
        except:
            raise ValidationError({"error": "please provide valid sheet id"})
        return is_project_member(request, google_sheet.project, google_sheet.project.id)


class ProjectDashboardAccess(permissions.BasePermission):
    def has_permission(self, request, view):
        dashboard_id = view.kwargs.get("pk")
        try:
            dashboard = DashBoard.objects.get(id=dashboard_id)
        except:
            raise ValidationError({"error": "please provide valid dashboard id"})
        return is_project_member(request, dashboard.project, dashboard.project.id)


class ProjectDataModelAccess(permissions.BasePermission):
    def has_permission(self, request, view):
        project_id = view.kwargs.get("project_id")
        try:
            project = Project.objects.get(id=project_id)
        except:
            raise ValidationError({"error": "please provide valid project id"})
        return is_project_member(request, project, project_id)


class ProjectExcelDataAccess(permissions.BasePermission):
    def has_permission(self, request, view):
        excel_id = view.kwargs.get("pk")
        try:
            excel = ExcelData.objects.get(id=excel_id)
        except:
            raise ValidationError({"error": "please provide valid excel id"})
        return is_project_member(request, excel.project, excel.project.id)


class ProjectJsonDataAccess(permissions.BasePermission):
    def has_permission(self, request, view):
        json_id = view.kwargs.get("pk")
        try:
            json = JsonData.objects.get(id=json_id)
        except:
            raise ValidationError({"error": "please provide valid json id"})
        return is_project_member(request, json.project, json.project.id)


class ProjectConnectionAccess(permissions.BasePermission):
    def has_permission(self, request, view):
        connection_id = view.kwargs.get("pk")
        try:
            connection = Connection.objects.get(id=connection_id)
        except:
            raise ValidationError({"error": "please provide valid connection id"})
        return is_project_member(request, connection.project, connection.project.id)


class ProjectMembersAccess(permissions.BasePermission):
    def has_permission(self, request, view):
        SAFE_METHODS = ["GET", "HEAD", "OPTIONS"]
        project_id = view.kwargs.get("pk")
        if (
            request.method in SAFE_METHODS
            or request.user
            and request.user.is_authenticated
        ):
            return True
        else:
            try:
                project = Project.objects.get(id=project_id)
            except:
                raise ValidationError({"error": "please provide valid project id"})
            return is_project_member(request, project, project_id)


class ProjectModuleAccess(permissions.BasePermission):
    def has_permission(self, request, view):
        SAFE_METHODS = ["GET", "HEAD", "OPTIONS"]
        project_id = view.kwargs.get("pk")
        if (
            request.method in SAFE_METHODS
            or request.user
            and request.user.is_authenticated
        ):
            return True
        else:
            try:
                project = Project.objects.get(id=project_id)
            except:
                raise ValidationError({"error": "please provide valid project id"})
            return is_project_member(request, project, project_id)
