import django_filters

# notebook
from dflux.db.models import (
    Project,
    Connection,
    Query,
    DataModel,
    Charts,
    ChartTrigger,
    DashBoard,
    ProjectMembers,
    ExcelData,
    JsonData,
)
from django.db.models import Q
from dflux.db.models import TenantUser


class ProjectFilter(django_filters.FilterSet):
    search = django_filters.CharFilter(field_name="name", lookup_expr="icontains")

    class Meta:
        model = Project
        fields = ["search"]


class ConnectionFilter(django_filters.FilterSet):
    search = django_filters.CharFilter(field_name="name", lookup_expr="icontains")

    class Meta:
        model = Connection
        fields = ["search"]


class ExcelDataFilter(django_filters.FilterSet):
    search = django_filters.CharFilter(field_name="tablename", lookup_expr="icontains")

    class Meta:
        model = ExcelData
        fields = ["search"]


class JsonDataFilter(django_filters.FilterSet):
    search = django_filters.CharFilter(field_name="tablename", lookup_expr="icontains")

    class Meta:
        model = JsonData
        fields = ["search"]


class QueryFilter(django_filters.FilterSet):
    search = django_filters.CharFilter(field_name="name", lookup_expr="icontains")

    class Meta:
        model = Query
        fields = ["search"]


class DataModelFilter(django_filters.FilterSet):
    search = django_filters.CharFilter(field_name="name", lookup_expr="icontains")

    class Meta:
        model = DataModel
        fields = ["search"]


class ChartsFilter(django_filters.FilterSet):
    search = django_filters.CharFilter(field_name="name", lookup_expr="icontains")

    class Meta:
        model = Charts
        fields = ["search"]


class ChartTriggerFilter(django_filters.FilterSet):
    search = django_filters.CharFilter(field_name="name", lookup_expr="icontains")

    class Meta:
        model = ChartTrigger
        fields = ["search"]


class DashboardFilter(django_filters.FilterSet):
    search = django_filters.CharFilter(field_name="name", lookup_expr="icontains")

    class Meta:
        model = DashBoard
        fields = ["search"]


class DashboardFilter(django_filters.FilterSet):
    search = django_filters.CharFilter(field_name="name", lookup_expr="icontains")

    class Meta:
        model = DashBoard
        fields = ["search"]


class ProjectMemberFilter(django_filters.FilterSet):
    search = django_filters.CharFilter(field_name="user", method="get_collaborator")

    class Meta:
        model = ProjectMembers
        fields = ["search"]

    def get_collaborator(self, queryset, name, value):
        return queryset.filter(
            Q(user__username__icontains=value)
            | Q(user__email=value)
            | Q(user__first_name__icontains=value)
            | Q(user__last_name__icontains=value)
        )


class UserFilter(django_filters.FilterSet):
    search = django_filters.CharFilter(method="get_user")

    class Meta:
        model = TenantUser
        fields = ["search"]

    def get_user(self, queryset, name, value):
        return queryset.filter(
            Q(username__icontains=value)
            | Q(email__icontains=value)
            | Q(first_name__icontains=value)
            | Q(last_name__icontains=value)
        )
