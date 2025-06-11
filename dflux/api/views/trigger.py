from rest_framework import status
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated

from django.http import Http404
from django.utils import timezone
from django.shortcuts import get_object_or_404

from dflux.api.serializers import (
    TriggrSerializer,
    ChartTriggrSerializer,
    TriggerOutputSerializer,
)
from dflux.api.views.base import BaseAPIView
from dflux.db.models import Query, Project, Charts, Trigger, ChartTrigger, TriggerOutput

from .filters import ChartTriggerFilter
from .permissions import (
    ProjectOwnerOrCollaborator,
    ProjectChartTriggerAccess,
    ProjectModuleAccess,
)

from django_celery_beat.models import PeriodicTask, CrontabSchedule


# class TriggerView( BaseAPIView):
#     permission_classes = (IsAuthenticated,)

#     def get(self, request, pk):
#         triggers = Trigger.objects.filter(project__id=pk)
#         serializer = TriggrSerializer(triggers, many=True)
#         return Response(serializer.data)

#     def post(self, request, pk):
#         # creating cronschedule object
#         cron = request.data.get("cron_expression")
#         schedule, created = CrontabSchedule.objects.get_or_create(
#             minute=cron.get("minute", None),
#             hour=cron.get("hour", None),
#             day_of_week=cron.get("day_of_week", None),
#             day_of_month=cron.get("day_of_month", None),
#             month_of_year=cron.get("month_of_year", None),
#         )

#         project = get_object_or_404(Project, id=pk)
#         request.data["project"] = project.id
#         serializer = TriggrSerializer(data=request.data)
#         if serializer.is_valid(raise_exception=True):
#             # trigger object saved
#             trigger = serializer.save()
#             try:
#                 periodic_task, created = PeriodicTask.objects.get_or_create(
#                     name=request.data.get("name"),
#                     task="dflux.bgtasks.tasks.execute_query_send_mail",
#                     crontab=schedule,
#                     args=[trigger.id],
#                     start_time=timezone.now(),
#                 )
#             except Exception as e:
#                 return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
#             return Response(serializer.data)


# class TriggerDetailView( BaseAPIView):
#     permission_classes = (IsAuthenticated,)

#     def get(self, request, pk):
#         trigger = get_object_or_404(Trigger, id=pk)
#         serializer = TriggrSerializer(trigger)
#         return Response(serializer.data)

#     def put(self, request, pk):

#         # creating cronschedule object
#         cron = request.data.get("cron_expression")
#         schedule, created = CrontabSchedule.objects.get_or_create(
#             minute=cron.get("minute", None),
#             hour=cron.get("hour", None),
#             day_of_week=cron.get("day_of_week", None),
#             day_of_month=cron.get("day_of_month", None),
#             month_of_year=cron.get("month_of_year", None),
#         )

#         trigger = get_object_or_404(Trigger, id=pk)
#         periodic_task = get_object_or_404(PeriodicTask, name=trigger.name)
#         # updating periodic_task name
#         periodic_task.name = request.data.get("name")
#         periodic_task.crontab = schedule
#         periodic_task.save()

#         serializer = TriggrSerializer(trigger, data=request.data, partial=True)
#         if serializer.is_valid(raise_exception=True):
#             serializer.save()
#             return Response(serializer.data)

#     def delete(self, request, pk):
#         trigger = get_object_or_404(Trigger, id=pk)
#         # delete trigger
#         trigger.delete()

#         periodic_task = PeriodicTask.objects.filter(name=trigger.name)
#         # delete trigger from periodic tasks
#         periodic_task.delete()
#         return Response(status=status.HTTP_204_NO_CONTENT)


class TriggerOutputViews(BaseAPIView):
    """
    API endpoint that allows view list of all the trigger outputs.

    * Requires JWT authentication.
    * This endpoint will allows only GET method.
    """

    permissions = (IsAuthenticated,)

    def get(self, request, pk):
        trigger_data = TriggerOutput.objects.filter(trigger__query=pk)
        serializer = TriggerOutputSerializer(trigger_data, many=True)
        return Response(serializer.data)


class ChartTriggerView(BaseAPIView):
    """
    API endpoint that allows view list of all the chart triggers or create new chart trigger.

    * Requires JWT authentication.
    * Only collaborator or owner of the project can access.
    * This endpoint will allows only GET, POST methods.
    """

    permission_classes = [ProjectModuleAccess]

    def get(self, request, pk):
        """
        This method allows view list of all the chart triggers.
        """
        try:
            get_object_or_404(Project, id=pk)
            chart_triggers = ChartTriggerFilter(
                request.GET, queryset=ChartTrigger.objects.filter(project__id=pk)
            ).qs
            trigger = Trigger.objects.filter(project_id=pk)
            serializer = ChartTriggrSerializer(chart_triggers, many=True)
            list = []
            for i, obj in enumerate(serializer.data):
                a = Charts.objects.get(id=obj["chart"])
                newdict = {"query_id": trigger[i].query.id, "chart_name": a.name}
                newdict.update(obj)
                list.append(newdict)
            return Response(list)
        except Exception as e:
            return Response({"msg": str(e)}, status=status.HTTP_400_BAD_REQUEST)

    def post(self, request, pk):
        """
        This method will allows create new chart trigger.
        """
        # creating cronschedule object
        cron = request.data.get("cron_expression")
        schedule, created = CrontabSchedule.objects.get_or_create(
            minute=cron.get("minute", None),
            hour=cron.get("hour", None),
            day_of_week=cron.get("day_of_week", None),
            day_of_month=cron.get("day_of_month", None),
            month_of_year=cron.get("month_of_year", None),
        )

        project = get_object_or_404(Project, id=pk)
        request.data["project"] = project.id
        get_object_or_404(Query, id=request.data["query"])
        serializer = ChartTriggrSerializer(data=request.data)

        if serializer.is_valid(raise_exception=True):
            # chart trigger object saved
            chart_trigger = serializer.save()
        request.data["charttrigger"] = chart_trigger.id
        trigger_serializer = TriggrSerializer(data=request.data)
        if trigger_serializer.is_valid(raise_exception=True):
            # trigger object is saved
            trigger = trigger_serializer.save()
            try:
                periodic_task, created = PeriodicTask.objects.get_or_create(
                    name=request.data.get("name"),
                    task="dflux.bgtasks.tasks.executetrigger",
                    crontab=schedule,
                    args=[trigger.id, chart_trigger.id],
                    start_time=timezone.now(),
                )

            except Exception as e:
                return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
            return Response(
                {
                    "chart_trigger": serializer.data,
                }
            )


class ChartTriggerDetailView(BaseAPIView):
    """
    API endpoint that allows view, update, delete individual chart trigger details.

    * Requires JWT authentication.
    * Only collaborator or owner of the project can access.
    * This endpoint will allows only GET, PUT, DELETE methods.
    """

    permission_classes = (IsAuthenticated, ProjectChartTriggerAccess)

    def get(self, request, pk):
        """
        This method allows view individual chart trigger details.
        """
        chart_trigger = get_object_or_404(ChartTrigger, id=pk)
        get_object_or_404(Project, id=chart_trigger.project.id, user=request.user.id)
        trigger = get_object_or_404(Trigger, charttrigger=chart_trigger.id)
        serializer = ChartTriggrSerializer(chart_trigger)
        newdict = {"query_id": trigger.query.id}
        newdict.update(serializer.data)
        return Response(newdict)

    def put(self, request, pk):
        """
        This method allows update individual chart trigger details.
        """
        # creating cronschedule object
        cron = request.data.get("cron_expression")
        schedule, created = CrontabSchedule.objects.get_or_create(
            minute=cron.get("minute", None),
            hour=cron.get("hour", None),
            day_of_week=cron.get("day_of_week", None),
            day_of_month=cron.get("day_of_month", None),
            month_of_year=cron.get("month_of_year", None),
        )
        chart_trigger = get_object_or_404(ChartTrigger, id=pk)
        get_object_or_404(Project, id=chart_trigger.project.id)
        periodic_task = get_object_or_404(PeriodicTask, name=chart_trigger.name)
        # updating periodic_task name
        periodic_task.name = request.data.get("name")
        periodic_task.crontab = schedule
        periodic_task.save()

        chartserializer = ChartTriggrSerializer(
            chart_trigger, data=request.data, partial=True
        )
        if chartserializer.is_valid(raise_exception=True):
            # trigger object saved
            chartserializer.save()
        trigger = get_object_or_404(Trigger, charttrigger=pk)
        serializer = TriggrSerializer(trigger, data=request.data, partial=True)
        if serializer.is_valid(raise_exception=True):
            serializer.save()

            return Response(chartserializer.data)

    def delete(self, request, pk):
        """
        This method allows delete individual chart trigger details.
        """
        chart_trigger = get_object_or_404(ChartTrigger, id=pk)
        get_object_or_404(Project, id=chart_trigger.project.id)
        trigger = get_object_or_404(Trigger, charttrigger=chart_trigger.id)
        # delete trigger
        trigger.delete()
        chart_trigger.delete()
        periodic_task = PeriodicTask.objects.filter(name=trigger.name)
        periodic_task.delete()
        periodic_task = PeriodicTask.objects.filter(name=chart_trigger.name)
        # delete trigger from periodic tasks
        periodic_task.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)
