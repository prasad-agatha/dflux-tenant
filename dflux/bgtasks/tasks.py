# import datetime
# import asyncio

# from decimal import Decimal
# from celery import shared_task

# from django.contrib.auth.models import User

# from dflux.bgtasks.celery import app
# from dflux.db.models.trigger import ChartTrigger
# from dflux.db.models import (
#     Connection,
#     Query,
#     Trigger,
#     TriggerOutput,
#     ChartTrigger,
#     ShareCharts,
#     Project,
#     Profile,
# )
# from dflux.api.serializers import TriggerOutputSerializer
# from dflux.api.views.query import connection_establishment, query_invoke
# from dflux.utils.emails import emails


# @shared_task
# def add(x, y):
#     return x + y


# def execute_sql_query(connection_id, sql_query):

#     # checking connection exists or not
#     try:
#         db = Connection.objects.get(id=connection_id)
#     except Exception as e:
#         return str(e)

#     # database connection establishment
#     try:
#         conn = connection_establishment(db)
#         cursor = conn.cursor()
#         loop = asyncio.new_event_loop()
#         result = loop.run_until_complete(query_invoke(cursor, sql_query))
#         loop.close()
#         data = []
#         for record in result:
#             record_ = {}
#             for key, value in record.items():
#                 if isinstance(value, Decimal):
#                     record_[key] = int(str(value))
#                 else:
#                     record_[key] = value
#             data.append(record_)
#         return data
#     except Exception as e:
#         print(str(e))


# def execute_query_send_mail(trigger_id):
#     try:
#         # getting all the triggers
#         trigger = Trigger.objects.get(id=trigger_id)
#         TriggerOutput.objects.filter(trigger=trigger_id).delete()
#         queryset = Query.objects.get(id=trigger.query.pk)
#         # print(trigger.email)
#         # print(trigger.query.connection.id, trigger.query.raw_sql)
#         data = execute_sql_query(trigger.query.connection.id, trigger.query.raw_sql)
#         queryset.extra = None
#         queryset.extra = {}
#         queryset.extra["data"] = data
#         queryset.save()
#         # final_data = json.dumps(data, default=myconverter)
#         output_data = {"trigger": trigger.id, "data": data}
#         serializer = TriggerOutputSerializer(data=output_data)
#         if serializer.is_valid(raise_exception=True):
#             serializer.save()
#             # print(serializer.data)
#         now = datetime.datetime.now()
#         project = Project.objects.get(id=trigger.project_id)
#         emails.send_trigger_success_email(trigger, project)

#         print("mail send successfully")

#     except Exception as e:
#         trigger = Trigger.objects.get(id=trigger_id)
#         emails.send_trigger_error_email(trigger, e)


# def send_chart_link(trigger_id):
#     try:
#         import uuid

#         token = uuid.uuid1().hex
#         # getting all the chart triggers
#         chart_trigger = ChartTrigger.objects.get(id=trigger_id)
#         # creating token for chart
#         ShareCharts.objects.get_or_create(charts=chart_trigger.chart, token=token)
#         project = Project.objects.get(id=chart_trigger.project_id)
#         emails.send_chart_link_email(chart_trigger, token)
#         print("chart trigger mail send successfully")

#     except Exception as e:
#         emails.send_chart_link_error_email(chart_trigger, e)


# @shared_task
# def executetrigger(trigger_id, charttrigger_id):
#     execute_query_send_mail(trigger_id)
#     send_chart_link(charttrigger_id)


# @app.task
# def inactivate_users_after_15_days():
#     from django.utils import timezone
#     from datetime import timedelta

#     users = (
#         User.objects.filter(is_active=True)
#         .exclude(email__endswith="@intellectdata.com")
#         .exclude(email__endswith="@soulpageit.com")
#     )

#     for user in users:
#         try:
#             profile = Profile.objects.filter(user=user).first()
#             if profile is not None:
#                 # if extended date is None take user joined date
#                 if user.date_joined and profile.extended_date is None:
#                     last_date = (user.date_joined + timedelta(days=15)).date()
#                     if timezone.now().date() > last_date:
#                         user.is_active = False
#                         user.save()
#                 else:
#                     last_date = (profile.extended_date + timedelta(days=15)).date()
#                     if timezone.now().date() > last_date:
#                         user.is_active = False
#                         user.save()
#         except Exception as e:
#             print(e)
