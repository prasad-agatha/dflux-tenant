# import os

# from celery import Celery
# from celery.schedules import crontab

# # set the default Django settings module for the 'celery' program.
# os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dflux.settings")

# CELERY_BROKER_URL = os.environ.get("CELERY_BROKER_URL")
# app = Celery("dflux", broker=CELERY_BROKER_URL)
# # broker="amqp://admin:mypass@rabbitmq:5672//

# # Using a string here means the worker doesn't have to serialize
# # the configuration object to child processes.
# # - namespace='CELERY' means all celery-related configuration keys
# #   should have a `CELERY_` prefix.
# app.config_from_object("django.conf:settings", namespace="CELERY")

# # Load task modules from all registered Django app configs.
# app.autodiscover_tasks()
# app.conf.timezone = "UTC"


# app.conf.beat_schedule = {
#     "run-inactive-users-everyday": {
#         "task": "dflux.bgtasks.tasks.inactivate_users_after_15_days",
#         # run everyday at midnight
#         "schedule": crontab(minute=0, hour=0),
#     },
# }
