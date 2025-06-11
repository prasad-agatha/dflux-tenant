from django.urls import path

from dflux.api.views.python_code_execution import PythonCodeRunnerView

urlpatterns = [
    # python code runner
    path("python/", PythonCodeRunnerView.as_view(), name="execute-python-code"),
]
