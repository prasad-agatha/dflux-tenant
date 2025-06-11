from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated

from .python_runtime import exec_code

from dflux.api.views.base import BaseAPIView


class PythonCodeRunnerView(BaseAPIView):
    """
    API endpoint that allows execute the python code blocks or python files.

    * Requires JWT authentication.
    * This endpoint will allows only POST method.
    """

    permission_classes = (IsAuthenticated,)

    def post(self, request):
        src_files = []
        # handle source code entered in editor
        python_code = request.data.get("python_code")
        if python_code:
            src_files.append({"name": "main.py", "content": python_code.encode()})

        python_files = request.FILES.getlist("python_file[]")
        # iterate over uploaded files
        if python_files is not None:
            for py_file in python_files:
                src_files.append({"name": py_file.name, "content": py_file.read()})
        # result from untrusted code executed in sandbox
        result = exec_code(src_files)
        return Response(result)
