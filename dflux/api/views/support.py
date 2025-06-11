from decouple import config

from rest_framework import status
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated

from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger

from dflux.db.models import Support
from dflux.api.views.base import BaseAPIView
from dflux.api.serializers import SupportSerializer
from dflux.utils.emails import emails


class SupportView(BaseAPIView):
    """
    API endpoint that allows view list of all the support objects or create new support object.
    - This will send mail support team
    - this will send support confirmation emails to users

    * Requires JWT authentication.
    * This endpoint will allows only GET, POST methods.
    """

    permission_classes = (IsAuthenticated,)

    def get(self, request):
        """
        This method will allows view list of all the support objects.
        """
        support = Support.objects.all()
        page = request.query_params.get("page", 1)
        paginator = Paginator(support, 20)
        try:
            if int(page) > paginator.num_pages:
                return Response("no objects found")
            support = paginator.page(page)
        except PageNotAnInteger:
            support = paginator.page(1)
        except EmptyPage:
            support = paginator.page(paginator.num_pages)
        serializer = SupportSerializer(support, many=True)
        page_info = {
            "count": paginator.count,
            "pages": paginator.num_pages,
        }
        return Response({"info": page_info, "results": serializer.data})

    def post(self, request):
        """
        - This method will allows create new support object into the database.
        - This send users mails to support team.
        """
        input_data = dict(request.data)
        serializer = SupportSerializer(data=input_data)
        email = request.user.email
        support_email = config("SUPPORT_EMAIL")
        if serializer.is_valid(raise_exception=True):
            serializer.save()
            # emails.send_support_email(request, email, support_email)
            data = serializer.data
            data["to_emails"] = [email, support_email]
            return Response(
                data,
                status=status.HTTP_201_CREATED,
            )
