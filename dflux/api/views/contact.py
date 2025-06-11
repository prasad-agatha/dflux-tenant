from decouple import config

from rest_framework import status
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated

from django.http import Http404
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger


from dflux.db.models import ContactSale
from dflux.api.views.base import BaseAPIView
from dflux.api.serializers import ContactSaleSerializer
from dflux.utils.emails import emails


class ContactSaleView(BaseAPIView):
    """
    API endpoint that allows view list of all the contact sales or create new contact sale.

    * Requires JWT authentication.
    * This endpoint will allows only GET, POST methods.
    """

    permission_classes = (IsAuthenticated,)

    def get(self, request):
        """
        View list of all the contact sales.
        """
        contact_sales = ContactSale.objects.all()
        page = request.query_params.get("page", 1)
        paginator = Paginator(contact_sales, 20)
        try:
            if int(page) > paginator.num_pages:
                return Response("no objects found")
            contact_sales = paginator.page(page)
        except PageNotAnInteger:
            contact_sales = paginator.page(1)
        except EmptyPage:
            contact_sales = paginator.page(paginator.num_pages)
        serializer = ContactSaleSerializer(contact_sales, many=True)
        page_info = {
            "count": paginator.count,
            "pages": paginator.num_pages,
        }
        return Response({"info": page_info, "results": serializer.data})

    def post(self, request):
        """
        Create new contact sale.
        """
        input_data = dict(request.data)
        email = request.user.email
        support_email = config("SUPPORT_EMAIL")
        serializer = ContactSaleSerializer(data=input_data)
        if serializer.is_valid(raise_exception=True):
            serializer.save()
            # emails.send_contact_email(request, email, support_email)
            # sales submiter
            data = serializer.data
            data["to_emails"] = [email, support_email]
            return Response(data, status=status.HTTP_201_CREATED)


class ContactSaleDetailView(BaseAPIView):
    """
    API endpoint that allows view, update, delete individual contact sale details.

    * Requires JWT authentication.
    * This endpoint will allows only GET, PUT, DELETE methods.
    """

    permission_classes = (IsAuthenticated,)

    def get_object(self, pk):
        """
        Get contact sale object using the pk value.
        """
        try:
            return ContactSale.objects.get(pk=pk)
        except ContactSale.DoesNotExist:
            raise Http404

    def get(self, request, pk):
        """
        Get contact sale details.
        """
        contact_sale = self.get_object(pk)
        serializer = ContactSaleSerializer(contact_sale)
        return Response(serializer.data)

    def put(self, request, pk):
        """
        Update contact sale details.
        """
        contact_sale = self.get_object(pk)
        serializer = ContactSaleSerializer(
            contact_sale, data=request.data, partial=True
        )
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, pk):
        """
        Delete contact sale details.
        """
        contact_sale = self.get_object(pk)
        contact_sale.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)
