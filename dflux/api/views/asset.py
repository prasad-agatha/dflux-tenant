from rest_framework import status
from rest_framework.response import Response


from django.shortcuts import get_object_or_404

from dflux.db.models import MediaAsset
from dflux.api.views.base import BaseAPIView
from dflux.api.serializers import MediaAssetSerializer


class MediaAssetView(BaseAPIView):
    """
    API endpoint that allows upload files into to default file storage(like s3 bucket etc) or view the uploaded files.

    * Authentication not required.
    * This endpoint will allows only GET, POST methods.
    """

    def get(self, request):
        """
        Return a list of all the media files.
        """

        queryset = MediaAsset.objects.all()

        serializer = MediaAssetSerializer(queryset, many=True)

        return Response(serializer.data, status=status.HTTP_200_OK)

    def post(self, request):
        """
        Upload file into default file storage.
        """

        myDict = dict(request.data)
        try:
            if len(myDict["name[]"]) != len(myDict["asset[]"]):
                return Response(
                    {"Error": "length of name[], length of asset[] not equal"}
                )
            else:
                dict_data = [
                    {"name": name, "asset": asset}
                    for name, asset in zip(myDict["name[]"], myDict["asset[]"])
                ]

        except Exception as e:
            return Response(
                {"Error": "Payload error" + str(e)}, status=status.HTTP_400_BAD_REQUEST
            )
        serializer = MediaAssetSerializer(data=dict_data, many=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class MediaAssetDetailView(BaseAPIView):
    """
    API endpoint that allows view individual file details or delete individual file.

    * Authentication not required.
    * This endpoint will allows only GET, DELETE methods.
    """

    def get(self, request, pk):
        """
        View individual file details
        """

        queryset = get_object_or_404(MediaAsset, id=pk)
        serializer = MediaAssetSerializer(queryset)
        return Response(serializer.data, status=status.HTTP_200_OK)

    # delete store by id
    def delete(self, request, pk):
        """
        Delete individual file details
        """

        queryset = get_object_or_404(MediaAsset, id=pk)
        queryset.delete()
        return Response({"message": "Delete Success"}, status=status.HTTP_200_OK)
