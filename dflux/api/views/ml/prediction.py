import pandas as pd

from rest_framework import status
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated

from .auto_ml import predict

from dflux.db.models import DataModel
from dflux.api.views.base import BaseAPIView


class Predict(BaseAPIView):
    """
    API endpoint that allows get the the result form the auto_ml.py corresponding function.
    - auto_ml.py stores all the ds related utility functions
    - This will convert Prediction df into Json.

    * Requires JWT authentication.
    * This endpoint will allows only POST method.
    """

    permission_classes = (IsAuthenticated,)

    def post(self, request, pk):

        try:
            model = DataModel.objects.get(id=pk)
            model_weights_pkl = model.pickle_url
            scaler_pkl = model.scaler_url

            meta = model.meta_data
            meta.pre_processing["labelencoding"] = False
            metadata = {
                "name": meta.name,
                "query": meta.query.raw_sql,
                "skipped": meta.skipped,
                "pre_processing": meta.pre_processing,
                "modeling_type": meta.modeling_type,
                "algorithms": meta.algorithms,
            }
            input_array = request.data.get("input_array")
            df = pd.DataFrame(input_array)
            try:
                prediction = predict(metadata, model_weights_pkl, scaler_pkl, df)
            except Exception as e:
                return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
            return Response({"prediction": prediction})
        except DataModel.DoesNotExist:
            return Response(
                {"msg": "please provide valid model id"},
                status=status.HTTP_400_BAD_REQUEST,
            )
