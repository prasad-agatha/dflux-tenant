import pandas as pd

from rest_framework import status
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated

from dflux.db.models import DataModel
from dflux.api.serializers import DataModelMetaDataSerializer
from dflux.api.views.base import BaseAPIView
from dflux.api.views.ml.auto_ml import auto_arima_model, time_series_predict


class TimeSeriesPreprocessingEndpoint(BaseAPIView):
    """
    API endpoint that allows get the the result form the auto_ml.py corresponding function.
    - auto_ml.py stores all the ds related utility functions


    * Response Fields :
    ----------------------
    "forecast": forecast,
    "rmse_score": rmse,
    "pickle_url": pickle_url,
    "meta_data": serializer.data,

    * Requires JWT authentication.
    * This endpoint will allows only POST method.
    """

    permission_classes = (IsAuthenticated,)

    def post(self, request):
        try:
            data = request.data.get("data")
            seasonality = request.data.get("seasonality")
            date_column = request.data.get("date_column")
            target_column = request.data.get("target_column")
            meta_data = request.data.get("meta_data")
            # save user meta info
            meta_data["pre_processing"] = {
                "seasonality": seasonality,
                "date_column": date_column,
                "target_column": target_column,
            }
            meta_data["target_variable"] = target_column
            serializer = DataModelMetaDataSerializer(data=meta_data)
            if serializer.is_valid(raise_exception=True):
                serializer.save()

            # converting input data into df
            df = pd.DataFrame(data)
            forecast, rmse, pickle_url = auto_arima_model(
                df, date_column, target_column, seasonality
            )
            return Response(
                {
                    "forecast": forecast,
                    "rmse_score": rmse,
                    "pickle_url": pickle_url,
                    "meta_data": serializer.data,
                }
            )

        except Exception as e:
            return Response({"msg": str(e)}, status=status.HTTP_400_BAD_REQUEST)


class TimeSeriesMetaDataEndpoint(BaseAPIView):
    """
    API endpoint that allows view meta data for the time series object.

    * Authentication not required.
    * This endpoint will allows only POST method.
    """

    # permission_classes = (IsAuthenticated,)

    def post(self, request):
        serializer = DataModelMetaDataSerializer(data=request.data)
        if serializer.is_valid(raise_exception=True):
            serializer.save()
            return Response(serializer.data)


class TimeSeriesPredictionEndpoint(BaseAPIView):
    """
    API endpoint that allows predict the data.

    * Authentication not required.
    * This endpoint will allows only POST method.
    """

    # permission_classes = (IsAuthenticated,)

    def post(self, request, pk):
        try:
            model = DataModel.objects.get(id=pk)
            pickle_url = model.pickle_url
            meta = model.meta_data
            date_column = meta.pre_processing.get("date_column", None)
            data = request.data.get("data")
            # converting input data into df
            df = pd.DataFrame(data)
            data = df[date_column]
            forecast = time_series_predict(data, pickle_url)
            return Response(forecast)
        except Exception as e:
            return Response({"msg": str(e)}, status=status.HTTP_400_BAD_REQUEST)
