import pandas as pd

from rest_framework import status
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated

from dflux.api.views.base import BaseAPIView

from dflux.api.views.ml.auto_ml import (
    linear_regression,
    support_vector_machine_regressor,
    decision_tree_regressor,
    random_forest_regressor,
    xgb_regressor,
    kneighbors_regressor,
    polynomial_regression,
    lasso_regressor,
    ridge_regressor,
    elasticnet_regression,
    sgd_regression,
    gradientboosting_regression,
    # lgbm_regression,
    # catboost_regression,
)


class LinearRegression(BaseAPIView):
    permission_classes = (IsAuthenticated,)

    def post(self, request):
        try:
            data = request.data.get("data")
            target_variable = request.data.get("target_variable")
            output_file_name = request.data.get("output_file_name")
            # converting input data into df
            df = pd.DataFrame(data)
            rmse_score = linear_regression(df, target_variable, output_file_name)
            X_test = linear_regression(df, target_variable, output_file_name)
            return Response({"rmse_score": rmse_score, "X_test": X_test})

        except Exception as e:
            return Response({"msg": str(e)}, status=status.HTTP_400_BAD_REQUEST)


class SupportVectorRegressor(BaseAPIView):
    """
    API endpoint that allows get the the result form the auto_ml.py corresponding function.
    - auto_ml.py stores all the ds related utility functions
    - This will convert SupportVectorRegressor df into Json.

    * Requires JWT authentication.
    * This endpoint will allows only POST method.
    """

    permission_classes = (IsAuthenticated,)

    def post(self, request):
        try:
            data = request.data.get("data")
            target_variable = request.data.get("target_variable")
            output_file_name = request.data.get("output_file_name")
            # converting input data into df
            df = pd.DataFrame(data)
            rmse_score = support_vector_machine_regressor(
                df, target_variable, output_file_name
            )
            X_test = support_vector_machine_regressor(
                df, target_variable, output_file_name
            )
            return Response({"rmse_score": rmse_score, "X_test": X_test})
        except Exception as e:
            return Response({"msg": str(e)}, status=status.HTTP_400_BAD_REQUEST)


class DecisionTreeRegressor(BaseAPIView):
    """
    API endpoint that allows get the the result form the auto_ml.py corresponding function.
    - auto_ml.py stores all the ds related utility functions
    - This will convert DecisionTreeRegressor df into Json.

    * Requires JWT authentication.
    * This endpoint will allows only POST method.
    """

    permission_classes = (IsAuthenticated,)

    def post(self, request):
        try:
            data = request.data.get("data")
            target_variable = request.data.get("target_variable")
            output_file_name = request.data.get("output_file_name")
            # converting input data into df
            df = pd.DataFrame(data)
            rmse_score = decision_tree_regressor(df, target_variable, output_file_name)
            X_test = decision_tree_regressor(df, target_variable, output_file_name)
            return Response({"rmse_score": rmse_score, "X_test": X_test})
        except Exception as e:
            return Response({"msg": str(e)}, status=status.HTTP_400_BAD_REQUEST)


class RandomForestRegressor(BaseAPIView):
    """
    API endpoint that allows get the the result form the auto_ml.py corresponding function.
    - auto_ml.py stores all the ds related utility functions
    - This will convert RandomForestRegressor df into Json.

    * Requires JWT authentication.
    * This endpoint will allows only POST method.
    """

    permission_classes = (IsAuthenticated,)

    def post(self, request):
        try:
            data = request.data.get("data")
            target_variable = request.data.get("target_variable")
            output_file_name = request.data.get("output_file_name")
            # converting input data into df
            df = pd.DataFrame(data)
            rmse_score = random_forest_regressor(df, target_variable, output_file_name)
            X_test = random_forest_regressor(df, target_variable, output_file_name)
            return Response({"rmse_score": rmse_score, "X_test": X_test})
        except Exception as e:
            return Response({"msg": str(e)}, status=status.HTTP_400_BAD_REQUEST)


class XGBRegressor(BaseAPIView):
    """
    API endpoint that allows get the the result form the auto_ml.py corresponding function.
    - auto_ml.py stores all the ds related utility functions
    - This will convert XGBRegressor df into Json.

    * Requires JWT authentication.
    * This endpoint will allows only POST method.
    """

    permission_classes = (IsAuthenticated,)

    def post(self, request):
        try:
            data = request.data.get("data")
            target_variable = request.data.get("target_variable")
            output_file_name = request.data.get("output_file_name")
            # converting input data into df
            df = pd.DataFrame(data)
            rmse_score = xgb_regressor(df, target_variable, output_file_name)
            X_test = xgb_regressor(df, target_variable, output_file_name)
            return Response({"rmse_score": rmse_score, "X_test": X_test})
        except Exception as e:
            return Response({"msg": str(e)}, status=status.HTTP_400_BAD_REQUEST)


class KNeighborsRegressor(BaseAPIView):
    """
    API endpoint that allows get the the result form the auto_ml.py corresponding function.
    - auto_ml.py stores all the ds related utility functions
    - This will convert KNeighborsRegressor df into Json.

    * Requires JWT authentication.
    * This endpoint will allows only POST method.
    """

    permission_classes = (IsAuthenticated,)

    def post(self, request):
        try:
            data = request.data.get("data")
            target_variable = request.data.get("target_variable")
            output_file_name = request.data.get("output_file_name")
            # converting input data into df
            df = pd.DataFrame(data)
            rmse_score = kneighbors_regressor(df, target_variable, output_file_name)
            X_test = kneighbors_regressor(df, target_variable, output_file_name)
            return Response({"rmse_score": rmse_score, "X_test": X_test})
        except Exception as e:
            return Response({"msg": str(e)}, status=status.HTTP_400_BAD_REQUEST)


class PolynomialRegression(BaseAPIView):
    """
    API endpoint that allows get the the result form the auto_ml.py corresponding function.
    - auto_ml.py stores all the ds related utility functions
    - This will convert PolynomialRegression df into Json.

    * Requires JWT authentication.
    * This endpoint will allows only POST method.
    """

    permission_classes = (IsAuthenticated,)

    def post(self, request):
        try:
            data = request.data.get("data")
            target_variable = request.data.get("target_variable")
            output_file_name = request.data.get("output_file_name")
            # converting input data into df
            df = pd.DataFrame(data)
            rmse_score = polynomial_regression(df, target_variable, output_file_name)
            X_test = polynomial_regression(df, target_variable, output_file_name)
            return Response({"rmse_score": rmse_score, "X_test": X_test})
        except Exception as e:
            return Response({"msg": str(e)}, status=status.HTTP_400_BAD_REQUEST)


class LassoRegressor(BaseAPIView):
    """
    API endpoint that allows get the the result form the auto_ml.py corresponding function.
    - auto_ml.py stores all the ds related utility functions
    - This will convert LassoRegressor df into Json.

    * Requires JWT authentication.
    * This endpoint will allows only POST method.
    """

    permission_classes = (IsAuthenticated,)

    def post(self, request):
        try:
            data = request.data.get("data")
            target_variable = request.data.get("target_variable")
            output_file_name = request.data.get("output_file_name")
            # converting input data into df
            df = pd.DataFrame(data)
            rmse_score = lasso_regressor(df, target_variable, output_file_name)
            X_test = lasso_regressor(df, target_variable, output_file_name)
            return Response({"rmse_score": rmse_score, "X_test": X_test})
        except Exception as e:
            return Response({"msg": str(e)}, status=status.HTTP_400_BAD_REQUEST)


class RidgeRegressor(BaseAPIView):
    """
    API endpoint that allows get the the result form the auto_ml.py corresponding function.
    - auto_ml.py stores all the ds related utility functions
    - This will convert RidgeRegressor df into Json.

    * Requires JWT authentication.
    * This endpoint will allows only POST method.
    """

    permission_classes = (IsAuthenticated,)

    def post(self, request):
        try:
            data = request.data.get("data")
            target_variable = request.data.get("target_variable")
            output_file_name = request.data.get("output_file_name")
            # converting input data into df
            df = pd.DataFrame(data)
            rmse_score = ridge_regressor(df, target_variable, output_file_name)
            X_test = ridge_regressor(df, target_variable, output_file_name)
            return Response({"rmse_score": rmse_score, "X_test": X_test})
        except Exception as e:
            return Response({"msg": str(e)}, status=status.HTTP_400_BAD_REQUEST)


class ElasticnetRegression(BaseAPIView):
    """
    API endpoint that allows get the the result form the auto_ml.py corresponding function.
    - auto_ml.py stores all the ds related utility functions
    - This will convert ElasticnetRegression df into Json.

    * Requires JWT authentication.
    * This endpoint will allows only POST method.
    """

    permission_classes = (IsAuthenticated,)

    def post(self, request):
        try:
            data = request.data.get("data")
            target_variable = request.data.get("target_variable")
            output_file_name = request.data.get("output_file_name")
            # converting input data into df
            df = pd.DataFrame(data)
            rmse_score = elasticnet_regression(df, target_variable, output_file_name)
            X_test = elasticnet_regression(df, target_variable, output_file_name)
            return Response({"rmse_score": rmse_score, "X_test": X_test})
        except Exception as e:
            return Response({"msg": str(e)}, status=status.HTTP_400_BAD_REQUEST)


class SGDRegression(BaseAPIView):
    """
    API endpoint that allows get the the result form the auto_ml.py corresponding function.
    - auto_ml.py stores all the ds related utility functions
    - This will convert SGDRegression df into Json.

    * Requires JWT authentication.
    * This endpoint will allows only POST method.
    """

    permission_classes = (IsAuthenticated,)

    def post(self, request):
        try:
            data = request.data.get("data")
            target_variable = request.data.get("target_variable")
            output_file_name = request.data.get("output_file_name")
            # converting input data into df
            df = pd.DataFrame(data)
            rmse_score = sgd_regression(df, target_variable, output_file_name)
            X_test = sgd_regression(df, target_variable, output_file_name)
            return Response({"rmse_score": rmse_score, "X_test": X_test})
        except Exception as e:
            return Response({"msg": str(e)}, status=status.HTTP_400_BAD_REQUEST)


class GradientboostingRegression(BaseAPIView):
    """
    API endpoint that allows get the the result form the auto_ml.py corresponding function.
    - auto_ml.py stores all the ds related utility functions
    - This will convert GradientboostingRegression df into Json.

    * Requires JWT authentication.
    * This endpoint will allows only POST method.
    """

    permission_classes = (IsAuthenticated,)

    def post(self, request):
        try:
            data = request.data.get("data")
            target_variable = request.data.get("target_variable")
            output_file_name = request.data.get("output_file_name")
            # converting input data into df
            df = pd.DataFrame(data)
            rmse_score = gradientboosting_regression(
                df, target_variable, output_file_name
            )
            X_test = gradientboosting_regression(df, target_variable, output_file_name)
            return Response({"rmse_score": rmse_score, "X_test": X_test})

        except Exception as e:
            return Response({"msg": str(e)}, status=status.HTTP_400_BAD_REQUEST)


# class LGBMRegression( BaseAPIView):
#     permission_classes = (IsAuthenticated,)

#     def post(self, request):
#         try:
#             data = request.data.get("data")
#             target_variable = request.data.get("target_variable")
#             output_file_name = request.data.get("output_file_name")
#             # converting input data into df
#             df = pd.DataFrame(data)
#             print(df, "LGBMRegression")
#             rmse_score = lgbm_regression(df, target_variable, output_file_name)
#             X_test = lgbm_regression(df, target_variable, output_file_name)
#             return Response({"rmse_score": rmse_score, "X_test": X_test})

#         except Exception as e:
#             return Response({"msg": str(e)}, status=status.HTTP_400_BAD_REQUEST)


# class CatBoostRegression( BaseAPIView):
#     permission_classes = (IsAuthenticated,)

#     def post(self, request):
#         try:
#             data = request.data.get("data")
#             target_variable = request.data.get("target_variable")
#             output_file_name = request.data.get("output_file_name")
#             # converting input data into df
#             df = pd.DataFrame(data)
#             print(df, "CatBoostRegression")
#             rmse_score = catboost_regression(df, target_variable, output_file_name)
#             X_test = catboost_regression(df, target_variable, output_file_name)
#             return Response({"rmse_score": rmse_score, "X_test": X_test})

#         except Exception as e:
#             return Response({"msg": str(e)}, status=status.HTTP_400_BAD_REQUEST)


class RegressionMethods(BaseAPIView):
    """
    API endpoint that allows get the the result form the auto_ml.py corresponding function.
    - auto_ml.py stores all the ds related utility functions
    - This will convert RegressionMethods df into Json.

    * Requires JWT authentication.
    * This endpoint will allows only POST method.
    """

    permission_classes = (IsAuthenticated,)

    def post(self, request):
        regression_methods = {
            "linear_regression": linear_regression,
            "support_vector_machine_regressor": support_vector_machine_regressor,
            "decision_tree_regressor": decision_tree_regressor,
            "random_forest_regressor": random_forest_regressor,
            "xgb_regressor": xgb_regressor,
            "kneighbors_regressor": kneighbors_regressor,
            "polynomial_regression": polynomial_regression,
            "lasso_regressor": lasso_regressor,
            "ridge_regressor": ridge_regressor,
            "elasticnet_regression": elasticnet_regression,
            "sgd_regression": sgd_regression,
            "gradientboosting_regression": gradientboosting_regression,
            # "lgbm_regression": lgbm_regression,
            # "catboost_regression": catboost_regression,
        }
        output = {}
        try:
            input_regression_methods = request.data.get("regression")
            for input_regression_method in input_regression_methods:
                regression = regression_methods.get(input_regression_method)
                data = request.data.get("data")
                df = pd.DataFrame(data)
                target_variable = request.data.get("target_variable")
                output_file_name = request.data.get("output_file_name")
                output[input_regression_method] = regression(
                    df, target_variable, output_file_name
                )
            return Response(output)
        except Exception as e:
            return Response({"msg": str(e)}, status=status.HTTP_400_BAD_REQUEST)
