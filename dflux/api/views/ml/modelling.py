import pandas as pd

from rest_framework import status
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated

from dflux.api.serializers import DataModelMetaDataSerializer
from dflux.api.views.utils import (
    classification_method_result,
    multiple_classification_methods,
    regression_methods,
)

from dflux.api.views.base import BaseAPIView
from dflux.api.views.ml.auto_ml import (
    train_and_test_split,
    logistic_regression,
    support_vector_classifier,
    decision_tree,
    random_forest,
    xgboost,
    knn,
    naive_bayes_classifier,
    multinomailNB_classifier,
    adaboost_classifier,
    multi_layer_perceptron_classifier,
    # lgbm_regression,
    # catboost_regression,
    model_evaluation_for_regression,
)
from .config import modelling_methods, cross_validation_techniques


class TrainAndTestSplit(BaseAPIView):
    """
    API endpoint that allows get the the result form the auto_ml.py corresponding function.
    - auto_ml.py stores all the ds related utility functions
    - This will convert TrainAndTestSplit df into Json.

    * Requires JWT authentication.
    * This endpoint will allows only POST method.
    """

    permission_classes = (IsAuthenticated,)

    def post(self, request):
        try:
            data = request.data.get("data")
            target_variable = request.data.get("target_variable")
            # converting input data into df
            df = pd.DataFrame(data)
            X_train, X_test, y_train, y_test = train_and_test_split(df, target_variable)
            return Response(
                {
                    "x_train": X_train,
                    "x_test": X_test,
                    "y_train": y_train,
                    "y_test": y_test,
                }
            )
        except Exception as e:
            return Response({"msg": str(e)}, status=status.HTTP_400_BAD_REQUEST)


class LogisticRegressionModel(BaseAPIView):
    """
    API endpoint that allows get the the result form the auto_ml.py corresponding function.
    - auto_ml.py stores all the ds related utility functions
    - This will convert LogisticRegressionModel df into Json.

    * Requires JWT authentication.
    * This endpoint will allows only POST method.
    """

    permission_classes = (IsAuthenticated,)

    def post(self, request):
        method = {"logistic regression model": logistic_regression}
        try:
            input_method = method.get(request.data.get("fit_model"))
            result = classification_method_result(request, input_method)
            return Response(result)
        except Exception as e:
            return Response({"msg": str(e)}, status=status.HTTP_400_BAD_REQUEST)


class SupportVectorClassifier(BaseAPIView):
    """
    API endpoint that allows get the the result form the auto_ml.py corresponding function.
    - auto_ml.py stores all the ds related utility functions
    - This will convert SupportVectorClassifier df into Json.

    * Requires JWT authentication.
    * This endpoint will allows only POST method.
    """

    permission_classes = (IsAuthenticated,)

    def post(self, request):
        method = {"support vector classifier": support_vector_classifier}
        try:
            input_method = method.get(request.data.get("fit_model"))
            result = classification_method_result(request, input_method)
            return Response(result)
        except Exception as e:
            return Response({"msg": str(e)}, status=status.HTTP_400_BAD_REQUEST)


class DecisionTreeClassifier(BaseAPIView):
    """
    API endpoint that allows get the the result form the auto_ml.py corresponding function.
    - auto_ml.py stores all the ds related utility functions
    - This will convert DecisionTreeClassifier df into Json.

    * Requires JWT authentication.
    * This endpoint will allows only POST method.
    """

    permission_classes = (IsAuthenticated,)

    def post(self, request):
        method = {"decision tree classifier": decision_tree}
        try:
            input_method = method.get(request.data.get("fit_model"))
            result = classification_method_result(request, input_method)
            return Response(result)
        except Exception as e:
            return Response({"msg": str(e)}, status=status.HTTP_400_BAD_REQUEST)


class RandomForestClassifier(BaseAPIView):
    """
    API endpoint that allows get the the result form the auto_ml.py corresponding function.
    - auto_ml.py stores all the ds related utility functions
    - This will convert RandomForestClassifier df into Json.

    * Requires JWT authentication.
    * This endpoint will allows only POST method.
    """

    permission_classes = (IsAuthenticated,)

    def post(self, request):
        method = {"random forest classifier": random_forest}
        try:
            input_method = method.get(request.data.get("fit_model"))
            result = classification_method_result(request, input_method)
            return Response(result)
        except Exception as e:
            return Response({"msg": str(e)}, status=status.HTTP_400_BAD_REQUEST)


class XGBClassifier(BaseAPIView):
    """
    API endpoint that allows get the the result form the auto_ml.py corresponding function.
    - auto_ml.py stores all the ds related utility functions
    - This will convert XGBClassifier df into Json.

    * Requires JWT authentication.
    * This endpoint will allows only POST method.
    """

    permission_classes = (IsAuthenticated,)

    def post(self, request):
        method = {"xgboost classifier": xgboost}
        try:
            input_method = method.get(request.data.get("fit_model"))
            result = classification_method_result(request, input_method)
            return Response(result)
        except Exception as e:
            return Response({"msg": str(e)}, status=status.HTTP_400_BAD_REQUEST)


class KNNClassifier(BaseAPIView):
    """
    API endpoint that allows get the the result form the auto_ml.py corresponding function.
    - auto_ml.py stores all the ds related utility functions
    - This will convert KNNClassifier df into Json.

    * Requires JWT authentication.
    * This endpoint will allows only POST method.
    """

    permission_classes = (IsAuthenticated,)

    def post(self, request):
        method = {"knn classifier": knn}
        try:
            input_method = method.get(request.data.get("fit_model"))
            result = classification_method_result(request, input_method)
            return Response(result)
        except Exception as e:
            return Response({"msg": str(e)}, status=status.HTTP_400_BAD_REQUEST)


class NaiveBayesClassifier(BaseAPIView):
    """
    API endpoint that allows get the the result form the auto_ml.py corresponding function.
    - auto_ml.py stores all the ds related utility functions
    - This will convert NaiveBayesClassifier df into Json.

    * Requires JWT authentication.
    * This endpoint will allows only POST method.
    """

    permission_classes = (IsAuthenticated,)

    def post(self, request):
        method = {"naive bayes classifier": naive_bayes_classifier}
        try:
            input_method = method.get(request.data.get("fit_model"))
            result = classification_method_result(request, input_method)
            return Response(result)
        except Exception as e:
            return Response({"msg": str(e)}, status=status.HTTP_400_BAD_REQUEST)


class MultinomailNBClassifier(BaseAPIView):
    """
    API endpoint that allows get the the result form the auto_ml.py corresponding function.
    - auto_ml.py stores all the ds related utility functions
    - This will convert MultinomailNBClassifier df into Json.

    * Requires JWT authentication.
    * This endpoint will allows only POST method.
    """

    permission_classes = (IsAuthenticated,)

    def post(self, request):
        method = {"multinomailNB classifier": multinomailNB_classifier}
        try:
            input_method = method.get(request.data.get("fit_model"))
            result = classification_method_result(request, input_method)
            return Response(result)
        except Exception as e:
            return Response({"msg": str(e)}, status=status.HTTP_400_BAD_REQUEST)


class AdaBoostClassifier(BaseAPIView):
    """
    API endpoint that allows get the the result form the auto_ml.py corresponding function.
    - auto_ml.py stores all the ds related utility functions
    - This will convert AdaBoostClassifier df into Json.

    * Requires JWT authentication.
    * This endpoint will allows only POST method.
    """

    permission_classes = (IsAuthenticated,)

    def post(self, request):
        method = {"adaboost classifier": adaboost_classifier}
        try:
            input_method = method.get(request.data.get("fit_model"))
            result = classification_method_result(request, input_method)
            return Response(result)
        except Exception as e:
            return Response({"msg": str(e)}, status=status.HTTP_400_BAD_REQUEST)


class MultiLayerPerceptronClassifier(BaseAPIView):
    """
    API endpoint that allows get the the result form the auto_ml.py corresponding function.
    - auto_ml.py stores all the ds related utility functions
    - This will convert MultiLayerPerceptronClassifier df into Json.

    * Requires JWT authentication.
    * This endpoint will allows only POST method.
    """

    permission_classes = (IsAuthenticated,)

    def post(self, request):
        method = {
            "multi layer perceptron classifier": multi_layer_perceptron_classifier
        }
        try:
            input_method = method.get(request.data.get("fit_model"))
            result = classification_method_result(request, input_method)
            return Response(result)
        except Exception as e:
            return Response({"msg": str(e)}, status=status.HTTP_400_BAD_REQUEST)


class ModellingMethods(BaseAPIView):
    """
    API endpoint that allows get the the result form the auto_ml.py corresponding function.
    - auto_ml.py stores all the ds related utility functions
    - This will convert ModellingMethods df into Json.

    * Requires JWT authentication.
    * This endpoint will allows only POST method.
    """

    permission_classes = (IsAuthenticated,)

    def post(self, request):
        # save meta data
        meta_data = request.data.get("meta_data")
        serializer = DataModelMetaDataSerializer(data=meta_data)
        if serializer.is_valid(raise_exception=True):
            meta_data_obj = serializer.save()
        output = {}
        input_modelling_methods = request.data.get("modelling")
        modeling_type = request.data.get("modeling_type")
        target_variable = request.data.get("target_variable")
        custom_selection = request.data.get("custom_selection")
        data = request.data.get("data")
        for input_modelling_method in input_modelling_methods:
            df = pd.DataFrame(data)
            modelling = modelling_methods.get(input_modelling_method)
            if modeling_type == "classification":
                output_file_name = (
                    f"{meta_data_obj.id}_{input_modelling_method}".replace(" ", "_")
                )
                try:
                    response = multiple_classification_methods(
                        request,
                        input_modelling_method,
                        df,
                        target_variable,
                        output_file_name,
                        modelling,
                        custom_selection,
                        cross_validation_techniques,
                    )
                    output[input_modelling_method] = response
                except Exception as e:
                    output[input_modelling_method] = {
                        "input_modelling_method": input_modelling_method,
                        "confusion_matrix_result": [],
                        "classification_report_result_mean": {
                            "precision": 0,
                            "recall": 0,
                            "f1-score": 0,
                            "support": 0,
                        },
                        "classification_report_result_reset_index": [],
                        "accuracy_score": 0,
                        "error_msg": str(e),
                        "model_status": "Failed",
                    }
            elif modeling_type == "regression":
                try:
                    output_file_name = (
                        f"{meta_data_obj.id}_{input_modelling_method}".replace(" ", "_")
                    )
                    response = regression_methods(
                        request,
                        df,
                        target_variable,
                        output_file_name,
                        input_modelling_method,
                        modelling,
                        custom_selection,
                    )
                    output[input_modelling_method] = response
                except Exception as e:
                    return Response({"msg": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        return Response({"output": output, "meta_data": serializer.data})
