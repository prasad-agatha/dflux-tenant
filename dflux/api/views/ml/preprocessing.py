import pandas as pd

from rest_framework import status
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated

from dflux.api.views.base import BaseAPIView

from .auto_ml import (
    data_type_correction,
    auto_remove_unwanted_columns,
    auto_imputer,
    remove_correlated_columns,
    label_encoding,
    one_hot_encoding,
    ordinal_encoder,
    categorical_value_encoder_transformer,
    standard_scale,
    robust_scale,
    min_max_scale,
    square_root_transformation,
    cube_root_transformation,
    log_transformation,
    square_transformation,
    numeric_transformations,
    adasyn,
    smote,
    resampling_fit_transform,
    detecting_outliers_by_Zscore,
    detecting_outliers_by_IOR,
    detecting_outliers_by_local_outlier_factor,
    detecting_outliers_by_isolation_forest,
    detecting_outliers_by_elliptic_envelope,
    drop_duplicate_columns_rows_and_unique_value_columns,
    get_optimial_number_of_features_count,
    feature_generation_by_pca,
)


class DataTypeCorrection(BaseAPIView):
    """
    API endpoint that allows get the the result form the auto_ml.py corresponding function.
    - auto_ml.py stores all the ds related utility functions
    - This will convert DataTypeCorrection df into Json.

    * Requires JWT authentication.
    * This endpoint will allows only POST method.
    """

    permission_classes = (IsAuthenticated,)

    def post(self, request):
        try:
            data = request.data.get("data")
            # converting input data into df
            df = pd.DataFrame(data)
            # df after dataTypeCorrection
            output_df = data_type_correction(df)
            return Response(output_df)
        except Exception as e:
            return Response({"msg": str(e)}, status=status.HTTP_400_BAD_REQUEST)


class AutoRemoveUnwantedcolumns(BaseAPIView):
    """
    API endpoint that allows get the the result form the auto_ml.py corresponding function.
    - auto_ml.py stores all the ds related utility functions
    - This will convert AutoRemoveUnwantedcolumns df into Json.

    * Requires JWT authentication.
    * This endpoint will allows only POST method.
    """

    permission_classes = (IsAuthenticated,)

    def post(self, request):
        try:
            data = request.data.get("data")
            # converting input data into df
            df = pd.DataFrame(data)
            # df after AutoRemoveUnwantedcolumns
            output_df = auto_remove_unwanted_columns(df)
            return Response(output_df)
        except Exception as e:
            return Response({"msg": str(e)}, status=status.HTTP_400_BAD_REQUEST)


class AutoImputer(BaseAPIView):
    """
    API endpoint that allows get the the result form the auto_ml.py corresponding function.
    - auto_ml.py stores all the ds related utility functions
    - This will convert AutoImputer df into Json.

    * Requires JWT authentication.
    * This endpoint will allows only POST method.
    """

    permission_classes = (IsAuthenticated,)

    def post(self, request):
        try:
            data = request.data.get("data")
            # converting input data into df
            df = pd.DataFrame(data)
            # df after AutoImputer
            output_df = auto_imputer(df)
            return Response(output_df)
        except Exception as e:
            return Response({"msg": str(e)}, status=status.HTTP_400_BAD_REQUEST)


class RemoveCorrelatedColumns(BaseAPIView):
    """
    API endpoint that allows get the the result form the auto_ml.py corresponding function.
    - auto_ml.py stores all the ds related utility functions
    - This will convert RemoveCorrelatedColumns df into Json.

    * Requires JWT authentication.
    * This endpoint will allows only POST method.
    """

    permission_classes = (IsAuthenticated,)

    def post(self, request):
        try:
            data = request.data.get("data")
            threshold = request.data.get("threshold")
            # converting input data into df
            df = pd.DataFrame(data)
            # df after RemoveCorrelatedColumns
            output_df = remove_correlated_columns(df, threshold)
            return Response(output_df)
        except Exception as e:
            return Response({"msg": str(e)}, status=status.HTTP_400_BAD_REQUEST)


class LabelEncoding(BaseAPIView):
    """
    API endpoint that allows get the the result form the auto_ml.py corresponding function.
    - auto_ml.py stores all the ds related utility functions
    - This will convert LabelEncoding df into Json.

    * Requires JWT authentication.
    * This endpoint will allows only POST method.
    """

    permission_classes = (IsAuthenticated,)

    def post(self, request):
        method = {
            "labelencoding": label_encoding,
        }
        try:
            data = request.data.get("data")
            target_variable = request.data.get("target_variable")
            # converting input data into df
            df = pd.DataFrame(data)
            # df after TargetLabelEncoder
            output_df = label_encoding(df, target_variable)
            return Response(output_df)
        except Exception as e:
            return Response({"msg": str(e)}, status=status.HTTP_400_BAD_REQUEST)


class OneHotEncoding(BaseAPIView):
    """
    API endpoint that allows get the the result form the auto_ml.py corresponding function.
    - auto_ml.py stores all the ds related utility functions
    - This will convert OneHotEncoding df into Json.

    * Requires JWT authentication.
    * This endpoint will allows only POST method.
    """

    permission_classes = (IsAuthenticated,)

    def post(self, request):
        method = {
            "onehotencoding": one_hot_encoding,
        }
        try:
            data = request.data.get("data")
            target_variable = request.data.get("target_variable")
            encoder_transform = method.get(request.data.get("encoder_transform"))
            # converting input data into df
            df = pd.DataFrame(data)
            # df after TargetLabelEncoder
            output_df = categorical_value_encoder_transformer(
                df, encoder_transform, target_variable
            )
            return Response(output_df)

        except Exception as e:
            return Response({"msg": str(e)}, status=status.HTTP_400_BAD_REQUEST)


class OrdinalEncoder(BaseAPIView):
    """
    API endpoint that allows get the the result form the auto_ml.py corresponding function.
    - auto_ml.py stores all the ds related utility functions
    - This will convert OrdinalEncoder df into Json.

    * Requires JWT authentication.
    * This endpoint will allows only POST method.
    """

    permission_classes = (IsAuthenticated,)

    def post(self, request):
        method = {
            "ordinalencoder": ordinal_encoder,
        }
        try:
            data = request.data.get("data")
            target_variable = request.data.get("target_variable")
            encoder_transform = method.get(request.data.get("encoder_transform"))
            # converting input data into df
            df = pd.DataFrame(data)

            # df after TargetLabelEncoder
            output_df = categorical_value_encoder_transformer(
                df, encoder_transform, target_variable
            )
            return Response(output_df)

        except Exception as e:
            return Response({"msg": str(e)}, status=status.HTTP_400_BAD_REQUEST)


class StandardScale(BaseAPIView):
    """
    API endpoint that allows get the the result form the auto_ml.py corresponding function.
    - auto_ml.py stores all the ds related utility functions
    - This will convert StandardScale df into Json.

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
            # df after StandardScale
            output_df = standard_scale(df, target_variable)
            return Response(output_df)
        except Exception as e:
            return Response({"msg": str(e)}, status=status.HTTP_400_BAD_REQUEST)


class MinMaxScale(BaseAPIView):
    """
    API endpoint that allows get the the result form the auto_ml.py corresponding function.
    - auto_ml.py stores all the ds related utility functions
    - This will convert MinMaxScale df into Json.

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
            # df after MinMaxScale
            output_df = min_max_scale(df, target_variable)
            return Response(output_df)
        except Exception as e:
            return Response({"msg": str(e)}, status=status.HTTP_400_BAD_REQUEST)


class Adasyn(BaseAPIView):
    """
    API endpoint that allows get the the result form the auto_ml.py corresponding function.
    - auto_ml.py stores all the ds related utility functions
    - This will convert Adasyn df into Json.

    * Requires JWT authentication.
    * This endpoint will allows only POST method.
    """

    permission_classes = (IsAuthenticated,)

    def post(self, request):
        method = {"adasyn": adasyn}
        try:
            data = request.data.get("data")
            target_variable = request.data.get("target_variable")
            resampling_method = method.get(request.data.get("resampling_method"))
            # converting input data into df
            df = pd.DataFrame(data)
            output_df = resampling_fit_transform(df, target_variable, resampling_method)
            return Response(output_df)
        except Exception as e:
            return Response({"msg": str(e)}, status=status.HTTP_400_BAD_REQUEST)


class Smote(BaseAPIView):
    """
    API endpoint that allows get the the result form the auto_ml.py corresponding function.
    - auto_ml.py stores all the ds related utility functions
    - This will convert Smote df into Json.

    * Requires JWT authentication.
    * This endpoint will allows only POST method.
    """

    permission_classes = (IsAuthenticated,)

    def post(self, request):
        method = {"smote": smote}
        try:
            data = request.data.get("data")
            target_variable = request.data.get("target_variable")
            resampling_method = method.get(request.data.get("resampling_method"))
            # converting input data into df
            df = pd.DataFrame(data)
            output_df = resampling_fit_transform(df, target_variable, resampling_method)
            # print(output, "Smote")
            return Response(output_df)

        except Exception as e:
            return Response({"msg": str(e)}, status=status.HTTP_400_BAD_REQUEST)


class DetectingOutliersByZscore(BaseAPIView):
    """
    API endpoint that allows get the the result form the auto_ml.py corresponding function.
    - auto_ml.py stores all the ds related utility functions
    - This will convert DetectingOutliersByZscore df into Json.

    * Requires JWT authentication.
    * This endpoint will allows only POST method.
    """

    permission_classes = (IsAuthenticated,)

    def post(self, request):
        try:
            data = request.data.get("data")
            # converting input data into df
            df = pd.DataFrame(data)
            # df after StandardScale
            output_df = detecting_outliers_by_Zscore(df)
            print(output_df)
            return Response(output_df)

        except Exception as e:
            return Response({"msg": str(e)}, status=status.HTTP_400_BAD_REQUEST)


class DetectingOutliersByIOR(BaseAPIView):
    """
    API endpoint that allows get the the result form the auto_ml.py corresponding function.
    - auto_ml.py stores all the ds related utility functions
    - This will convert DetectingOutliersByIOR df into Json.

    * Requires JWT authentication.
    * This endpoint will allows only POST method.
    """

    permission_classes = (IsAuthenticated,)

    def post(self, request):
        try:
            data = request.data.get("data")
            # converting input data into df
            df = pd.DataFrame(data)
            # df after StandardScale
            output_df = detecting_outliers_by_IOR(df)
            print(output_df)
            return Response(output_df)

        except Exception as e:
            return Response({"msg": str(e)}, status=status.HTTP_400_BAD_REQUEST)


class DetectingOutliersByLocalOutlinerFactor(BaseAPIView):
    """
    API endpoint that allows get the the result form the auto_ml.py corresponding function.
    - auto_ml.py stores all the ds related utility functions
    - This will convert DetectingOutliersByLocalOutlinerFactor df into Json.

    * Requires JWT authentication.
    * This endpoint will allows only POST method.
    """

    permission_classes = (IsAuthenticated,)

    def post(self, request):
        try:
            data = request.data.get("data")
            # converting input data into df
            df = pd.DataFrame(data)
            # df after StandardScale
            output_df = detecting_outliers_by_local_outlier_factor(df)
            print(output_df)
            return Response(output_df)

        except Exception as e:
            return Response({"msg": str(e)}, status=status.HTTP_400_BAD_REQUEST)


class DetectingOutliersByIsolationForest(BaseAPIView):
    """
    API endpoint that allows get the the result form the auto_ml.py corresponding function.
    - auto_ml.py stores all the ds related utility functions
    - This will convert DetectingOutliersByIsolationForest df into Json.

    * Requires JWT authentication.
    * This endpoint will allows only POST method.
    """

    permission_classes = (IsAuthenticated,)

    def post(self, request):
        try:
            data = request.data.get("data")
            # converting input data into df
            df = pd.DataFrame(data)
            # df after StandardScale
            output_df = detecting_outliers_by_isolation_forest(df)
            print(output_df)
            return Response(output_df)

        except Exception as e:
            return Response({"msg": str(e)}, status=status.HTTP_400_BAD_REQUEST)


class DetectingOutliersByEllipiticEnvelope(BaseAPIView):
    """
    API endpoint that allows get the the result form the auto_ml.py corresponding function.
    - auto_ml.py stores all the ds related utility functions
    - This will convert DetectingOutliersByEllipiticEnvelope df into Json.

    * Requires JWT authentication.
    * This endpoint will allows only POST method.
    """

    permission_classes = (IsAuthenticated,)

    def post(self, request):
        try:
            data = request.data.get("data")
            # converting input data into df
            df = pd.DataFrame(data)
            # df after StandardScale
            output_df = detecting_outliers_by_elliptic_envelope(df)
            print(output_df)
            return Response(output_df)

        except Exception as e:
            return Response({"msg": str(e)}, status=status.HTTP_400_BAD_REQUEST)


class StandardScaler(BaseAPIView):
    """
    API endpoint that allows get the the result form the auto_ml.py corresponding function.
    - auto_ml.py stores all the ds related utility functions
    - This will convert StandardScaler df into Json.

    * Requires JWT authentication.
    * This endpoint will allows only POST method.
    """

    permission_classes = (IsAuthenticated,)

    def post(self, request):
        method = {"standardscaler": standard_scale}
        try:
            data = request.data.get("data")
            transformation_function = method.get(
                request.data.get("transformation_function")
            )

            # converting input data into df
            df = pd.DataFrame(data)
            # df after dataTypeCorrection
            output_df = numeric_transformations(df, transformation_function)
            return Response(output_df)
        except Exception as e:
            return Response({"msg": str(e)}, status=status.HTTP_400_BAD_REQUEST)


class RobustScaler(BaseAPIView):
    """
    API endpoint that allows get the the result form the auto_ml.py corresponding function.
    - auto_ml.py stores all the ds related utility functions
    - This will convert RobustScaler df into Json.

    * Requires JWT authentication.
    * This endpoint will allows only POST method.
    """

    permission_classes = (IsAuthenticated,)

    def post(self, request):
        method = {"robustscaler": robust_scale}
        try:
            data = request.data.get("data")
            transformation_function = method.get(
                request.data.get("transformation_function")
            )

            # converting input data into df
            df = pd.DataFrame(data)
            # df after dataTypeCorrection
            output_df = numeric_transformations(df, transformation_function)
            return Response(output_df)
        except Exception as e:
            return Response({"msg": str(e)}, status=status.HTTP_400_BAD_REQUEST)


class MinMaxScaling(BaseAPIView):
    """
    API endpoint that allows get the the result form the auto_ml.py corresponding function.
    - auto_ml.py stores all the ds related utility functions
    - This will convert MinMaxScaling df into Json.

    * Requires JWT authentication.
    * This endpoint will allows only POST method.
    """

    permission_classes = (IsAuthenticated,)

    def post(self, request):
        method = {"minmaxscaling": min_max_scale}
        try:
            data = request.data.get("data")
            transformation_function = method.get(
                request.data.get("transformation_function")
            )

            # converting input data into df
            df = pd.DataFrame(data)
            # df after dataTypeCorrection
            output_df = numeric_transformations(df, transformation_function)
            return Response(output_df)
        except Exception as e:
            return Response({"msg": str(e)}, status=status.HTTP_400_BAD_REQUEST)


class SquareRootTransformation(BaseAPIView):
    """
    API endpoint that allows get the the result form the auto_ml.py corresponding function.
    - auto_ml.py stores all the ds related utility functions
    - This will convert SquareRootTransformation df into Json.

    * Requires JWT authentication.
    * This endpoint will allows only POST method.
    """

    permission_classes = (IsAuthenticated,)

    def post(self, request):
        method = {"squareroottransformation": square_root_transformation}
        try:
            data = request.data.get("data")
            transformation_function = method.get(
                request.data.get("transformation_function")
            )

            # converting input data into df
            df = pd.DataFrame(data)
            # df after dataTypeCorrection
            output_df = numeric_transformations(df, transformation_function)
            return Response(output_df)
        except Exception as e:
            return Response({"msg": str(e)}, status=status.HTTP_400_BAD_REQUEST)


class CubeRootTransformation(BaseAPIView):
    """
    API endpoint that allows get the the result form the auto_ml.py corresponding function.
    - auto_ml.py stores all the ds related utility functions
    - This will convert CubeRootTransformation df into Json.

    * Requires JWT authentication.
    * This endpoint will allows only POST method.
    """

    permission_classes = (IsAuthenticated,)

    def post(self, request):
        method = {"cuberoottransformation": cube_root_transformation}
        try:
            data = request.data.get("data")
            transformation_function = method.get(
                request.data.get("transformation_function")
            )

            # converting input data into df
            df = pd.DataFrame(data)
            # df after dataTypeCorrection
            output_df = numeric_transformations(df, transformation_function)
            return Response(output_df)
        except Exception as e:
            return Response({"msg": str(e)}, status=status.HTTP_400_BAD_REQUEST)


class LogTransformation(BaseAPIView):
    """
    API endpoint that allows get the the result form the auto_ml.py corresponding function.
    - auto_ml.py stores all the ds related utility functions
    - This will convert LogTransformation df into Json.

    * Requires JWT authentication.
    * This endpoint will allows only POST method.
    """

    permission_classes = (IsAuthenticated,)

    def post(self, request):
        method = {"logtransformation": log_transformation}
        try:
            data = request.data.get("data")
            transformation_function = method.get(
                request.data.get("transformation_function")
            )

            # converting input data into df
            df = pd.DataFrame(data)
            # df after dataTypeCorrection
            output_df = numeric_transformations(df, transformation_function)
            return Response(output_df)
        except Exception as e:
            return Response({"msg": str(e)}, status=status.HTTP_400_BAD_REQUEST)


class SquareTransformation(BaseAPIView):
    """
    API endpoint that allows get the the result form the auto_ml.py corresponding function.
    - auto_ml.py stores all the ds related utility functions
    - This will convert SquareTransformation df into Json.

    * Requires JWT authentication.
    * This endpoint will allows only POST method.
    """

    permission_classes = (IsAuthenticated,)

    def post(self, request):
        method = {"squaretransformation": square_transformation}
        try:
            data = request.data.get("data")
            transformation_function = method.get(
                request.data.get("transformation_function")
            )

            # converting input data into df
            df = pd.DataFrame(data)
            # df after dataTypeCorrection
            output_df = numeric_transformations(df, transformation_function)
            return Response(output_df)
        except Exception as e:
            return Response({"msg": str(e)}, status=status.HTTP_400_BAD_REQUEST)


class DropDuplicateColumnsRowsAndUniqueValueColumns(BaseAPIView):
    """
    API endpoint that allows get the the result form the auto_ml.py corresponding function.
    - auto_ml.py stores all the ds related utility functions
    - This will convert DropDuplicateColumnsRowsAndUniqueValueColumns df into Json.

    * Requires JWT authentication.
    * This endpoint will allows only POST method.
    """

    permission_classes = (IsAuthenticated,)

    def post(self, request):
        try:
            data = request.data.get("data")
            # converting input data into df
            df = pd.DataFrame(data)
            # df after dataTypeCorrection
            output_df = drop_duplicate_columns_rows_and_unique_value_columns(df)
            return Response(output_df)
        except Exception as e:
            return Response({"msg": str(e)}, status=status.HTTP_400_BAD_REQUEST)


class FeatureGenerationByPCA(BaseAPIView):
    """
    API endpoint that allows get the the result form the auto_ml.py corresponding function.
    - auto_ml.py stores all the ds related utility functions
    - This will convert FeatureGenerationByPCA df into Json.

    * Requires JWT authentication.
    * This endpoint will allows only POST method.
    """

    permission_classes = (IsAuthenticated,)

    def post(self, request):
        try:
            data = request.data.get("data")
            target_column = request.data.get("target_column")
            # converting input data into df
            df = pd.DataFrame(data)
            # df after dataTypeCorrection
            output_df = feature_generation_by_pca(df, target_column)
            return Response(output_df)

        except Exception as e:
            return Response({"msg": str(e)}, status=status.HTTP_400_BAD_REQUEST)


class PreprocessingMethods(BaseAPIView):
    """
    API endpoint that allows get the the result form the auto_ml.py corresponding function.
    - auto_ml.py stores all the ds related utility functions
    - This will convert PreprocessingMethods df into Json.

    * Requires JWT authentication.
    * This endpoint will allows only POST method.
    """

    permission_classes = (IsAuthenticated,)

    def post(self, request):
        preprocessing_methods = {
            "data_type_correction": data_type_correction,
            "auto_imputer": auto_imputer,
            "auto_remove_unwanted_columns": auto_remove_unwanted_columns,
            "remove_correlated_columns": remove_correlated_columns,
            "standard_scale": standard_scale,
            "min_max_scale": min_max_scale,
            "robust_scale": robust_scale,
            "detecting_outliers_by_zscore": detecting_outliers_by_Zscore,
            "detecting_outliers_by_ior": detecting_outliers_by_IOR,
            "detecting_outliers_by_local_outlier_factor": detecting_outliers_by_local_outlier_factor,
            "detecting_outliers_by_isolation_forest": detecting_outliers_by_isolation_forest,
            "detecting_outliers_by_elliptic_envelope": detecting_outliers_by_elliptic_envelope,
            "labelencoding": label_encoding,
            "onehotencoding": categorical_value_encoder_transformer,
            "adasyn": adasyn,
            "smote": smote,
            # "standardscaler": standard_scale,
            # "robustscaler": robust_scaler,
            "cuberoottransformation": cube_root_transformation,
            "squareroottransformation": square_root_transformation,
            "logtransformation": log_transformation,
            "square_transformation": square_transformation,
            "dropduplicatecolumnsrowsanduniquevaluecolumns": drop_duplicate_columns_rows_and_unique_value_columns,
            "featuregenerationbypca": feature_generation_by_pca,
        }

        try:
            label_encoder = None
            input_preprocessing_methods = request.data.get("preprocessing")
            data = request.data.get("data")
            df = pd.DataFrame(data)

            target_variable = request.data.get("target_variable")
            threshold = (
                request.data.get("threshold") if request.data.get("threshold") else 0.90
            )
            target_column = request.data.get("target_variable")

            if "data_type_correction" in input_preprocessing_methods:
                preprocessing = preprocessing_methods.get("data_type_correction")
                df = preprocessing(df)
            if "auto_remove_unwanted_columns" in input_preprocessing_methods:
                preprocessing = preprocessing_methods.get(
                    "auto_remove_unwanted_columns"
                )
                df = preprocessing(df)
            if (
                "dropduplicatecolumnsrowsanduniquevaluecolumns"
                in input_preprocessing_methods
            ):
                preprocessing = preprocessing_methods.get(
                    "dropduplicatecolumnsrowsanduniquevaluecolumns"
                )
                df = preprocessing(df)
            if "auto_imputer" in input_preprocessing_methods:
                preprocessing = preprocessing_methods.get("auto_imputer")
                df = preprocessing(df)
            if "remove_correlated_columns" in input_preprocessing_methods:
                preprocessing = preprocessing_methods.get("remove_correlated_columns")
                target_variable = request.data.get("target_variable")
                df = preprocessing(df, target_variable, threshold)

            if "detecting_outliers_by_zscore" in input_preprocessing_methods:
                preprocessing = preprocessing_methods.get(
                    "detecting_outliers_by_zscore"
                )
                df = preprocessing(df)
            if "detecting_outliers_by_ior" in input_preprocessing_methods:
                preprocessing = preprocessing_methods.get("detecting_outliers_by_ior")
                df = preprocessing(df)
            if (
                "detecting_outliers_by_local_outlier_factor"
                in input_preprocessing_methods
            ):
                preprocessing = preprocessing_methods.get(
                    "detecting_outliers_by_local_outlier_factor"
                )
                df = preprocessing(df)
            if "detecting_outliers_by_isolation_forest" in input_preprocessing_methods:
                preprocessing = preprocessing_methods.get(
                    "detecting_outliers_by_isolation_forest"
                )
                df = preprocessing(df)
            if "detecting_outliers_by_elliptic_envelope" in input_preprocessing_methods:
                preprocessing = preprocessing_methods.get(
                    "detecting_outliers_by_elliptic_envelope"
                )
                df = preprocessing(df)

            if "onehotencoding" in input_preprocessing_methods:
                preprocessing = preprocessing_methods.get("onehotencoding")
                target_variable = request.data.get("target_variable")
                df = preprocessing(df, one_hot_encoding, target_variable)
            if "labelencoding" in input_preprocessing_methods:
                preprocessing = preprocessing_methods.get("labelencoding")
                target_variable = request.data.get("target_variable")
                df, label_encoder = preprocessing(df, target_variable)
                label_encoder = {
                    str(label): index for index, label in enumerate(label_encoder)
                }
            if "standard_scale" in input_preprocessing_methods:
                preprocessing = preprocessing_methods.get("standard_scale")
                df = preprocessing(df, target_column)

            if "robust_scale" in input_preprocessing_methods:
                preprocessing = preprocessing_methods.get("robust_scale")
                df = preprocessing(df, target_column)

            if "min_max_scale" in input_preprocessing_methods:
                preprocessing = preprocessing_methods.get("min_max_scale")
                df = preprocessing(df, target_column)

                df, scaler = df
            else:
                df, scaler = df

            return Response(
                {"data": df, "scaler_url": scaler, "label_encoder": label_encoder}
            )
        except Exception as e:
            return Response({"msg": str(e)}, status=status.HTTP_400_BAD_REQUEST)
