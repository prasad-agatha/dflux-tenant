from django.urls import path

from dflux.api.views.ml.loading import LoadCSVData, LoadDBData
from dflux.api.views.ml.preprocessing import (
    DataTypeCorrection,
    AutoRemoveUnwantedcolumns,
    AutoImputer,
    RemoveCorrelatedColumns,
    LabelEncoding,
    OneHotEncoding,
    OrdinalEncoder,
    Adasyn,
    Smote,
    # CategoricalValueEncoder,
    StandardScale,
    MinMaxScale,
    PreprocessingMethods,
    DetectingOutliersByZscore,
    DetectingOutliersByIOR,
    DetectingOutliersByLocalOutlinerFactor,
    DetectingOutliersByIsolationForest,
    DetectingOutliersByEllipiticEnvelope,
    StandardScaler,
    RobustScaler,
    MinMaxScaling,
    SquareRootTransformation,
    CubeRootTransformation,
    LogTransformation,
    SquareTransformation,
    DropDuplicateColumnsRowsAndUniqueValueColumns,
    FeatureGenerationByPCA,
)

from dflux.api.views.ml.modelling import (
    TrainAndTestSplit,
    LogisticRegressionModel,
    SupportVectorClassifier,
    DecisionTreeClassifier,
    RandomForestClassifier,
    XGBClassifier,
    KNNClassifier,
    NaiveBayesClassifier,
    MultinomailNBClassifier,
    AdaBoostClassifier,
    MultiLayerPerceptronClassifier,
    ModellingMethods,
)
from dflux.api.views.ml.regression import (
    LinearRegression,
    SupportVectorRegressor,
    DecisionTreeRegressor,
    RandomForestRegressor,
    XGBRegressor,
    RegressionMethods,
    KNeighborsRegressor,
    PolynomialRegression,
    LassoRegressor,
    RidgeRegressor,
    ElasticnetRegression,
    SGDRegression,
    GradientboostingRegression,
    # LGBMRegression,
    # CatBoostRegression,
)
from dflux.api.views.ml.prediction import Predict
from dflux.api.views.ml.time_series import (
    TimeSeriesPreprocessingEndpoint,
    TimeSeriesMetaDataEndpoint,
    TimeSeriesPredictionEndpoint,
)


urlpatterns = [
    # loading
    path("load-csv/", LoadCSVData.as_view(), name="load-csv-data"),
    path("load-db-data/", LoadDBData.as_view(), name="load-db-data"),
    # preprocessing
    path(
        "data-type-correction/",
        DataTypeCorrection.as_view(),
        name="data-type-correction",
    ),
    path(
        "auto-remove-unwanted-columns/",
        AutoRemoveUnwantedcolumns.as_view(),
        name="auto-remove-unwanted-columns",
    ),
    path("auto-imputer/", AutoImputer.as_view(), name="auto-imputer"),
    path(
        "preprocessingmethods/",
        PreprocessingMethods.as_view(),
        name="preprocessingmethods",
    ),
    path(
        "remove-correlated-columns/",
        RemoveCorrelatedColumns.as_view(),
        name="remove-correlated-columns",
    ),
    path(
        "label-encoding/",
        LabelEncoding.as_view(),
        name="label-encoding",
    ),
    path(
        "onehotencoding/",
        OneHotEncoding.as_view(),
        name="onehotencoding",
    ),
    path(
        "ordinalencoder/",
        OrdinalEncoder.as_view(),
        name="ordinalencoder",
    ),
    path(
        "adasyn/",
        Adasyn.as_view(),
        name="adasyn",
    ),
    path(
        "smote/",
        Smote.as_view(),
        name="smote",
    ),
    path("standard-scale/", StandardScale.as_view(), name="standard-scale"),
    path("min-max-scale/", MinMaxScale.as_view(), name="min-max-scale"),
    path("standardscaler/", StandardScaler.as_view(), name="standardscaler"),
    path("robustscaler/", RobustScaler.as_view(), name="robustscaler"),
    path("minmaxscaling/", MinMaxScaling.as_view(), name="minmaxscaling"),
    path(
        "detectingoutliersbyZscore/",
        DetectingOutliersByZscore.as_view(),
        name="detectingoutliersbyZscore",
    ),
    path(
        "detectingoutliersbyIOR/",
        DetectingOutliersByIOR.as_view(),
        name="detectingoutliersbyZscore",
    ),
    path(
        "detectingoutliersbylocaloutlinerfactor/",
        DetectingOutliersByLocalOutlinerFactor.as_view(),
        name="detectingoutliersbyZscore",
    ),
    path(
        "detectingoutliersbyisolationforest/",
        DetectingOutliersByIsolationForest.as_view(),
        name="detectingoutliersbyZscore",
    ),
    path(
        "detectingoutliersbyellipiticenvelope/",
        DetectingOutliersByEllipiticEnvelope.as_view(),
        name="detectingoutliersbyZscore",
    ),
    path(
        "squareroottransformation/",
        SquareRootTransformation.as_view(),
        name="squareroottransformation",
    ),
    path(
        "cuberoottransformation/",
        CubeRootTransformation.as_view(),
        name="cuberoottransformation",
    ),
    path(
        "logtransformation/",
        LogTransformation.as_view(),
        name="logtransformation",
    ),
    path(
        "squaretransformation/",
        SquareTransformation.as_view(),
        name="squaretransformation",
    ),
    path(
        "dropduplicatecolumnsrowsanduniquevaluecolumns/",
        DropDuplicateColumnsRowsAndUniqueValueColumns.as_view(),
        name="dropduplicatecolumnsrowsanduniquevaluecolumns",
    ),
    path(
        "featuregenerationbypca/",
        FeatureGenerationByPCA.as_view(),
        name="featuregenerationbypca",
    ),
    # modelling
    path("train-test-split/", TrainAndTestSplit.as_view(), name="train-test-split"),
    path(
        "logistic-regression-model/",
        LogisticRegressionModel.as_view(),
        name="logistic-regression-model",
    ),
    path(
        "support-vector-classifier/",
        SupportVectorClassifier.as_view(),
        name="support-vector-classifier",
    ),
    path(
        "decision-tree-classifier/",
        DecisionTreeClassifier.as_view(),
        name="decision-tree-classifier",
    ),
    path(
        "random-forest-classifier/",
        RandomForestClassifier.as_view(),
        name="random-forest-classifier",
    ),
    path(
        "xgb-classifier/",
        XGBClassifier.as_view(),
        name="xgb-classifier",
    ),
    path(
        "knn-classifier/",
        KNNClassifier.as_view(),
        name="knn-classifier",
    ),
    path(
        "naivebayesclassifier/",
        NaiveBayesClassifier.as_view(),
        name="naivebayesclassifier",
    ),
    path(
        "multinomailnbclassifier/",
        MultinomailNBClassifier.as_view(),
        name="multinomailnbclassifier",
    ),
    path(
        "adaboostclassifier/",
        AdaBoostClassifier.as_view(),
        name="adaboostclassifier",
    ),
    path(
        "multilayerperceptronclassifier/",
        MultiLayerPerceptronClassifier.as_view(),
        name="multilayerperceptronclassifier",
    ),
    path("modellingmethods/", ModellingMethods.as_view(), name=" modellingmethods"),
    # regression
    path(
        "linear-regression/",
        LinearRegression.as_view(),
        name="linear-regression",
    ),
    path(
        "support-vector-regression/",
        SupportVectorRegressor.as_view(),
        name="support-vector-regression",
    ),
    path(
        "decision-tree-regression/",
        DecisionTreeRegressor.as_view(),
        name="decision-tree-regression",
    ),
    path(
        "random-forest-regression/",
        RandomForestRegressor.as_view(),
        name="random-forest-regression",
    ),
    path(
        "xgb-regression/",
        XGBRegressor.as_view(),
        name="xgb-regression",
    ),
    path(
        "kneighborsregressor/",
        KNeighborsRegressor.as_view(),
        name="kneighborsregressor",
    ),
    path(
        "polynomialregression/",
        PolynomialRegression.as_view(),
        name="polynomialregression",
    ),
    path(
        "regressionmethods/",
        RegressionMethods.as_view(),
        name="regressionmethods",
    ),
    path(
        "lassoregressor/",
        LassoRegressor.as_view(),
        name="lassoregressor",
    ),
    path(
        "ridgeregressor/",
        RidgeRegressor.as_view(),
        name="ridgeregressor",
    ),
    path(
        "elasticnetregression/",
        ElasticnetRegression.as_view(),
        name="elasticnetregression",
    ),
    path(
        "sgdregression/",
        SGDRegression.as_view(),
        name="sgdregression",
    ),
    path(
        "gradientboostingregression/",
        GradientboostingRegression.as_view(),
        name="gradientboostingregression",
    ),
    # path(
    #     "lgbmregression/",
    #     LGBMRegression.as_view(),
    #     name="gradientboostingregression",
    # ),
    # path(
    #     "catboostregression/",
    #     CatBoostRegression.as_view(),
    #     name="gradientboostingregression",
    # ),
    # prediction,
    path(
        "models/<int:pk>/prediction/",
        Predict.as_view(),
        name="prediction",
    ),
    # time series
    path(
        "timeseries/preprocessing/",
        TimeSeriesPreprocessingEndpoint.as_view(),
        name="timeseries-preprocessing",
    ),
    path(
        "timeseries/metadata/",
        TimeSeriesMetaDataEndpoint.as_view(),
        name="timeseries-metadata",
    ),
    path(
        "models/<int:pk>/timeseries/prediction/",
        TimeSeriesPredictionEndpoint.as_view(),
        name="timeseries-prediction",
    ),
    # path(
    #     "target-label-encoder/",
    #     CategoricalValueEncoder.as_view(),
    #     name="target-label-encoder",
    # ),
]
