# import pandas as pd
import os, re, asyncio, pyodbc

from urllib.request import urlretrieve

from django.db import connection
from django.db import connection as db_connection

from django.core.files.base import ContentFile
from django.core.files.uploadedfile import InMemoryUploadedFile

from dflux.db.models import TenantUser

from dflux.db.models import ExcelData, JsonData

from dflux.api.views.query import connection_establishment, query_invoke

from dflux.api.views.ml.auto_ml import (
    model_evaluation_for_classification,
    model_evaluation_for_regression,
)

from dflux.api.views.ml.auto_ml import (
    lp_out_cross_validation,
    lo_out_cross_validation,
    hold_out_cross_validation,
    rep_rand_sample_cross_validation,
    k_fold_cross_validation,
    k_fold_stratified_cross_validation,
    nested_cross_validation,
)

from rest_framework.response import Response
from rest_framework_simplejwt.tokens import RefreshToken

from dflux.api.views.ml.config import modelling_methods, cross_validation_techniques


def get_tokens_for_user(user):
    """
    This method allows to generate JWT access, refresh token.
    """
    refresh = RefreshToken.for_user(user)
    print(refresh)

    return {
        "refresh": str(refresh),
        "access": str(refresh.access_token),
    }


def generate_signup_token():
    """
    - This method will generate token in signup time
    - This token will be use full for user reset their passwords.
    """
    import uuid

    token = uuid.uuid4().hex + uuid.uuid4().hex
    return token


def create_database_string(request):
    """
    This function will allows create database connection string with required details.
    """
    db = {
        "name": request.data.get("name"),
        "engine": request.data.get("engine"),
        "dbname": request.data.get("dbname"),
        "username": request.data.get("username"),
        "password": request.data.get("password"),
        "port": request.data.get("port"),
        "host": request.data.get("host"),
    }
    return db


def check_user_status(email):
    """
    This method will allows check the user email exits or not in database.
    - if user email exits it will return True else it will return False
    """
    if TenantUser.objects.filter(email=email).exists():
        return True
    return False


def generate_token(user, project):
    """
    This method will allows generate JWT token using the user object, project.
    """
    import jwt

    JWT_SECRET = "soulpage"
    JWT_ALGORITHM = "HS256"
    payload = {"user": user, "project": project, "user_status": check_user_status(user)}
    jwt_token = jwt.encode(payload, JWT_SECRET, JWT_ALGORITHM, {"exp": "24hr"})
    return str(jwt_token).strip("b").strip("'")


def convert_file_django_object(url):
    """
    This function will allows create django file object using the s3 bucket url.
    """
    file_name = "attachment"
    urlretrieve(url, file_name)
    # Get the Django file object
    with open(file_name, "rb") as img:
        image = ContentFile(img.read())
        django_file_object = InMemoryUploadedFile(
            image,
            None,
            f"{file_name}.jpg",
            "image/jpeg",
            image.tell,
            None,
        )
        if os.path.exists(file_name):
            os.remove(file_name)


#         return django_file_object


def create_table(table_name=None, columns=None, data_type=None, file_type=None):
    """
    Create new table if table not exits in the database using dynamic columns and datatypes.
    """
    cursor = connection.cursor()
    if file_type == "csv_or_excel":
        sql_query = (
            """CREATE TABLE IF NOT EXISTS """
            + table_name
            + " ("
            + ",".join(
                [
                    "{k} {v}".format(
                        k=columns[i], v=data_type.get(columns[i], "varchar")
                    )
                    for i in range(len(columns))
                ]
            )
            + ")"
            ""
        )
    else:
        sql_query = (
            """CREATE TABLE IF NOT EXISTS """
            + table_name
            + " ("
            + " VARCHAR(2000),".join(columns)
            + " VARCHAR(2000))"
            ""
        )

    # print(sql_query)
    cursor.execute(sql_query)
    connection.commit()
    print("Table Created.")


def insert_data_into_table(table_name=None, columns=None, records=None, file_type=None):
    """
    Insert records into table if given table name exits in the database.
    """
    cursor = connection.cursor()
    if file_type == "csv_or_excel":
        for record in records:
            insert_sql_query = (
                "INSERT INTO "
                + table_name
                + "("
                + ",".join(columns)
                + ") VALUES"
                + str(record).replace("None", "NULL")
            )
            # print(insert_sql_query)
            cursor.execute(insert_sql_query)
    else:
        sql_query = (
            "INSERT INTO "
            + table_name
            + "("
            + ",".join(columns)
            + ") VALUES"
            + "("
            + ",".join(["%s" for colum in columns])
            + ")"
        )
        cursor.executemany(sql_query, records)
    connection.commit()
    return f"Table {table_name} created and Records inserted successfully."


def delete_table(table_name):
    """
    Drop the table if given table name exits in the database.
    """
    cursor = connection.cursor()
    sql_query = """DROP TABLE IF EXISTS """ + table_name
    cursor.execute(sql_query)
    print("Table deleted.")
    return True


def get_excel_columns(db):
    query = "select column_name from information_schema.columns where table_name = '{tablename}'".format(
        tablename=db[0].tablename
    )

    cursor = connection.cursor()
    cursor.execute(query)
    queryset = cursor.fetchall()
    columns = list(
        map(
            lambda v: re.sub(r"\(('\w+'),\)", r"\1", "".join(v)),
            queryset,
        )
    )
    return columns


def internal_connection_schema(request, db):
    excel_id = request.query_params.get("excel_id")
    json_id = request.query_params.get("json_id")
    table_id = request.query_params.get("table_id")
    zip_id = request.query_params.get("zip_id")

    if excel_id is not None:
        db = ExcelData.objects.filter(
            connection=db,
            id=request.query_params.get("excel_id"),
            # user=request.user,
        )

    if json_id is not None:
        db = JsonData.objects.filter(
            connection=db,
            id=request.query_params.get("json_id"),
            # user=request.user,
        )
    # if table_id is not None:
    #     db = CustomTable.objects.filter(
    #         connection=db,
    #         id=request.query_params.get("table_id"),
    #         # user=request.user,
    #     )
    # if zip_id is not None:
    #     db = ZipFileTable.objects.filter(
    #         connection=db,
    #         id=request.query_params.get("zip_id"),
    #         # user=request.user,
    #     )

    # if db is not None:
    if len(db) > 0:
        # query = "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'your_table_name' AND table_schema = 'your_schema_name'"
        query = "select table_name,array_agg(column_name) as columns,array_agg(data_type) as datatype from information_schema.columns where table_schema not in ('information_schema', 'pg_catalog') and table_name in ({tablename}) group by  table_name,table_schema order by table_schema,table_name".format(
            tablename=",".join(list(map(lambda x: "'{}'".format(x.tablename), db)))
        )
        cursor = db_connection.cursor()
        cursor.execute(query)
        columns = [column[0] for column in cursor.description]
        results = []
        for row in cursor.fetchall():
            newdict = {
                k: v if k != "columns" and k != "datatype" else re.findall("\w+", v)
                for (k, v) in zip(columns, row)
            }
            results.append(
                {
                    "table_name": newdict["table_name"],
                    "columns": [
                        {"name": name, "type": datatype}
                        for name, datatype in zip(
                            newdict["columns"],
                            newdict["datatype"],
                        )
                    ],
                    "suggestions": [name for name in newdict["columns"]],
                }
            )
        return results


def external_connection_schema(db):

    # database connection establishment
    conn = connection_establishment(db)
    cursor = conn.cursor()

    if "mysql" in str(db.engine):
        query = "show tables"
    elif "oracle" in str(db.engine):
        query = "SELECT table_name From all_tables WHERE owner='ADMIN' ORDER BY owner, table_name;"
    elif "mssql" in str(db.engine):
        query = "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES"
    else:
        # query = "SELECT table_name FROM information_schema.tables WHERE table_schema='public' AND table_type='BASE TABLE';"
        query = "select table_name,array_agg(column_name) as columns,array_agg(data_type) as datatype from information_schema.columns where table_schema not in ('information_schema', 'pg_catalog') group by  table_name,table_schema order by table_schema,table_name"
    try:
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(query_invoke(cursor, query))
        loop.close()
        data = [
            {
                "table_name": table["table_name"],
                "columns": [
                    {"name": name, "type": datatype}
                    for name, datatype in zip(
                        re.findall("\w+", table["columns"]),
                        re.findall("\w+", table["datatype"]),
                    )
                ],
                "suggestions": [name for name in re.findall("\w+", table["columns"])],
            }
            for table in result
        ]

        return data
    except pyodbc.ProgrammingError as e:
        return Response({"error": str(e)})


def classification_method_result(request, method):
    # cross validation techniques
    cross_validation_techniques = {
        "leave_p_out": lp_out_cross_validation,
        "leave_one_out": lo_out_cross_validation,
        "hold_out": hold_out_cross_validation,
        "repeated_random_sampling": rep_rand_sample_cross_validation,
        "k_fold": k_fold_cross_validation,
        "k_fold_stratified": k_fold_stratified_cross_validation,
        "nested_k_fold": nested_cross_validation,
    }

    data = request.data.get("data")
    target_variable = request.data.get("target_variable")
    output_file_name = request.data.get("output_file_name")
    custom_selection = request.data.get("custom_selection")
    fit_model = method
    cv = request.data.get("cv")
    gcv = request.data.get("gcv")
    cv_params = request.data.get("cv_params")
    gcv_params = request.data.get("gcv_params")
    cross_validation_ = cross_validation_techniques.get(
        request.data.get("cross_validation"), None
    )

    # converting input data into df
    df = pd.DataFrame(data)
    (
        confusion_matrix_result,
        classification_report_result_mean,
        classification_report_result_reset_index,
        accuracy_score,
        labels_order,
        false_positive_rate,
        true_positive_rate,
        thresholds,
        auc_score,
        x_test,
        datetime,
        model_status,
        pickle_url,
        result,
    ) = model_evaluation_for_classification(
        df,
        target_variable,
        output_file_name,
        fit_model,
        custom_selection,
        cv,
        cross_validation_,
        cv_params,
        gcv,
        gcv_params,
    )
    result = {
        "confusion_matrix_result": confusion_matrix_result,
        "classification_report_result_mean": classification_report_result_mean,
        "classification_report_result_reset_index": classification_report_result_reset_index,
        "accuracy_score": accuracy_score,
        "labels_order": labels_order,
        "datetime": datetime,
        "false_positive_rate": false_positive_rate,
        "true_positive_rate": true_positive_rate,
        "thresholds": thresholds,
        "auc_score": [{int(key): val} for key, val in auc_score.items()],
        "x_test": x_test,
        "model_status": model_status,
        "pickle_url": pickle_url,
        "result": result,
    }
    return result


def multiple_classification_methods(
    request,
    input_modelling_method,
    df,
    target_variable,
    output_file_name,
    modelling,
    custom_selection,
    cross_validation_techniques,
):
    cv = request.data.get("hyper_params").get(input_modelling_method).get("cv")
    gcv = request.data.get("hyper_params").get(input_modelling_method).get("gcv")
    cv_params = (
        request.data.get("hyper_params").get(input_modelling_method).get("cv_params")
    )
    gcv_params = (
        request.data.get("hyper_params").get(input_modelling_method).get("gcv_params")
    )
    cross_validation_ = cross_validation_techniques.get(
        request.data.get("hyper_params")
        .get(input_modelling_method)
        .get("cross_validation")
    )
    (
        confusion_matrix_result,
        classification_report_result_mean,
        classification_report_result_reset_index,
        accuracy_score,
        labels_order,
        false_positive_rate,
        true_positive_rate,
        thresholds,
        auc_score,
        X_test,
        datetime,
        model_status,
        pickle_url,
        result,
    ) = model_evaluation_for_classification(
        df,
        target_variable,
        output_file_name,
        modelling,
        custom_selection,
        cv,
        cross_validation_,
        cv_params,
        gcv,
        gcv_params,
    )
    model_response = {
        "input_modelling_method": input_modelling_method,
        "confusion_matrix_result": confusion_matrix_result,
        "classification_report_result_mean": classification_report_result_mean,
        "classification_report_result_reset_index": classification_report_result_reset_index,
        "accuracy_score": accuracy_score,
        "labels_order": labels_order,
        "false_positive_rate": false_positive_rate,
        "true_positive_rate": true_positive_rate,
        "thresholds": thresholds,
        "auc_score": [{int(key): val} for key, val in auc_score.items()],
        "X_test": X_test,
        "datetime": datetime,
        "model_status": model_status,
        "pickle_url": pickle_url,
        "result": result,
    }
    return model_response


def regression_methods(
    request,
    df,
    target_variable,
    output_file_name,
    input_modelling_method,
    modelling,
    custom_selection,
):
    cv = request.data.get("hyper_params").get(input_modelling_method).get("cv")
    gcv = request.data.get("hyper_params").get(input_modelling_method).get("gcv")
    cv_params = (
        request.data.get("hyper_params").get(input_modelling_method).get("cv_params")
    )
    gcv_params = (
        request.data.get("hyper_params").get(input_modelling_method).get("gcv_params")
    )
    cross_validation_ = cross_validation_techniques.get(
        request.data.get("hyper_params")
        .get(input_modelling_method)
        .get("cross_validation")
    )

    rmse_score, X_test, pickle_url, result = model_evaluation_for_regression(
        df,
        target_variable,
        output_file_name,
        modelling,
        custom_selection,
        cv,
        cross_validation_,
        cv_params,
        gcv,
        gcv_params,
    )
    # NaN to blank value
    X_test.fillna("", inplace=True),
    return {
        "rmse_score": rmse_score,
        "x_test": X_test,
        "pickle_url": pickle_url,
        "result": result,
    }
