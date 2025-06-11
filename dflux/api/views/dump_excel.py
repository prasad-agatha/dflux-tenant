import json
import numpy as np
import pandas as pd

from django.db.models import Q
from django.shortcuts import get_object_or_404

from rest_framework import status
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated

from dflux.api.serializers import ExcelDataSerializer
from dflux.api.views.base import BaseAPIView
from dflux.db.models import Connection, Project, ExcelData

from .filters import ExcelDataFilter
from .utils import create_table, insert_data_into_table, delete_table, get_excel_columns
from .permissions import (
    ProjectOwnerOrCollaborator,
    ProjectExcelDataAccess,
    ProjectModuleAccess,
)


class DumpExcelData(BaseAPIView):
    """
    API endpoint that allows dump the excel, csv, google sheets into the database.
    This endpoint will return list of all the excel, csv, google sheet tables.

    * Requires JWT authentication.
    * Only collaborator or owner of the project can access.
    * This endpoint will allows only GET, POST methods.
    """

    permission_classes = (IsAuthenticated, ProjectModuleAccess)

    def get(self, request, pk):
        """
        This method will return list of all the excel, csv, google sheet tables.
        """
        try:
            queryset = ExcelDataFilter(
                request.GET,
                queryset=ExcelData.objects.select_related("project", "connection")
                .filter(
                    Q(created_by__id=request.user.id, project__id=pk)
                    | Q(project__id=pk, project__projectmembers__user=request.user),
                )
                .distinct(),
            ).qs
            serializer = ExcelDataSerializer(queryset, many=True)
            return Response(serializer.data, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({"msg": str(e)}, status=status.HTTP_400_BAD_REQUEST)

    def post(self, request, pk):
        """
        This method will allows dump the excel, csv, google sheets into the database.
        """
        try:
            project = Project.objects.get(id=pk)
            internal, is_created = Connection.objects.get_or_create(
                project=project,
                connection_type="INTERNAL",
            )
            table_name = request.data.get("table_name").replace(" ", "_").lower()
            excel_file = request.FILES.get("file")
            sheet_name = request.data.get("sheet_name")
            sheet_url = request.data.get("sheet_url")
            data_type = json.loads(request.data.get("data_types"))

            # if file type is google sheet url
            file_type = request.data.get("file_type")
            if file_type is not None and file_type == "google_sheets":
                url = sheet_url
                url_extention = "export?format=xlsx"
                google_sheet_url = f"{url}/{url_extention}"
                xls = pd.ExcelFile(google_sheet_url)
                sheets_name = xls.sheet_names
                df = xls.parse(sheet_name)

            # if file is csv or excel
            else:
                try:
                    df = pd.read_excel(excel_file, sheet_name=sheet_name).replace(
                        {"'": ""}, regex=True
                    )
                except:
                    df = pd.read_csv(excel_file, sep=",").replace({"'": ""}, regex=True)

            # cleaned dataframe
            cleaned_df = df.fillna(np.nan).replace([np.nan], [None])
            columns = list(data_type.keys())

            # convert all dates cols into str
            date_cols = cleaned_df.select_dtypes(include="datetime64")
            for col in date_cols.columns:
                cleaned_df[col] = cleaned_df[col].astype(str)

            create_table(
                table_name=table_name,
                columns=columns,
                data_type=data_type,
                file_type="csv_or_excel",
            )
            ExcelData.objects.get_or_create(
                project=project,
                connection=internal,
                tablename=table_name,
                file_type=file_type,
            )

            # dump data
            records = cleaned_df.to_records(index=False)
            response = insert_data_into_table(
                table_name=table_name,
                columns=columns,
                records=records.tolist(),
                file_type="csv_or_excel",
            )
            return Response({"msg": response})
        except Exception as e:
            file_type = request.data.get("file_type")
            table_name = request.data.get("table_name").replace(" ", "_").lower()
            internal, is_created = Connection.objects.get_or_create(
                project=project,
                connection_type="INTERNAL",
            )
            delete_table(table_name)
            if ExcelData.objects.filter(
                project=project,
                connection=internal,
                tablename=table_name,
                file_type=file_type,
            ).exists():
                excel_object = ExcelData.objects.filter(
                    project=project,
                    connection=internal,
                    tablename=table_name,
                    file_type=file_type,
                )
                excel_object.delete()
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)


class DumpExcelDataDetail(BaseAPIView):
    """
    API endpoint that allows view, delete individual Excel table data.

    * Requires JWT authentication.
    * Only collaborator or owner of the project can access.
    * This endpoint will allows only GET, DELETE methods.
    """

    permission_classes = (IsAuthenticated, ProjectExcelDataAccess)

    def get(self, request, pk):
        """
        This method will allows view individual Excel data details.
        """

        try:
            db = (
                ExcelData.objects.select_related("user", "project", "connection")
                .filter(id=pk, user=request.user)
                .first()
            )
            if db:
                serializer = ExcelDataSerializer(db)
                columns = get_excel_columns(db)
                serializer.data["columns"] = columns
                return Response(serializer.data)

            else:
                return Response([])
        except:
            return Response(
                {"error": "Please give valid Excel id"},
                status=status.HTTP_400_BAD_REQUEST,
            )

    def delete(self, request, pk, format=None):
        """
        This method will allows delete individual Excel data details.
        """
        excel_data = get_object_or_404(ExcelData, id=pk)
        delete_table(excel_data.tablename)
        excel_data.delete()
        return Response(
            {"message": "Delete Success"}, status=status.HTTP_204_NO_CONTENT
        )


class GoogleSheetParserEndpoint(BaseAPIView):
    """
    API endpoint that allows parse the google sheet and return the field name and type.
    - If google sheet url must be public access.

    * Authentication not required.
    * This endpoint will allows only POST method.
    """

    def post(self, request):
        """
        This method will allows parse the google sheet url and return the each field datatype.
        """

        try:
            url = request.data.get("sheet_url")
            url_extention = "export?format=xlsx"
            google_sheet_url = f"{url}/{url_extention}"
            # df = pd.read_excel(google_sheet_url)
            xls = pd.ExcelFile(google_sheet_url)

            sheets_name = xls.sheet_names

            sheets = []
            for i in sheets_name:
                df = xls.parse(i)
                # cleaned dataframe
                cleaned_df = df.fillna(np.nan).replace([np.nan], [None])

                # creating table with excelfile headers
                columns = [col.replace(" ", "_").lower() for col in cleaned_df.columns]
                # convert all float cols into int
                # float_cols = cleaned_df.select_dtypes(include="float64")
                # for col in float_cols.columns:
                #     cleaned_df[col] = cleaned_df[col].astype(int)

                # convert all dates cols into str
                date_cols = cleaned_df.select_dtypes(include="datetime64")
                for col in date_cols.columns:
                    cleaned_df[col] = cleaned_df[col].astype(str)

                data_types_initialization = {
                    "int64": "int",
                    "float64": "float",
                    "datetime64": "date",
                    "object": "varchar",
                }
                data_types = []

                for column, type in zip(columns, cleaned_df.dtypes):
                    if not column.find("unnamed") != -1:
                        column_ = {
                            "field": column,
                            "type": data_types_initialization.get(str(type)),
                        }
                        data_types.append(column_)
                sheet = {"name": i, "columns": data_types}
                sheets.append(sheet)
            return Response({"sheets": sheets})
        except Exception as e:
            return Response(
                {"error": "google sheet must be public access"},
                status=status.HTTP_400_BAD_REQUEST,
            )
