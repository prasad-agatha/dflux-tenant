FROM python:3.8.16

RUN apt-get update && apt-get install -y --no-install-recommends \
    unixodbc-dev \
    unixodbc \
    libpq-dev \
    odbc-postgresql

WORKDIR /apps


COPY /requirements ./requirements
COPY requirements.txt ./
RUN ls
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .

# install odbc drivers
RUN dpkg -i odbc-drivers/*.deb

RUN python manage.py collectstatic --noinput
RUN python manage.py migrate 

EXPOSE 8000

ENV DJANGO_SETTINGS_MODULE dflux.settings.production

CMD ["uvicorn", "dflux.asgi:application", "--host", "0.0.0.0", "--port", "8000", "--workers", "5"]