## Run Locally

Clone the project

```bash
  git clone https://github.com/soulpage/dflux-api.git
```

Go to the project directory

```bash
  cd dflux-api
```

Create virtual environment in linux

```bash
  python3 -m venv venv
```

Create virtual environment in windows

```bash
  python -m venv venv
```

Activate virtual environment in linux

```bash
 source venv/bin/activate
 ```

 Activate virtual environment in windows

```bash
 venv\Scripts\activate.bat
 ```

Install dependencies

```bash
  pip install -r requirements/local.txt
```

Migrate poject models

```bash
   python manage.py makemigrations --settings=dflux.settings.local
   ```

```bash
   python manage.py migrate --settings=dflux.settings.local
```

Start the server

```bash
 python manage.py runserver --settings=dflux.settings.local
```

Once server is started below output will render in your terminal

```bash
Watching for file changes with StatReloader
Performing system checks...

System check identified no issues (0 silenced).
September 30, 2021 - 03:09:58
Django version 3.2.5, using settings 'dflux.settings.local'
Starting ASGI/Channels version 3.0.4 development server at http://127.0.0.1:8000/
Quit the server with CONTROL-C.
```


if you get any error related to to environment varible create  .env file in project directory and the below environment varible in that file
## Environment Variables

To run this project, you will need to add the following environment variables to your .env file

`DB_NAME=<DB_NAME>`

`DB_USER=<DB_USER>`

`DB_PASSWORD=<DB_PASSWORD>`

`DB_HOST=<DB_HOST>`

`DOMAIN_NAME=<DOMAIN_NAME>`

`SECRET_KEY=<SECRET_KEY>`

`EMAIL_BACKEND=<EMAIL_BACKEND>`

`DEFAULT_FROM_EMAIL=<DEFAULT_FROM_EMAIL>`

`SERVER_EMAIL=<SERVER_EMAIL>`

`MAILGUN_API_KEY=<MAILGUN_API_KEY>`

`MAILGUN_SENDER_DOMAIN=<MAILGUN_SENDER_DOMAIN>`

`AWS_REGION=<AWS_REGION>`

`AWS_S3_BUCKET_NAME=<AWS_S3_BUCKET_NAME>`

`AWS_ACCESS_KEY_ID=<AWS_ACCESS_KEY_ID>`

`AWS_SECRET_ACCESS_KEY=<AWS_SECRET_ACCESS_KEY>`


Now go to browser and type the below url you will see the welcome page of the project

```bash
    http://localhost:8000/
```
