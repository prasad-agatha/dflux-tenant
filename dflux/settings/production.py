"""Production settings and globals."""
from .common import *  # noqa
import dj_database_url
from decouple import config


import sentry_sdk
from sentry_sdk.integrations.django import DjangoIntegration

sentry_sdk.init(
    environment="production",
    dsn=config("SENTRY_DSN"),
    integrations=[DjangoIntegration()],
    # Set traces_sample_rate to 1.0 to capture 100%
    # of transactions for performance monitoring.
    # We recommend adjusting this value in production.
    traces_sample_rate=0.4,
    # If you wish to associate users to errors (assuming you are using
    # django.contrib.auth) you may enable sending PII data.
    send_default_pii=True,
)

# Database
DEBUG = False

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql_psycopg2",
        "NAME": config("DB_NAME"),
        "USER": config("DB_USER"),
        "PASSWORD": config("DB_PASSWORD"),
        "HOST": config("DB_HOST"),
    }
}


CACHES = {
    "default": {
        "BACKEND": "django_redis.cache.RedisCache",
        "LOCATION": os.environ.get("REDIS_URL"),
        "OPTIONS": {
            "CLIENT_CLASS": "django_redis.client.DefaultClient",
        },
    }
}

from logtail import LogtailHandler
import logging

handler = LogtailHandler(source_token="svd95181VE1bJKtYcUHYEUne", level=logging.INFO)

logger = logging.getLogger(__name__)
logger.handlers = []
logger.addHandler(handler)

import os

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
        },
    },
    "root": {
        "handlers": ["console"],
        "level": "WARNING",
    },
    "loggers": {
        "django": {
            "handlers": ["console"],
            "level": os.getenv("DJANGO_LOG_LEVEL", "INFO"),
            "propagate": False,
        },
    },
}

# CORS WHITELIST ON PROD
CORS_ORIGIN_WHITELIST = [
    # "https://example.com",
    # "https://sub.example.com",
    # "http://localhost:8080",
    # "http://127.0.0.1:9000"
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://dflux-cms.vercel.app",
    "https://insight.intellect2.ai",
    "https://*.dflux.ai",
]
# Parse database configuration from $DATABASE_URL
# DATABASES["default"] = dj_database_url.config()
SITE_ID = 2

# Enable Connection Pooling (if desired)
# DATABASES['default']['ENGINE'] = 'django_postgrespool'

# Honor the 'X-Forwarded-Proto' header for request.is_secure()
SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")

# Allow all host headers
ALLOWED_HOSTS = ["*"]


# Simplified static file serving.
STATICFILES_STORAGE = "whitenoise.storage.CompressedManifestStaticFilesStorage"
CORS_ORIGIN_ALLOW_ALL = True

# The AWS region to connect to.
AWS_REGION = config("AWS_REGION")
# The AWS access key to use.
AWS_ACCESS_KEY_ID = config("AWS_ACCESS_KEY_ID")
# The AWS secret access key to use.
AWS_SECRET_ACCESS_KEY = config("AWS_SECRET_ACCESS_KEY")
# The name of the bucket to store files in.
AWS_S3_BUCKET_NAME = config("AWS_S3_BUCKET_NAME")

# To upload  media files to S3
DEFAULT_FILE_STORAGE = "django_s3_storage.storage.S3Storage"

AWS_S3_BUCKET_AUTH = False
