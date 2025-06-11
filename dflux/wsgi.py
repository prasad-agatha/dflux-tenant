"""
WSGI config for dflux project.

It exposes the WSGI callable as a module-level variable named ``application``.

"""

import os
import dotenv
from django.core.wsgi import get_wsgi_application

dotenv.load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dflux.settings.production")


application = get_wsgi_application()
