import os
import dotenv
from django.core.asgi import get_asgi_application


dotenv.load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dflux.settings.production")


application = get_asgi_application()
