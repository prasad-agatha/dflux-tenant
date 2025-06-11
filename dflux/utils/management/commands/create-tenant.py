from django.core.management.base import BaseCommand
from django.db import IntegrityError

from dflux.db.models import Tenant


class Command(BaseCommand):
    help = "Create a new tenant"

    def handle(self, *args, **options):
        name = input("Enter Tenant Name:")
        prefix = input("Enter Prefix URL:")
        try:
            tenant = Tenant.objects.create(name=name, subdomain_prefix=prefix)
            self.stdout.write(
                self.style.SUCCESS(f"Successfully Created with Tenant ID: {tenant.id}")
            )
        except IntegrityError:
            self.stderr.write("Tenant prefix already taken!")
