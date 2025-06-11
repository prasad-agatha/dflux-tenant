from django.http.response import HttpResponse
from django.shortcuts import render
from django.views import View
from django.shortcuts import get_object_or_404, redirect

from dflux.db.models.base import TenantUser

from .forms import (
    TenantForm,
    TenantUpdateForm,
    TenantUserForm,
)
from dflux.db.models import Tenant
from dflux.utils.emails import emails

# Create your views here.


class TenantView(View):
    def get(self, request):
        context = {"form": TenantForm(), "user_form": TenantUserForm()}
        return render(request, "tenant/tenant.html", context)

    def post(self, request):
        form = TenantForm(request.POST)
        username = request.POST.get("username")
        email = request.POST.get("email")
        password = request.POST.get("password")
        if form.is_valid():
            tenant = form.save()
            # create user in tenant
            user = TenantUser.objects.create(
                username=username,
                email=email,
                tenant=tenant,
                tenant_superuser=True,
                role="admin",
            )
            user.set_password(password)
            user.save()
            # tenant registation email
            emails.send_tenant_registration_email(user)
            context = {"tenants": Tenant.objects.all()}
            return render(request, "tenant/tenant_list.html", context)

        else:
            return HttpResponse("Tenant, admin creation failed")


class TenantDetailsView(View):
    def get(self, request):
        context = {"tenants": Tenant.objects.all()}
        return render(request, "tenant/tenant_list.html", context)


class TenantUpdateView(View):
    def get(self, request, pk):
        context = {}
        obj = get_object_or_404(Tenant, id=pk)
        tenant_superuser = TenantUser.objects.filter(
            tenant=obj, tenant_superuser=True
        ).first()
        if tenant_superuser is not None:
            context["superuser"] = (
                tenant_superuser.email if tenant_superuser.email else ""
            )

        form = TenantUpdateForm(request.POST or None, instance=obj)
        context["form"] = form

        return render(request, "tenant/update_tenant.html", context)

    def post(self, request, pk):
        context = {}
        obj = get_object_or_404(Tenant, id=pk)

        form = TenantUpdateForm(request.POST or None, instance=obj)

        if form.is_valid():
            form.save()
            return redirect("tenants_list")

        # add form dictionary to context
        context["form"] = form

        return render(request, "tenant/update_tenant.html", context)
