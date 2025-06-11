from django import forms

from dflux.db.models import Tenant, TenantUser


class TenantUserForm(forms.ModelForm):
    email = forms.EmailField(required=True)

    class Meta:
        model = TenantUser
        fields = ["username", "email", "password"]


class TenantForm(forms.ModelForm):
    class Meta:
        model = Tenant
        fields = ["name", "subdomain_prefix"]

    def clean_subdomain_prefix(self):
        subdomain_prefix = self.cleaned_data["subdomain_prefix"]
        special_characters = "!@#$%^&*()-+?_=,<>/,''"
        if subdomain_prefix:
            for special_char in special_characters:
                if special_char in subdomain_prefix:
                    raise forms.ValidationError("special characters are not allowed")
        return subdomain_prefix


class TenantUpdateForm(forms.ModelForm):
    class Meta:
        model = Tenant
        fields = ["name", "subdomain_prefix"]
