from decouple import config


from django.core.mail import EmailMessage
from django.template.loader import render_to_string

from django.core.mail import send_mail, EmailMultiAlternatives

from dflux.db.models import TenantUser
from dflux.api.views.utils import convert_file_django_object


def send_mail_to_users(to_email, subject, body):
    """
    This will allows send emails to users
    """
    email_from = config("DEFAULT_FROM_EMAIL")
    recipient_list = [to_email]
    subject = subject
    body = body
    email = EmailMessage(subject, body, email_from, recipient_list)
    email.content_subtype = "html"
    email.send()


class Email:
    def __init__(self):
        self.client_name = config("CLIENT_NAME")
        self.server_name = config("SERVER_NAME")
        self.client_logo = config("CLIENT_LOGO")
        self.from_mail = config("DEFAULT_FROM_EMAIL")
        self.team = config("TEAM")
        self.address_line1 = config("ADDRESS_LINE1")
        self.address_line2 = config("ADDRESS_LINE2")
        self.twitter_site = config("TWITTER_SITE")
        self.twitter_logo = config("TWITTER_LOGO")
        self.facebook_logo = config("FACEBOOK_LOGO")
        self.linkedin_logo = config("LINKEDIN_LOGO")
        self.website = config("WEBSITE")
        self.domain = config("DOMAIN_NAME")
        self.mail_id = config("MAIL_ID")
        self.from_email = config("DEFAULT_FROM_EMAIL")

    def send_registration_email(self, request, token, tenant):
        email = request.data.get("email")
        subject = f"{self.client_name} - Account created successfully"
        to_mail = [email]
        token = token
        template = "mail/welcome.html"
        body = render_to_string(
            template,
            {
                "user": request.data.get("username"),
                "server_name": self.server_name,
                "client_logo": self.client_logo,
                "mail_id": self.mail_id,
                "team": self.team,
                "address_line1": self.address_line1,
                "address_line2": self.address_line2,
                "twitter_site": self.twitter_site,
                "twitter_logo": self.twitter_logo,
                "facebook_logo": self.facebook_logo,
                "linkedin_logo": self.linkedin_logo,
                "website": self.website,
                "domain": tenant.subdomain_prefix,
                "token": token,
            },
        )
        send_mail(subject, body, self.from_mail, to_mail, html_message=body)

    def send_password_reset_email(self, user, email, uid):
        subject = f"{self.client_name}:Password reset request "
        template = "mail/password_reset.html"
        body = render_to_string(
            template,
            {
                "uid": uid,
                "origin": config("DOMAIN_NAME"),
                "user": user.first_name,
                "server_name": self.server_name,
                "client_logo": self.client_logo,
                "mail_id": self.mail_id,
                "team": self.team,
                "address_line1": self.address_line1,
                "address_line2": self.address_line2,
                "twitter_site": self.twitter_site,
                "twitter_logo": self.twitter_logo,
                "facebook_logo": self.facebook_logo,
                "linkedin_logo": self.linkedin_logo,
                "website": self.website,
                "tenant": user.tenant.subdomain_prefix,
            },
        )
        send_mail_to_users(to_email=email, subject=subject, body=body)

    def send_project_invitation_mail_to_users(self, users, request):
        """
        This function will allows send project invitation emails to users.
        """

        for user in users:
            subject = f"{self.client_name} - collaboration invitation for {user['project_name']} in {self.server_name}"
            to_mail = [user["invitee"]]
            for email in to_mail:
                to_user, at, company = email.rpartition("@")
                template = "mail/project_invite.html"
                body = render_to_string(
                    template,
                    {
                        "username": request.user.first_name,
                        "token": user["token"],
                        "origin": f"https://{request.user.tenant.subdomain_prefix}.dflux.ai",
                        # "invitee": request.data["invitee"],
                        "project_name": user["project_name"],
                        "server_name": self.server_name,
                        "client_logo": self.client_logo,
                        "mail_id": self.mail_id,
                        # "email": "team@dflux.com",
                        "team": self.team,
                        "address_line1": self.address_line1,
                        "address_line2": self.address_line2,
                        "twitter_site": self.twitter_site,
                        "twitter_logo": self.twitter_logo,
                        "facebook_logo": self.facebook_logo,
                        "linkedin_logo": self.linkedin_logo,
                        "website": self.website,
                        "to_user": to_user,
                    },
                )
                send_mail(subject, body, self.from_mail, to_mail, html_message=body)

    def send_chart_email(self, request, url):
        to_mail = request.data.get("emails")
        chart_name = request.data.get("name")
        template = "mail/chart_sent_email.html"
        body = render_to_string(
            template,
            {
                "username": f"{request.user.first_name} {request.user.last_name}",
                "chart_name": chart_name,
                "url": url,
                "team": self.team,
                "server_name": self.server_name,
                "client_name": self.client_name,
                "client_logo": self.client_logo,
                "mail_id": self.mail_id,
                "address_line1": self.address_line1,
                "address_line2": self.address_line2,
                "twitter_site": self.twitter_site,
                "twitter_logo": self.twitter_logo,
                "facebook_logo": self.facebook_logo,
                "linkedin_logo": self.linkedin_logo,
                "website": self.website,
            },
        )
        subject = f"{self.client_name} - Shared  chart - {chart_name}"
        email_prefix = config("EMAIL_PREFIX")
        send_mail(
            subject,
            body,
            f"{email_prefix} <{self.from_mail}>",
            to_mail,
            html_message=body,
        )
        return

    def send_dashboard_email(self, request, url):
        to_mail = request.data.get("emails")
        dashboard_name = request.data.get("name")
        template = "mail/dashboard_sent_email.html"
        body = render_to_string(
            template,
            {
                "username": f"{request.user.first_name} {request.user.last_name}",
                "dashboard_name": dashboard_name,
                "url": url,
                "team": self.team,
                "server_name": self.server_name,
                "client_name": self.client_name,
                "client_logo": self.client_logo,
                "mail_id": self.mail_id,
                "address_line1": self.address_line1,
                "address_line2": self.address_line2,
                "twitter_site": self.twitter_site,
                "twitter_logo": self.twitter_logo,
                "facebook_logo": self.facebook_logo,
                "linkedin_logo": self.linkedin_logo,
                "website": self.website,
            },
        )
        subject = f"{self.client_name} - Shared  dashboard - {dashboard_name}"
        email_prefix = config("EMAIL_PREFIX")
        send_mail(
            subject,
            body,
            f"{email_prefix} <{self.from_mail}>",
            to_mail,
            html_message=body,
        )
        return

    def send_contact_email(self, request, email, support_email):
        profile = TenantUser.objects.filter(user=request.user).first()
        body = request.data.get("description")
        ip = request.META.get("REMOTE_ADDR")
        template = "mail/contact-sales-submitter-dflux.html"
        body = render_to_string(
            template,
            {
                "username": f"{request.user.first_name} {request.user.last_name}",
                "team": config("TEAM"),
                "server_name_two": config("SERVER_NAME_TWO"),
                "server_name": config("SERVER_NAME"),
                "address_line1": config("ADDRESS_LINE1"),
                "address_line2": config("ADDRESS_LINE2"),
                "twitter_site": config("TWITTER_SITE"),
                "twitter_logo": config("TWITTER_LOGO"),
                "facebook_logo": config("FACEBOOK_LOGO"),
                "linkedin_logo": config("LINKEDIN_LOGO"),
                "website": config("WEBSITE"),
                "client_logo": config("CLIENT_LOGO"),
            },
        )
        input_subject = request.data.get("subject")
        email_prefix = config("EMAIL_PREFIX")
        submiter_email_subject = f"{self.client_name} -  Sales request: {input_subject}"
        send_mail(
            submiter_email_subject,
            body,
            f"{email_prefix} <{self.from_email}>",
            [email],
            html_message=body,
        )

        # send mail to receiver
        template = "mail/contact-sales-reciever-dflux.html"
        body = render_to_string(
            template,
            {
                "message": request.data.get("message"),
                "username": f"{request.user.first_name} {request.user.last_name}",
                "contact_number": profile.contact_number
                if profile.contact_number
                else "",
                "email": request.user.email,
                "company": profile.company if profile.company else "",
                "industry": profile.industry if profile.industry else "",
                "role": profile.role if profile.role else "",
                "subject": request.data.get("subject"),
                "team": config("TEAM"),
                "server_name": config("SERVER_NAME"),
                "server_name_two": config("SERVER_NAME_TWO"),
                "address_line1": config("ADDRESS_LINE1"),
                "address_line2": config("ADDRESS_LINE2"),
                "twitter_site": config("TWITTER_SITE"),
                "twitter_logo": config("TWITTER_LOGO"),
                "facebook_logo": config("FACEBOOK_LOGO"),
                "linkedin_logo": config("LINKEDIN_LOGO"),
                "website": config("WEBSITE"),
                "client_logo": config("CLIENT_LOGO"),
                "client_name": config("CLIENT_NAME"),
                "ip": ip,
            },
        )
        receiver_email_subject = f"{self.client_name} -  Sales request: {input_subject}"
        email_prefix = config("EMAIL_PREFIX")
        send_mail(
            receiver_email_subject,
            body,
            f"{email_prefix} <{self.from_email}>",
            [support_email],
            html_message=body,
        )

        return

    def send_support_email(self, request, email, support_email):
        profile = TenantUser.objects.filter(user=request.user).first()
        body = request.data.get("description")
        ip = request.META.get("REMOTE_ADDR")
        input_subject = request.data.get("subject")
        template = "mail/support-request-submitter-dflux.html"
        # send mail to submiter
        body = render_to_string(
            template,
            {
                "username": f"{request.user.first_name} {request.user.last_name}",
                "team": config("TEAM"),
                "server_name": config("SERVER_NAME"),
                "address_line1": config("ADDRESS_LINE1"),
                "address_line2": config("ADDRESS_LINE2"),
                "twitter_site": config("TWITTER_SITE"),
                "twitter_logo": config("TWITTER_LOGO"),
                "facebook_logo": config("FACEBOOK_LOGO"),
                "linkedin_logo": config("LINKEDIN_LOGO"),
                "website": config("WEBSITE"),
                "client_logo": config("CLIENT_LOGO"),
            },
        )
        email_prefix = config("EMAIL_PREFIX")
        submiter_email_subject = (
            f"{self.client_name} -  Support request: {input_subject}"
        )
        send_mail(
            submiter_email_subject,
            body,
            f"{email_prefix} <{self.from_email}>",
            [email],
            html_message=body,
        )

        # send mail to support
        template = "mail/support-request-reciever-dflux.html"
        body = render_to_string(
            template,
            {
                "username": f"{request.user.first_name} {request.user.last_name}",
                "contact_number": profile.contact_number
                if profile.contact_number
                else "",
                "email": request.user.email,
                "company": profile.company if profile.company else "",
                "industry": profile.industry if profile.industry else "",
                "role": profile.role if profile.role else "",
                "subject": request.data.get("subject"),
                "description": request.data.get("description"),
                "attachments": request.data.get("attachments"),
                "team": config("TEAM"),
                "server_name": config("SERVER_NAME"),
                "server_name_two": config("SERVER_NAME_TWO"),
                "address_line1": config("ADDRESS_LINE1"),
                "address_line2": config("ADDRESS_LINE2"),
                "twitter_site": config("TWITTER_SITE"),
                "twitter_logo": config("TWITTER_LOGO"),
                "facebook_logo": config("FACEBOOK_LOGO"),
                "linkedin_logo": config("LINKEDIN_LOGO"),
                "website": config("WEBSITE"),
                "client_logo": config("CLIENT_LOGO"),
                "client_name": config("CLIENT_NAME"),
                "ip": ip,
            },
        )
        email_prefix = config("EMAIL_PREFIX")

        receiver_email_subject = (
            f"{self.client_name} -  Support request: {input_subject}"
        )
        try:
            mail = EmailMultiAlternatives(
                receiver_email_subject,
                body,
                f"{email_prefix} <{self.from_email}>",
                [support_email],
            )

            attachments = request.data.get("attachment")
            if attachments:
                f = convert_file_django_object(attachments)
                mail.attach(f.name, f.read(), f.content_type)
                mail.attach_alternative(body, "text/html")
                mail.send()
            else:
                mail.attach_alternative(body, "text/html")
                mail.send()
        except Exception as e:
            print(str(e))

        return

    def send_team_invitation_mail_to_users(self, users, request):
        """
        This method will allows us send emails to users.
        """
        for user in users:
            subject = "Invite to join Team"
            to_mail = [user["user"]]
            template = "mail/team_invite.html"
            for email in to_mail:
                to_user, at, company = email.rpartition("@")
                body = render_to_string(
                    template,
                    {
                        "username": request.user.first_name,
                        "token": user["token"],
                        "client_logo": self.client_logo,
                        "mail_id": self.mail_id,
                        # "email": "team@dflux.com",
                        "team": self.team,
                        "server_name": self.server_name,
                        "address_line1": self.address_line1,
                        "address_line2": self.address_line2,
                        "twitter_site": self.twitter_site,
                        "twitter_logo": self.twitter_logo,
                        "facebook_logo": self.facebook_logo,
                        "linkedin_logo": self.linkedin_logo,
                        "website": self.website,
                        "to_user": to_user,
                    },
                )
                send_mail(subject, body, self.from_mail, to_mail, html_message=body)

    def send_trigger_success_email(self, trigger, project):
        template = "mail/trigger-execution.html"
        body = render_to_string(
            template,
            {
                "user": trigger.project.user.username,
                "chart_name": trigger.charttrigger.name,
                "updated_at": trigger.updated,
                "email": self.from_mail,
                "updated_at": trigger.updated,
                "server_name": self.server_name,
                "client_logo": self.client_logo,
                "address_line1": self.address_line1,
                "address_line2": self.address_line2,
                "twitter_site": self.twitter_site,
                "twitter_logo": self.twitter_logo,
                "facebook_logo": self.facebook_logo,
                "linkedin_logo": self.linkedin_logo,
                "website": self.website,
                "team": self.team,
            },
        )
        send_mail(
            f"{self.client_name} -Trigger Successful Remainder for {trigger.name}, {project.name}",
            body,
            # f"Trigger Executed successfully At {now.strftime('%Y-%m-%d %H:%M:%S')}.",
            config("DEFAULT_FROM_EMAIL"),
            trigger.email,
            html_message=body,
        )
        return

    def send_trigger_error_email(self, trigger, e):
        template = "mail/trigger-error.html"
        body = render_to_string(
            template,
            {
                "user": trigger.project.user.first_name,
                "error": f"{str(e)}",
                "chart_name": trigger.charttrigger.name,
                "updated_at": trigger.updated,
                "server_name": self.server_name,
                "client_logo": self.client_logo,
                "address_line1": self.address_line1,
                "address_line2": self.address_line2,
                "twitter_site": self.twitter_site,
                "twitter_logo": self.twitter_logo,
                "facebook_logo": self.facebook_logo,
                "linkedin_logo": self.linkedin_logo,
                "website": self.website,
                "team": self.team,
            },
        )
        send_mail(
            f"{self.client_name} - Action required - {trigger.name} failed",
            body,
            # f"{str(e)}",
            self.mail_id,
            trigger.email,
            html_message=body,
        )
        return

    def send_chart_link_email(self, chart_trigger, token):
        subject = (
            f"{self.client_name} - You’re invited to view the {chart_trigger.name}"
        )
        to_mail = chart_trigger.email
        template = "mail/share_trigger_chart_link.html"
        for email in to_mail:
            to_user, at, company = email.rpartition("@")
            body = render_to_string(
                template,
                {
                    "domain": self.domain,
                    "token": token,
                    "user": chart_trigger.project.user.first_name,
                    "chart_name": chart_trigger.name,
                    "team": self.team,
                    "to_user": to_user,
                    "server_name": self.server_name,
                    "client_logo": self.client_logo,
                    "address_line1": self.address_line1,
                    "address_line2": self.address_line2,
                    "twitter_site": self.twitter_site,
                    "twitter_logo": self.twitter_logo,
                    "facebook_logo": self.facebook_logo,
                    "linkedin_logo": self.linkedin_logo,
                    "website": self.website,
                },
            )
            send_mail(subject, body, self.from_mail, to_mail, html_message=body)
            return

    def send_chart_link_error_email(self, chart_trigger, e):
        send_mail(
            f"{chart_trigger.name}",
            f"{str(e)}",
            self.mail_id,
            [chart_trigger.email],
        )
        return

    def tenant_invitation_mail(self, to_mail, token, request):
        """
        This function will allows send tenant invitation emails to users.
        """

        subject = f"{request.user.tenant.subdomain_prefix} invitation"
        to_mail = [to_mail]
        template = "mail/tenant_invitation.html"
        domain_name = config("DOMAIN_NAME")
        body = render_to_string(
            template,
            {
                "token": token,
                "username": request.user.email,
                "tenant_id": request.user.tenant.id,
                "tenant_subdomain_prefix": request.user.tenant.subdomain_prefix,
                "fname": request.user.first_name,
                "lname": request.user.last_name,
                "email": request.data.get("email"),
                "origin": f"{domain_name}",
                "server_name": self.server_name,
                "client_logo": self.client_logo,
                "mail_id": self.mail_id,
                "team": self.team,
                "address_line1": self.address_line1,
                "address_line2": self.address_line2,
                "twitter_site": self.twitter_site,
                "twitter_logo": self.twitter_logo,
                "facebook_logo": self.facebook_logo,
                "linkedin_logo": self.linkedin_logo,
                "website": self.website,
                "company": request.user.company,
            },
        )
        send_mail(subject, body, self.from_mail, to_mail, html_message=body)

    def send_tenant_registration_email(self, user):
        subject = f"{self.client_name}:Tenant Registration "
        template = "mail/tenant_registration.html"
        app_domain = config("DOMAIN_NAME")
        body = render_to_string(
            template,
            {
                "origin": f"{app_domain}/tenants/{user.username}/set-password/{user.token}?company={user.company}&email={user.email}&name={user.first_name}&is_admin={user.tenant_superuser}&subdomain_prefix={user.tenant.subdomain_prefix}",
                "username": user.first_name,
                "server_name": self.server_name,
                "client_logo": self.client_logo,
                "mail_id": self.mail_id,
                "team": self.team,
                "address_line1": self.address_line1,
                "address_line2": self.address_line2,
                "twitter_site": self.twitter_site,
                "twitter_logo": self.twitter_logo,
                "facebook_logo": self.facebook_logo,
                "linkedin_logo": self.linkedin_logo,
                "website": self.website,
            },
        )
        send_mail_to_users(to_email=user.email, subject=subject, body=body)


emails = Email()
