# import pytz
# from django.utils import timezone

# from dflux.db.models import Profile
# from rest_framework.views import APIView
# from rest_framework.request import Request as RestFrameworkRequest


# class TimezoneMiddleware(object):
#     """
#     Middleware to properly handle the users timezone
#     """

#     def __init__(self, get_response):
#         self.get_response = get_response

#     def __call__(self, request):
#         # get django rest framework current user
#         drf_request: RestFrameworkRequest = APIView().initialize_request(request)
#         user = drf_request.user
#         if user.is_authenticated:
#             profile = Profile.objects.filter(user=request.user).first()
#             try:
#                 if profile and profile.timezone is not None:
#                     tz_str = profile.timezone
#                     timezone.activate(pytz.timezone(tz_str))
#             except Exception as e:
#                 print(str(e))
#         else:
#             timezone.deactivate()

#         response = self.get_response(request)
#         return response
