from allauth.socialaccount.adapter import DefaultSocialAccountAdapter
from allauth.socialaccount import app_settings


class CustomSocialAccountAdapter(DefaultSocialAccountAdapter):
    def is_auto_signup_allowed(self, request, sociallogin):
        # If email is verified, go straight to signup.
        return True
    
    def get_app(self, request, provider):
        from allauth.socialaccount.models import SocialApp
        try:
            return SocialApp.objects.get(provider=provider)
        except SocialApp.MultipleObjectsReturned:
            # Handle the case where multiple objects are found.
            raise Exception(f"Multiple social apps found for provider: {provider}")
