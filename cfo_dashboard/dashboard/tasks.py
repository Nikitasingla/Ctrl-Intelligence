from celery import shared_task
from .utils import generate_gemini_response

@shared_task
def process_message(user_message):
    return generate_gemini_response(user_message)