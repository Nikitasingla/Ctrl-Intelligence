# dashboard/templatetags/dashboard_extras.py

from django import template

register = template.Library()

@register.filter(name='get_item')
def get_item(dictionary, key):
    """Allows dictionary lookup with a variable key in templates."""
    return dictionary.get(key)