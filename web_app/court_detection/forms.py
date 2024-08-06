from django import forms
from .models import BasketballGame

class BasketballGameForm(forms.ModelForm):
    class Meta:
        model = BasketballGame
        fields = ['video']
