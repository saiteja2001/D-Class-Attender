from dataclasses import fields
from django import forms
from .models import *

class PhotoForm(forms.ModelForm):
    image = forms.ImageField(
        label='Image',
        widget=forms.ClearableFileInput(attrs={"multiple": True}),
    )

    class Meta:
        model = Photo
        fields = ("image",)

