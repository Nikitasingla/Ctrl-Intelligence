# from django import forms
# from .models import UploadedDocument

# class DocumentUploadForm(forms.ModelForm):
#     competitor_ticker = forms.CharField(
#         max_length=20, 
#         required=False,
#         widget=forms.TextInput(attrs={
#             'placeholder': 'e.g., INFY.NS for Infosys',
#             'class': 'form-control'
#         }),
#         help_text='Optional: Enter competitor stock ticker for comparison'
#     )
    
#     class Meta:
#         model = UploadedDocument
#         fields = ['file']
#         widgets = {
#             'file': forms.FileInput(attrs={
#                 'class': 'form-control',
#                 'accept': '.pdf,.csv,.xlsx,.xls'
#             })
#         }
    
#     def clean_file(self):
#         file = self.cleaned_data.get('file')
#         if file:
#             # Check file size (50MB limit)
#             if file.size > 50 * 1024 * 1024:
#                 raise forms.ValidationError('File size cannot exceed 50MB.')
            
#             # Check file type
#             valid_extensions = ['.pdf', '.csv', '.xlsx', '.xls']
#             file_extension = '.' + file.name.split('.')[-1].lower()
#             if file_extension not in valid_extensions:
#                 raise forms.ValidationError(
#                     'Invalid file type. Please upload PDF, CSV, or Excel files only.'
#                 )
#         return file




# forms.py - Corrected version

from django import forms
from .models import UploadedDocument

class DocumentUploadForm(forms.ModelForm):
    competitor_ticker = forms.CharField(
        max_length=20, 
        required=False,
        widget=forms.TextInput(attrs={
            'placeholder': 'e.g., RELIANCE.NS, GOOGL, TATASTEEL.BO',
            'class': 'form-control'
        }),
        # FIX: More descriptive help text for the user.
        help_text='Optional: Enter a valid stock ticker from Yahoo Finance. Add market suffix for non-US stocks (e.g., ".NS" for NSE, ".BO" for BSE).'
    )
    
    class Meta:
        model = UploadedDocument
        fields = ['file']
        widgets = {
            'file': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': '.pdf,.csv,.xlsx,.xls'
            })
        }
    
    def clean_file(self):
        file = self.cleaned_data.get('file')
        if file:
            # Check file size (50MB limit)
            if file.size > 50 * 1024 * 1024:
                raise forms.ValidationError('File size cannot exceed 50MB.')
            
            # Check file type
            valid_extensions = ['.pdf', '.csv', '.xlsx', '.xls']
            file_extension = '.' + file.name.split('.')[-1].lower()
            if file_extension not in valid_extensions:
                raise forms.ValidationError(
                    'Invalid file type. Please upload PDF, CSV, or Excel files only.'
                )
        return file