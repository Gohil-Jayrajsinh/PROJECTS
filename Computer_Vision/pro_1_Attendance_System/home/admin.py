from django.contrib import admin
from home.models import data, emp_data

admin.site.site_title = "Attendance System administration"
admin.site.site_header = "Attendance System administration"
admin.site.index_title = "Attendance System administration"

# Register your models here.

admin.site.register(data)
admin.site.register(emp_data)


