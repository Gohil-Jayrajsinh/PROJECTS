from django.db import models

# # Create your models here.

class data(models.Model):
    date = models.CharField(max_length=50)
    name = models.CharField(max_length=100)
    enter_time = models.CharField(max_length=50)
    exit_time = models.CharField(max_length=50)
    exit_date = models.CharField(max_length=50)



class emp_data(models.Model):
    name = models.CharField(max_length=100)
    img = models.CharField(max_length=100)
    
    


