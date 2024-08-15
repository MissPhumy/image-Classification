from django.db import models
from PIL import Image

class ImageClassificationModel(models.Model):
    # name = models.CharField(max_length=100)
    image = models.ImageField(upload_to='images/')
    predicted_class = models.CharField(max_length=100, blank=True)
    date_created = models.DateTimeField(auto_now_add=True)


    def __str__(self):
        return self.predicted_class