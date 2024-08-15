from rest_framework import serializers
from .models import ImageClassificationModel

class ImageClassificationModelSerializer(serializers.ModelSerializer):
	class Meta:
		model = ImageClassificationModel
		fields = '__all__'