from rest_framework import serializers
class AnalyzeSerializer(serializers.Serializer):
    text = serializers.CharField(max_length=10000, allow_blank=False, trim_whitespace=True)
