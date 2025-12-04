from rest_framework import serializers
from .models import StoryNode, NodeChoice

class NodeChoiceSerializer(serializers.ModelSerializer):
    next_node_id = serializers.ReadOnlyField(source='next_node.id')
    # [수정] 소스를 다시 choice_text로 변경 (API 출력명인 'text'는 유지)
    text = serializers.CharField(source='choice_text') 

    class Meta:
        model = NodeChoice
        fields = ('id', 'text', 'next_node_id', 'is_twist_path')

class StoryNodeSerializer(serializers.ModelSerializer):
    choices = NodeChoiceSerializer(many=True, read_only=True)

    class Meta:
        model = StoryNode
        fields = ('id', 'chapter_phase', 'content', 'choices')