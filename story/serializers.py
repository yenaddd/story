from rest_framework import serializers
from .models import StoryNode, NodeChoice

class NodeChoiceSerializer(serializers.ModelSerializer):
    next_node_id = serializers.ReadOnlyField(source='next_node.id')
    # choice_text -> action_text 변경
    text = serializers.CharField(source='action_text') 

    class Meta:
        model = NodeChoice
        fields = ('id', 'text', 'next_node_id', 'is_twist_path')

class StoryNodeSerializer(serializers.ModelSerializer):
    choices = NodeChoiceSerializer(many=True, read_only=True)

    class Meta:
        model = StoryNode
        fields = ('id', 'chapter_phase', 'content', 'choices')