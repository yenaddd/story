from rest_framework import serializers
from .models import StoryNode, NodeChoice

class NodeChoiceSerializer(serializers.ModelSerializer):
    next_node_id = serializers.ReadOnlyField(source='next_node.id')

    class Meta:
        model = NodeChoice
        fields = ('id', 'choice_text', 'next_node_id')

class StoryNodeSerializer(serializers.ModelSerializer):
    choices = NodeChoiceSerializer(many=True, read_only=True)

    class Meta:
        model = StoryNode
        fields = ('id', 'depth', 'freytag_stage', 'story_text', 'critique_score', 
                  'current_relationships', 'protagonist_state', 'choices',
                  'parent_node_id', 'parent_choice_text') # <--- 필드 추가
