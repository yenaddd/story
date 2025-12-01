from django.db import models

class StoryNode(models.Model):
    """
    스토리의 각 노드를 정의하는 모델 (V3: 캐릭터 관계 및 서사 정보 포함)
    """
    id = models.AutoField(primary_key=True)
    depth = models.IntegerField(default=0)
    freytag_stage = models.CharField(max_length=50, blank=True)
    
    # 이 노드를 유발한 부모 노드와 선택지 정보 (재생성 로직 추적용)
    parent_node_id = models.IntegerField(null=True, blank=True)
    parent_choice_text = models.CharField(max_length=500, blank=True, default="")
    
    # 관계 발전 추적을 위한 필드 (JSON 형태로 저장)
    current_relationships = models.JSONField(default=dict, blank=True)
    
    # 주인공의 심리/감정 상태 (서사 추적용)
    protagonist_state = models.CharField(max_length=255, default="", blank=True) 

    # 줄거리 텍스트 데이터 (최대 600자)
    story_text = models.TextField()
    
    # 감독 AI의 최종 평가
    critique_score = models.IntegerField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Node {self.id} (Depth: {self.depth}, Stage: {self.freytag_stage})"

class NodeChoice(models.Model):
    """
    특정 노드에서 다음 노드로 이어지는 선택지와 연결 정보를 정의
    """
    parent_node = models.ForeignKey(
        StoryNode, 
        related_name='choices', 
        on_delete=models.CASCADE
    )
    # 선택지는 '당신의 입장'에서 서술됨을 프롬프트로 강제합니다.
    choice_text = models.CharField(max_length=500)
    
    next_node = models.OneToOneField(
        StoryNode, 
        related_name='parent_choice', 
        on_delete=models.SET_NULL,
        null=True, 
        blank=True
    )

    def __str__(self):
        return f"Choice from {self.parent_node.id}: {self.choice_text[:30]}..."

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=['parent_node', 'choice_text'], name='unique_choice_per_node')
        ]
