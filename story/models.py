from django.db import models

class Genre(models.Model):
    name = models.CharField(max_length=100, unique=True)
    description = models.TextField(blank=True)

    def __str__(self):
        return self.name

class Cliche(models.Model):
    genre = models.ForeignKey(Genre, on_delete=models.CASCADE, related_name='cliches')
    title = models.CharField(max_length=200)
    summary = models.TextField(help_text="클리셰 상세 설명")
    structure_guide = models.JSONField(help_text="스토리 흐름 단계 (발단-전개-절정-결말)")
    example_work_summary = models.TextField(help_text="대표 예시 작품 줄거리 및 감정선")

    def __str__(self):
        return self.title

class Story(models.Model):
    title = models.CharField(max_length=200, default="생성된 스토리")
    user_world_setting = models.TextField()
    
    main_cliche = models.ForeignKey(Cliche, on_delete=models.SET_NULL, null=True, related_name='stories')
    
    # [수정] 다중 분기를 위해 단일 분기 필드(twist_point_node_id, twisted_synopsis, twist_cliche)는 제거하거나
    # 하위 호환성을 위해 남겨둘 수 있으나, 로직상 StoryBranch를 사용하므로 여기서는 깔끔하게 정리합니다.
    # 만약 기존 데이터 보존이 필요하다면 필드를 남겨두셔도 됩니다.
    
    synopsis = models.TextField(help_text="초기 전체 시놉시스 (Main Plot)")
    created_at = models.DateTimeField(auto_now_add=True)

# [신규] 다중 분기 정보를 저장하는 모델
class StoryBranch(models.Model):
    story = models.ForeignKey(Story, on_delete=models.CASCADE, related_name='branches')
    # 어떤 노드에서 갈라져 나왔는지 (부모 노드)
    parent_node = models.ForeignKey('StoryNode', on_delete=models.CASCADE, related_name='child_branches')
    # 이 분기의 시놉시스
    synopsis = models.TextField(help_text="분기된 시놉시스")
    hierarchy_id = models.CharField(max_length=50, default="", help_text="분기 계층 ID")
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Branch from Node {self.parent_node_id}"

class CharacterState(models.Model):
    story = models.ForeignKey(Story, on_delete=models.CASCADE, related_name='character_states')
    character_name = models.CharField(max_length=100)
    state_data = models.JSONField(default=dict) 
    update_context = models.CharField(max_length=100, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.character_name} ({self.update_context})"

class StoryNode(models.Model):
    story = models.ForeignKey(Story, on_delete=models.CASCADE, related_name='nodes')
    chapter_phase = models.CharField(max_length=50)
    content = models.TextField(help_text="2000자 내외의 줄거리")
    
    # 순서 파악을 위한 깊이(Depth) 필드 추가 (권장)
    depth = models.IntegerField(default=0)
    
    prev_node = models.ForeignKey('self', on_delete=models.SET_NULL, null=True, blank=True, related_name='next_nodes')
    is_twist_point = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Node {self.id} ({self.chapter_phase})"

class NodeChoice(models.Model):
    current_node = models.ForeignKey(StoryNode, on_delete=models.CASCADE, related_name='choices')
    
    # [수정] DB 컬럼명에 맞춰 'choice_text'로 되돌림 (의미는 '필수 행동'으로 사용)
    choice_text = models.CharField(max_length=500)
    
    result_text = models.TextField(help_text="행동 직후 묘사")
    next_node = models.ForeignKey(StoryNode, on_delete=models.SET_NULL, null=True, related_name='incoming_choices')
    is_twist_path = models.BooleanField(default=False)

    def __str__(self):
        return self.choice_text[:50]