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
    twist_cliche = models.ForeignKey(Cliche, on_delete=models.SET_NULL, null=True, blank=True, related_name='twisted_stories')
    twist_point_node_id = models.IntegerField(null=True, blank=True)
    
    synopsis = models.TextField(help_text="초기 전체 시놉시스")
    twisted_synopsis = models.TextField(blank=True, help_text="변주 후 변경된 시놉시스")
    created_at = models.DateTimeField(auto_now_add=True)

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