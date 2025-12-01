# story/models.py
from django.db import models
import json

class Genre(models.Model):
    name = models.CharField(max_length=100, unique=True)  # 추리, 로맨스, SF 등
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
    user_world_setting = models.TextField() # 사용자가 입력한 세계관
    main_cliche = models.ForeignKey(Cliche, on_delete=models.SET_NULL, null=True, related_name='stories')
    
    # 8번: 클리셰 변주(Twist) 정보
    twist_cliche = models.ForeignKey(Cliche, on_delete=models.SET_NULL, null=True, blank=True, related_name='twisted_stories')
    twist_point_node_id = models.IntegerField(null=True, blank=True) # 변주가 일어나는 분기점 노드 ID
    
    synopsis = models.TextField(help_text="전체 시놉시스 (초기)")
    twisted_synopsis = models.TextField(blank=True, help_text="변주 후 변경된 시놉시스")
    created_at = models.DateTimeField(auto_now_add=True)

class CharacterState(models.Model):
    """
    4, 10번: 인물 내면 변화 DB
    각 사건/챕터 진행 시점마다 인물의 상태를 스냅샷으로 저장하거나, 
    누적된 상태를 관리하기 위해 사용합니다.
    """
    story = models.ForeignKey(Story, on_delete=models.CASCADE, related_name='character_states')
    character_name = models.CharField(max_length=100)
    
    # 감정적, 정서적, 육체적, 사상적, 신뢰적, 애정적, 관계적 변화를 JSON으로 저장
    # 예: {"emotion": "angry", "trust_in_hero": 10, "ideology": "conflicted"}
    state_data = models.JSONField(default=dict) 
    last_updated_node_id = models.IntegerField(null=True, blank=True) # 어떤 노드까지 반영되었는지

    def __str__(self):
        return f"{self.character_name} in {self.story.title}"

class StoryNode(models.Model):
    story = models.ForeignKey(Story, on_delete=models.CASCADE, related_name='nodes')
    chapter_phase = models.CharField(max_length=50) # 발단, 전개, 절정, 결말
    content = models.TextField(help_text="2000자 내외의 줄거리")
    
    # 순서 제어를 위한 필드 (Linked List 형태 권장)
    prev_node = models.ForeignKey('self', on_delete=models.SET_NULL, null=True, blank=True, related_name='next_nodes')
    
    is_twist_point = models.BooleanField(default=False) # 여기가 장르가 바뀌는 지점인가?
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Node {self.id} ({self.chapter_phase})"

class NodeChoice(models.Model):
    current_node = models.ForeignKey(StoryNode, on_delete=models.CASCADE, related_name='choices')
    choice_text = models.CharField(max_length=500)
    
    # 7번: 선택지에 따른 직후 행동 결과 (다음 노드 앞부분에 붙을 텍스트)
    result_text = models.TextField(help_text="선택 직후 행동 묘사")
    
    # 다음 노드 (대부분 2개의 선택지가 같은 next_node를 가리킴)
    next_node = models.ForeignKey(StoryNode, on_delete=models.SET_NULL, null=True, related_name='incoming_choices')
    
    # 12번: 변주 지점 식별용 (True면 새로운 장르 루트로 이동)
    is_twist_path = models.BooleanField(default=False)

    def __str__(self):
        return self.choice_text[:50]