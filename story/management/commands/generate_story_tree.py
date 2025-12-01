from django.core.management.base import BaseCommand
from story.models import StoryNode, NodeChoice, Story
from story.llm_api import create_story_pipeline
from django.db import connection

class Command(BaseCommand):
    help = '터미널에서 스토리 생성 파이프라인을 실행합니다.'

    def add_arguments(self, parser):
        parser.add_argument('--world', type=str, required=True, help='세계관/인물 설정')
        # [수정] --arc, --branches 인자 제거 (더 이상 사용하지 않음)

    def handle(self, *args, **options):
        world_setting = options['world']
        
        self.stdout.write(self.style.WARNING('--- 스토리 생성 프로세스 시작 ---'))

        # 1. 기존 데이터 삭제 (초기화)
        Story.objects.all().delete()
        NodeChoice.objects.all().delete()
        StoryNode.objects.all().delete()
        
        # SQLite 시퀀스 초기화 (선택 사항)
        with connection.cursor() as cursor:
            cursor.execute("UPDATE sqlite_sequence SET seq = 0 WHERE name='story_storynode'")
            cursor.execute("UPDATE sqlite_sequence SET seq = 0 WHERE name='story_nodechoice'")

        # 2. 파이프라인 실행
        try:
            self.stdout.write("AI가 스토리를 생성 중입니다. (1~2분 소요)...")
            story_id = create_story_pipeline(world_setting)
            
            root_node = StoryNode.objects.filter(story_id=story_id).order_by('id').first()
            
            if root_node:
                self.stdout.write(self.style.SUCCESS(f'스토리 생성 완료! Root ID: {root_node.id}'))
                self.stdout.write(f"첫 문장: {root_node.content[:50]}...")
            else:
                self.stdout.write(self.style.ERROR("스토리 생성 실패: 노드가 생성되지 않았습니다."))
                
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"오류 발생: {e}"))
            import traceback
            traceback.print_exc()