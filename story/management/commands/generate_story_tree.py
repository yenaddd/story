from django.core.management.base import BaseCommand, CommandError
from story.models import StoryNode, NodeChoice
from story.llm_api import (
    set_global_config, 
    generate_full_story_tree, 
    call_llm_flow_definer,
    GLOBAL_STORY_CONFIG
)
import json
from django.db import connection

class Command(BaseCommand):
    help = '맞춤 설정 가능한 분기 구조로 스토리 트리를 생성합니다 (v5 병렬 처리 반영).'

    def add_arguments(self, parser):
        parser.add_argument('--world', type=str, required=True, help='세계관/인물 설정 요약')
        parser.add_argument('--arc', type=str, required=True, help='캐릭터 아크 이론')
        parser.add_argument('--branches', type=str, required=True, help='분기 설정 리스트 예: [2,3,2]')
        parser.add_argument('--workers', type=int, default=4, help='병렬 워커 수 (기본: 4)')
        # dbname은 settings.py의 DATABASES 설정에 의존하므로, 
        # 간단한 테스트를 위해 default DB를 사용하는 것으로 가정하거나 필요시 로직 추가

    def handle(self, *args, **options):
        world_setting = options['world']
        arc_type = options['arc']
        max_workers = options['workers']
        
        try:
            branch_str = options['branches'].replace(' ', '')
            branch_config = json.loads(branch_str)
        except Exception as e:
            raise CommandError(f"분기 설정 파싱 오류: {e}")

        # 1. 전역 설정 초기화
        set_global_config(world_setting, arc_type, branch_config, max_workers)
        
        self.stdout.write(self.style.WARNING('--- 스토리 생성 프로세스 시작 ---'))

        # 2. 전체 흐름 정의 (Flow Definer) 호출
        self.stdout.write("전체 줄거리(Flow Definer) 생성 중...")
        overall_plot = call_llm_flow_definer(world_setting, arc_type)
        GLOBAL_STORY_CONFIG["OVERALL_STORY_PLOT"] = overall_plot
        self.stdout.write(self.style.SUCCESS(f"전체 줄거리 생성 완료 (길이: {len(overall_plot)})"))

        # 3. 기존 데이터 삭제 (초기화)
        NodeChoice.objects.all().delete()
        StoryNode.objects.all().delete()
        
        # SQLite 시퀀스 초기화 (ID를 1부터 다시 시작하게 함)
        with connection.cursor() as cursor:
            cursor.execute("UPDATE sqlite_sequence SET seq = 0 WHERE name='story_storynode'")
            cursor.execute("UPDATE sqlite_sequence SET seq = 0 WHERE name='story_nodechoice'")

        # 4. 핵심 로직 호출 (llm_api의 병렬 함수 재사용)
        self.stdout.write(f"트리 생성 시작 (Max Workers: {max_workers})...")
        
        try:
            root_node = generate_full_story_tree(
                parent_node_id=0, 
                choice_text='이야기 시작'
            )
            self.stdout.write(self.style.SUCCESS(f'스토리 트리 생성 완료! Root ID: {root_node.id}'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"트리 생성 중 오류 발생: {e}"))