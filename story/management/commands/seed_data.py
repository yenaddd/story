# story/management/commands/seed_data.py
from django.core.management.base import BaseCommand
from story.models import Genre, Cliche
from story.cliche_data import CLICHE_LIST  # 1단계에서 만든 데이터 파일 import

class Command(BaseCommand):
    help = '준비된 대규모 장르 및 클리셰 데이터 적재'

    def handle(self, *args, **options):
        self.stdout.write("데이터 적재를 시작합니다...")
        
        count = 0
        for item in CLICHE_LIST:
            # 1. 장르 생성 (이미 있으면 가져오기)
            genre_obj, created = Genre.objects.get_or_create(name=item["genre"])
            
            # 2. 클리셰 생성 (이미 있으면 업데이트)
            Cliche.objects.update_or_create(
                genre=genre_obj,
                title=item["title"],
                defaults={
                    "summary": item["summary"],
                    "structure_guide": item["structure_guide"],
                    "example_work_summary": item["example_work_summary"]
                }
            )
            count += 1

        self.stdout.write(self.style.SUCCESS(f'총 {count}개의 클리셰 데이터 시딩 완료!'))