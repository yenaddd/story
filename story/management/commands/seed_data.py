# story/management/commands/seed_data.py
from django.core.management.base import BaseCommand
from story.models import Genre, Cliche
# 위에서 만든 MAP과 설명을 가져옵니다.
from story.cliche_data import GENRE_DESCRIPTIONS, CLICHE_DATA_MAP 

class Command(BaseCommand):
    help = '장르별로 구분된 클리셰 데이터를 DB에 적재합니다.'

    def handle(self, *args, **options):
        self.stdout.write("데이터 적재를 시작합니다...")
        
        total_cliche_count = 0
        
        # CLICHE_DATA_MAP을 순회하며 장르별로 처리
        for genre_name, cliche_list in CLICHE_DATA_MAP.items():
            
            # 1. 장르 생성 및 설명 업데이트
            genre_desc = GENRE_DESCRIPTIONS.get(genre_name, "")
            genre_obj, created = Genre.objects.get_or_create(name=genre_name)
            
            # 설명이 업데이트되었을 수 있으므로 저장
            genre_obj.description = genre_desc
            genre_obj.save()
            
            self.stdout.write(f"  [{genre_name}] 처리 중... ({len(cliche_list)}개)")

            # 2. 해당 장르의 클리셰들 생성
            for item in cliche_list:
                Cliche.objects.update_or_create(
                    genre=genre_obj,
                    title=item["title"],
                    defaults={
                        "summary": item["summary"],
                        "structure_guide": item["structure_guide"],
                        "example_work_summary": item["example_work_summary"]
                    }
                )
                total_cliche_count += 1

        self.stdout.write(self.style.SUCCESS(f'총 {total_cliche_count}개의 클리셰 및 장르 설명 적재 완료!'))