# story/management/commands/seed_data.py
from django.core.management.base import BaseCommand
from story.models import Genre, Cliche

class Command(BaseCommand):
    help = '초기 장르 및 클리셰 데이터 적재'

    def handle(self, *args, **options):
        # 1. 로맨스
        romance, _ = Genre.objects.get_or_create(name="로맨스")
        Cliche.objects.get_or_create(
            genre=romance,
            title="계약 관계 (가짜 연애/결혼)",
            defaults={
                "summary": "비즈니스/상황적 필요로 계약을 맺고 동거 혹은 연인 행세를 하다 진정한 사랑에 빠짐",
                "structure_guide": {
                    "1": "각자의 목적(상속 등)으로 계약 체결",
                    "2": "사소한 충돌과 오해 속에서 의외의 모습 발견",
                    "3": "계약 만료 임박 및 제3자의 방해, 위기",
                    "4": "서로의 진심 확인 및 해피엔딩"
                },
                "example_work_summary": "드라마 '사내맞선' 등 참조... (생략)"
            }
        )
        
        # 2. 판타지
        fantasy, _ = Genre.objects.get_or_create(name="판타지")
        Cliche.objects.get_or_create(
            genre=fantasy,
            title="회귀한 용사의 복수",
            defaults={
                "summary": "마왕을 물리쳤으나 동료에게 배신당해 죽은 용사가 과거로 돌아와 복수함",
                "structure_guide": {}, # 생략
                "example_work_summary": "..."
            }
        )
        
        self.stdout.write(self.style.SUCCESS('데이터 시딩 완료'))