# story/urls.py
from django.urls import path
from .views import StoryNodeDetail, story_play_view, story_creator_view, StoryResetAPIView, StoryStatusAPIView, story_home_view, story_plot_view

urlpatterns = [
    # 1. 홈 페이지 (상태 확인 및 분기)
    path('', story_home_view, name='story-home'),
    
    # 2. 스토리 생성 설정 페이지 (기존 creator view를 이 경로로 이동)
    path('creator/', story_creator_view, name='story-creator'),
    
    # 3. 스토리 체험 페이지 
    path('play/', story_play_view, name='story-play'),

    # [신규] 전체 줄거리 뷰어 페이지
    path('plot/', story_plot_view, name='story-plot'),
    
    # 4. 노드 데이터를 가져오거나 생성하는 API (기존 유지)
    path('api/node/<int:node_id>/', StoryNodeDetail.as_view(), name='node-detail'),

    # 5. 스토리 전체를 삭제하고 새 생성을 시작하는 API (기존 유지)
    path('api/reset/', StoryResetAPIView.as_view(), name='story-reset'),

    # 6. 스토리 상태 확인 API (기존 유지)
    path('api/status/', StoryStatusAPIView.as_view(), name='story-status'),
]