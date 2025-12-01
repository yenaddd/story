# story/views.py
from django.shortcuts import render, get_object_or_404
from rest_framework.views import APIView
from rest_framework.response import Response
from .models import Story, StoryNode, NodeChoice
from .llm_api import create_story_pipeline

# 1. 생성 페이지
def story_creator_view(request):
    return render(request, 'story/story_creator.html')

# 2. 플레이 페이지
def story_play_view(request):
    return render(request, 'story/story_play.html')

# 3. 스토리 생성 API (POST)
class StoryResetAPIView(APIView):
    def post(self, request):
        world_setting = request.data.get('world_setting')
        if not world_setting:
            return Response({'error': '설정을 입력해주세요.'}, status=400)
            
        try:
            # 14단계 파이프라인 실행
            story_id = create_story_pipeline(world_setting)
            
            # 생성된 스토리의 첫 번째 노드 찾기
            first_node = StoryNode.objects.filter(story_id=story_id).order_by('id').first()
            
            return Response({'root_node_id': first_node.id})
        except Exception as e:
            import traceback
            traceback.print_exc()
            return Response({'error': str(e)}, status=500)

# 4. 노드 상세 조회 API (GET) - 핵심 로직: 이전 선택지의 결과를 결합
class StoryNodeDetail(APIView):
    def get(self, request, node_id):
        node = get_object_or_404(StoryNode, id=node_id)
        
        # 이전 선택지 ID를 쿼리 파라미터로 받음 (?prev_choice=123)
        prev_choice_id = request.query_params.get('prev_choice')
        prefix_text = ""
        
        if prev_choice_id:
            try:
                choice = NodeChoice.objects.get(id=prev_choice_id)
                # 7번 요구사항: 선택지에 대한 직후 행동 묘사를 앞에 붙임
                prefix_text = f"[{choice.result_text}]\n\n"
            except:
                pass

        choices = node.choices.all()
        choices_data = [
            {
                'id': c.id, 
                'text': c.choice_text, 
                'next_node_id': c.next_node.id if c.next_node else None,
                'is_twist': c.is_twist_path
            } 
            for c in choices
        ]

        data = {
            'id': node.id,
            'story_text': prefix_text + node.content, # 결과 텍스트 결합
            'chapter': node.chapter_phase,
            'choices': choices_data
        }
        return Response(data)