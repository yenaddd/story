from django.shortcuts import render, get_object_or_404
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import Story, StoryNode, NodeChoice
from .llm_api import create_story_pipeline

# 1. [뷰] 홈 페이지: 스토리 존재 여부 확인 후 홈 화면 렌더링
def story_home_view(request):
    """
    스토리가 DB에 존재하는지 확인하고, '이어서 하기' 버튼 활성화 여부를 결정합니다.
    """
    story_exists = False
    # 가장 최근에 생성된 스토리와 노드가 있는지 확인
    if Story.objects.exists():
        last_story = Story.objects.last()
        if StoryNode.objects.filter(story=last_story).exists():
            story_exists = True

    context = {
        'story_exists': story_exists
    }
    return render(request, 'story/story_home.html', context)


# 2. [뷰] 스토리 생성 페이지
def story_creator_view(request):
    return render(request, 'story/story_creator.html')


# 3. [뷰] 스토리 체험 페이지
def story_play_view(request):
    return render(request, 'story/story_play.html')


# 4. [뷰] 전체 줄거리 보기 페이지
def story_plot_view(request):
    """
    현재 진행 중인 스토리의 시놉시스(줄거리)를 보여줍니다.
    """
    story = Story.objects.last()
    if not story:
        context = {'overall_plot': None, 'world_setting': None}
    else:
        # 반전(Twist)이 일어난 이후라면 변경된 시놉시스를 보여줍니다.
        plot = story.twisted_synopsis if story.twisted_synopsis else story.synopsis
        context = {
            'overall_plot': plot,
            'world_setting': story.user_world_setting
        }
    return render(request, 'story/story_plot.html', context)


# 5. [API] 스토리 상태 확인 (프론트엔드 체크용)
class StoryStatusAPIView(APIView):
    def get(self, request):
        # 스토리와 노드가 하나라도 있으면 존재하는 것으로 간주
        exists = Story.objects.exists() and StoryNode.objects.exists()
        return Response({'exists': exists})


# 6. [API] 스토리 생성 및 리셋
class StoryResetAPIView(APIView):
    def post(self, request):
        world_setting = request.data.get('world_setting')
        if not world_setting:
            return Response({'error': '설정을 입력해주세요.'}, status=400)
            
        try:
            # (선택 사항) DB 초기화가 필요하다면 아래 주석 해제
            # Story.objects.all().delete() 
            
            # 14단계 파이프라인 실행 (DeepSeek LLM 호출)
            story_id = create_story_pipeline(world_setting)
            
            # 생성된 스토리의 첫 번째 노드(루트 노드) 찾기
            first_node = StoryNode.objects.filter(story_id=story_id).order_by('id').first()
            
            if not first_node:
                return Response({'error': '스토리가 생성되었으나 루트 노드를 찾을 수 없습니다.'}, status=500)

            return Response({'root_node_id': first_node.id})
        except Exception as e:
            import traceback
            traceback.print_exc()
            return Response({'error': str(e)}, status=500)


# 7. [API] 노드 상세 조회 (선택지 결과 결합 로직 포함)
class StoryNodeDetail(APIView):
    def get(self, request, node_id):
        try:
            node = StoryNode.objects.get(id=node_id)
        except StoryNode.DoesNotExist:
             return Response({'error': 'Node not found'}, status=404)
        
        # 이전 선택지 ID를 쿼리 파라미터로 받음 (?prev_choice=123)
        # 이를 통해 "선택에 따른 직후 행동" 텍스트를 현재 노드 맨 앞에 붙여줌 (Linear Illusion)
        prev_choice_id = request.query_params.get('prev_choice')
        prefix_text = ""
        
        if prev_choice_id:
            try:
                choice = NodeChoice.objects.get(id=prev_choice_id)
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
        
        # 템플릿과의 호환성을 위해 추가 메타데이터 전달
        char_states = node.story.character_states.all()
        relationships = {}
        for cs in char_states:
             # JSON 객체를 문자열 등으로 변환하여 표시하거나 그대로 전달
             relationships[cs.character_name] = cs.state_data

        data = {
            'id': node.id,
            'story_text': prefix_text + node.content, 
            'chapter': node.chapter_phase,
            'choices': choices_data,
            'critique_score': 0,  # 기존 필드 호환용 더미 데이터
            'depth': 0,           # 기존 필드 호환용 더미 데이터
            'freytag_stage': node.chapter_phase,
            'protagonist_state': "스토리 진행 중...", 
            'current_relationships': relationships
        }
        
        return Response(data)