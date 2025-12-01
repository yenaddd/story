from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from .models import Story, StoryNode, NodeChoice
from .llm_api import create_story_pipeline

# 1. [뷰] 홈 페이지
def story_home_view(request):
    story_exists = False
    if Story.objects.exists() and StoryNode.objects.exists():
        story_exists = True
    return render(request, 'story/story_home.html', {'story_exists': story_exists})

# 2. [뷰] 스토리 생성 페이지
def story_creator_view(request):
    return render(request, 'story/story_creator.html')

# 3. [뷰] 스토리 체험 페이지
def story_play_view(request):
    return render(request, 'story/story_play.html')

# 4. [뷰] 전체 줄거리 보기
def story_plot_view(request):
    story = Story.objects.last()
    if not story:
        context = {'overall_plot': None, 'world_setting': None}
    else:
        # 두 가지 시놉시스를 모두 보여줌
        context = {
            'overall_plot': story.synopsis,
            'twisted_plot': story.twisted_synopsis,
            'world_setting': story.user_world_setting
        }
    return render(request, 'story/story_plot.html', context)

# 5. [API] 스토리 상태 확인
class StoryStatusAPIView(APIView):
    def get(self, request):
        exists = Story.objects.exists() and StoryNode.objects.exists()
        return Response({'exists': exists})

# ... (상단 import 및 기존 views 코드는 유지)

# 6. [API] 스토리 생성 (14단계 파이프라인 실행)
class StoryResetAPIView(APIView):
    def post(self, request):
        world_setting = request.data.get('world_setting')
        
        # [수정] 프론트엔드에서 더 이상 arc_type, branches를 보내지 않으므로 받지 않음
        
        if not world_setting:
            return Response({'error': '설정을 입력해주세요.'}, status=400)
            
        try:
            # 기존 데이터 삭제 (새로운 게임 시작)
            Story.objects.all().delete()
            StoryNode.objects.all().delete()
            NodeChoice.objects.all().delete()
            
            # 파이프라인 실행 (세계관 설정만 전달)
            story_id = create_story_pipeline(world_setting)
            
            # 루트 노드 찾기
            first_node = StoryNode.objects.filter(story_id=story_id).order_by('id').first()
            if not first_node:
                return Response({'error': '스토리 생성 실패: 노드가 없습니다.'}, status=500)

            return Response({'root_node_id': first_node.id})
        except Exception as e:
            import traceback
            traceback.print_exc()
            return Response({'error': str(e)}, status=500)

# 7. [API] 노드 상세 조회 (Illusion of Choice 구현 핵심)
class StoryNodeDetail(APIView):
    def get(self, request, node_id):
        try:
            node = StoryNode.objects.get(id=node_id)
        except StoryNode.DoesNotExist:
             return Response({'error': 'Node not found'}, status=404)
        
        # 7번 요구사항: 선택지에 따른 결과 텍스트를 현재 노드 줄거리 앞에 붙임
        prev_choice_id = request.query_params.get('prev_choice')
        prefix_text = ""
        
        if prev_choice_id:
            try:
                choice = NodeChoice.objects.get(id=prev_choice_id)
                # 예: [남자를 발로 차서 엉덩이에 멍이 들었다.]
                prefix_text = f"[{choice.result_text}]\n\n"
            except NodeChoice.DoesNotExist:
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
        
        # 인물 상태 데이터도 함께 전송 (프론트엔드 디버깅용)
        # 현재 스토리의 최신 상태를 보냄
        # (실제로는 노드별로 시점 상태를 저장하는 게 더 정확하지만, 여기선 최신값 전송)
        char_states = node.story.character_states.all()
        relationships = {}
        for cs in char_states:
             relationships[cs.character_name] = cs.state_data

        data = {
            'id': node.id,
            'story_text': prefix_text + node.content, # 결합된 텍스트 반환
            'chapter': node.chapter_phase,
            'choices': choices_data,
            'current_relationships': relationships
        }
        
        return Response(data)