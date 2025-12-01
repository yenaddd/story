from django.shortcuts import render, get_object_or_404, redirect
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import StoryNode, NodeChoice
from .serializers import StoryNodeSerializer
from django.http import Http404
from django.db import transaction,connection
# [수정] GLOBAL_STORY_CONFIG와 '전체흐름정의자' AI를 llm_api로부터 import합니다.
from .llm_api import (
    generate_full_story_tree, 
    set_global_config, 
    call_llm_flow_definer, # (신규) 1번 요청사항
    GLOBAL_STORY_CONFIG
)
from django.db.models import ObjectDoesNotExist # ObjectDoesNotExist 대신 StoryNode.DoesNotExist를 주로 사용합니다.
import json # json 모듈 추가

def story_home_view(request):
    """
    스토리 존재 여부(Node 1)를 확인하고, 그 상태를 템플릿으로 전달하여
    '이어서 플레이' 버튼 활성화 여부를 결정합니다.
    """
    story_exists = False
    try:
        # 루트 노드(ID=1)가 있는지 확인
        StoryNode.objects.get(id=1)
        story_exists = True
    except StoryNode.DoesNotExist:
        story_exists = False
        
    context = {
        'story_exists': story_exists
    }
    # 스토리 존재 여부 데이터를 context에 담아 홈 템플릿으로 전달합니다.
    return render(request, 'story/story_home.html', context) # 🚨 새로운 템플릿

    
# 1. [템플릿 뷰]: 스토리 생성 입력 페이지 (URL: /story/)
def story_creator_view(request):
    """사용자 스토리 생성 시작 페이지를 제공"""
    return render(request, 'story/story_creator.html')

# 2. [템플릿 뷰]: 스토리 체험 페이지 (URL: /story/play/)
def story_play_view(request):
    """사용자 스토리 리더 페이지를 제공"""
    return render(request, 'story/story_play.html')


# 3. [API 뷰]: 특정 노드 ID의 상세 정보를 JSON으로 반환 (생성 요청 포함 -> 생성 요청 제거)
class StoryNodeDetail(APIView):
    # @transaction.atomic # <- GET 요청은 트랜잭션이 불필요합니다.
    def get(self, request, node_id, format=None):
        try:
            node = StoryNode.objects.get(id=node_id)
            serializer = StoryNodeSerializer(node)
            return Response(serializer.data)
        except StoryNode.DoesNotExist:
            # 지정된 노드가 DB에 없을 경우 404를 반환
            raise Http404(f"Node {node_id} not found.")

# 4. [API 뷰]: 스토리 전체 삭제, 설정 저장 및 새 루트 노드 생성 API (POST 요청)
class StoryResetAPIView(APIView):
    """설정을 받아 기존 스토리를 삭제하고, 새로운 스토리 전체 트리를 생성합니다."""
    # [오류 수정] DB 락(Lock) 오류를 방지하기 위해 뷰 레벨의 @transaction.atomic을 제거합니다.
    # @transaction.atomic 
    def post(self, request, format=None):
        try:
            data = request.data
            
            # --- 1. 사용자 입력 설정 파싱 및 유효성 검사 ---
            world_setting = data.get('world_setting', '')
            arc_type = data.get('arc_type', 'Positive Arc')
            branches_str = data.get('branches', '[2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3]') # 기본값 제공 (총 깊이 12)

            # --- [수정된 부분 시작] ---
            # llm_api.py에 정의된 "현재의 기본 워커 수"를 읽어옵니다. (예: 8)
            default_workers = GLOBAL_STORY_CONFIG.get("MAX_CONCURRENT_WORKERS", 4) # 혹시 키가 없으면 4로 최후 방어

            try:
                # API 요청에 'max_workers'가 있으면 그 값을 쓰고, 없으면 llm_api.py의 'default_workers' 값을 사용합니다.
                max_workers = int(data.get('max_workers', default_workers)) 
                if max_workers < 1:
                    max_workers = 1
                if max_workers > 32: # 과도한 스레드를 방지하기 위한 상한선 (선택적)
                    print("최대 워커 수를 32로 제한합니다.")
                    max_workers = 32
            except ValueError:
                max_workers = default_workers # 입력값이 숫자가 아니면 llm_api.py의 기본값 사용
            # --- [수정된 부분 끝] ---

            try:
                # 문자열 리스트를 Python 리스트로 변환
                branches = json.loads(branches_str)
                if not isinstance(branches, list) or not all(isinstance(x, int) and x in [2, 3] for x in branches):
                     raise ValueError("분기 설정은 [2, 3]으로 구성된 정수 리스트여야 합니다.")
                if not branches:
                     raise ValueError("분기 설정은 비어 있을 수 없습니다.")
            except (json.JSONDecodeError, ValueError) as e:
                return Response({'error': f'분기 설정 파싱 오류: {e}'}, status=status.HTTP_400_BAD_REQUEST)

            if not world_setting:
                return Response({'error': '세계관 설정은 필수입니다.'}, status=status.HTTP_400_BAD_REQUEST)

            # --- 2. LLM 전역 설정 저장 ---
            # max_workers 값 (요청값 또는 llm_api.py의 기본값)을 llm_api로 전달
            set_global_config(world_setting, arc_type, branches, max_workers)
            
            # --- [신규] 1번 요청사항: 전체 스토리 줄거리 생성 ---
            # set_global_config가 호출된 직후, '전체흐름정의자'를 호출
            print("--- '전체흐름정의자' AI 호출 시작 ---")
            overall_plot = call_llm_flow_definer(world_setting, arc_type)
            # 생성된 전체 줄거리를 GLOBAL_STORY_CONFIG에 저장
            GLOBAL_STORY_CONFIG["OVERALL_STORY_PLOT"] = overall_plot
            print(f"--- '전체 스토리 줄거리' 생성 및 저장 완료 --- (길이: {len(overall_plot)})")
            if "실패" in overall_plot or "오류" in overall_plot:
                return Response({'error': f'전체 줄거리 생성 실패: {overall_plot}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            # --- [신규 로직 끝] ---

            
            # --- 3. 기존 데이터 삭제 ---
            NodeChoice.objects.all().delete()
            StoryNode.objects.all().delete()

            # [요청사항 수정] 다음 StoryNode와 NodeChoice가 ID=1로 생성되도록
            # SQLite 내부의 카운터를 0으로 초기화합니다.
            with connection.cursor() as cursor:
                # 'story_storynode' 테이블의 ID 카운터를 0으로 설정
                cursor.execute("UPDATE sqlite_sequence SET seq = 0 WHERE name='story_storynode'")
                # 'story_nodechoice' 테이블의 ID 카운터를 0으로 설정 (<<< *** 요청사항 수정된 부분 ***)
                cursor.execute("UPDATE sqlite_sequence SET seq = 0 WHERE name='story_nodechoice'")


            # --- 4. 루트 노드 (ID=1, Depth=0) 생성 시작 (요청 사항 반영) ---
            # 이제 generate_full_story_tree는 내부적으로 GLOBAL_STORY_CONFIG에 저장된
            # 'OVERALL_STORY_PLOT'를 참조하여 작업을 수행합니다.
            print(f"--- 전체 스토리 트리 생성 시작 (N={max_workers}) ---") 
            root_node = generate_full_story_tree(
                parent_node_id=0, 
                choice_text='이야기 시작'
            )
            print(f"--- 전체 스토리 트리 생성 완료 (Root: {root_node.id}) ---")
            
            # 5. 생성된 루트 노드의 ID 반환
            return Response({
                'message': 'New story configuration set and full story tree generated.', 
                'root_node_id': root_node.id
            }, status=status.HTTP_201_CREATED)
        
        except Exception as e:
            print(f"Story Reset/Generation Error: {e}")
            return Response({'error': f'스토리 생성 중 오류 발생: {e}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# 5. [API 뷰]: 스토리 상태 확인 (루트 노드 존재 여부)
class StoryStatusAPIView(APIView):
    """루트 노드(ID=1)의 존재 여부를 반환하여 기존 스토리 진행 가능 여부를 확인합니다."""
    def get(self, request, format=None):
        try:
            StoryNode.objects.get(id=1)
            # 루트 노드가 존재하면 진행 가능
            return Response({'exists': True}, status=status.HTTP_200_OK)
        except StoryNode.DoesNotExist:
            # 루트 노드가 존재하지 않으면 새로 생성 필요
            return Response({'exists': False}, status=status.HTTP_200_OK)

def story_plot_view(request):
    """
    생성된 전체 스토리 줄거리(Flow Definer 결과물)를 보여줍니다.
    """
    overall_plot = GLOBAL_STORY_CONFIG.get("OVERALL_STORY_PLOT", "생성된 전체 줄거리가 없습니다.")
    world_setting = GLOBAL_STORY_CONFIG.get("WORLD_SETTING", "설정 없음")
    
    context = {
        'overall_plot': overall_plot,
        'world_setting': world_setting
    }
    return render(request, 'story/story_plot.html', context)