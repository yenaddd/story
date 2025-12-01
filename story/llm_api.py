# story/llm_api.py
import os
import json
from openai import OpenAI  # DeepSeek는 OpenAI Compatible API 사용
from django.conf import settings
from .models import Genre, Cliche, Story, CharacterState, StoryNode, NodeChoice

# DeepSeek API 설정 (환경변수 또는 하드코딩)
# DeepSeek V3 API Base URL은 보통 https://api.deepseek.com 입니다.
# 모델명은 사용 가능한 버전에 맞춰 설정하세요 (예: "deepseek-chat" or "deepseek-v3")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-your-key-here")
BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-chat"  # 연우님이 사용하실 V3.1 모델명으로 변경 가능

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=BASE_URL)

def call_llm(system_prompt, user_prompt, json_format=False):
    """DeepSeek API 호출 래퍼"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    try:
        response_format = {"type": "json_object"} if json_format else None
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            response_format=response_format,
            temperature=0.7,
            max_tokens=4000 
        )
        content = response.choices[0].message.content
        if json_format:
            return json.loads(content)
        return content
    except Exception as e:
        print(f"LLM Error: {e}")
        # 오류 시 빈 값 반환 혹은 재시도 로직 필요
        return {} if json_format else ""

# --- 14단계 프로세스 구현 ---

def create_story_pipeline(user_world_setting):
    """전체 스토리 생성 파이프라인"""
    
    # 1. (입력됨) user_world_setting
    
    # 2. 클리셰 매칭
    matched_cliche = _match_cliche(user_world_setting)
    
    story = Story.objects.create(
        user_world_setting=user_world_setting,
        main_cliche=matched_cliche
    )

    # 3. 초기 시놉시스 생성 (2000자 내외)
    synopsis = _generate_initial_synopsis(story, matched_cliche)
    story.synopsis = synopsis
    story.save()

    # 4. 인물 내면 변화 DB 저장 (초기)
    _update_character_db(story, synopsis, is_initial=True)

    # 5 & 6. 챕터별 줄거리 생성 및 노드화 (선형 구조)
    nodes = _generate_linear_nodes(story, synopsis)

    # 7. 노드 간 연결 및 선택지 생성 (Illusion Choice)
    _connect_nodes_with_choices(nodes)

    # 8. 클리셰 비틀기(Twist) 지점 찾기
    twist_node_index = _find_twist_point(nodes)
    twist_node = nodes[twist_node_index]
    
    story.twist_point_node_id = twist_node.id
    story.save()

    # 9. 비틀기 이후 새로운 시놉시스 생성 (하이브리드 장르)
    new_synopsis, new_cliche = _generate_twisted_synopsis(story, nodes[:twist_node_index+1])
    story.twisted_synopsis = new_synopsis
    story.twist_cliche = new_cliche
    story.save()

    # 10. 인물 내면 변화 DB 업데이트 (새 시놉시스 반영)
    _update_character_db(story, new_synopsis, is_initial=False)

    # 11 & 11-2. 비틀린 이후의 새로운 노드 생성
    new_branch_nodes = _generate_linear_nodes(story, new_synopsis, start_phase_index=twist_node_index+1)

    # 12. 비틀기 지점(Twist Node)에서의 분기 처리 (기존 루트 vs 새 루트)
    # 기존 노드의 '다음 노드'는 이미 7번 과정에서 생성된 nodes[twist_node_index+1] 입니다.
    # 여기에 새로운 루트(new_branch_nodes[0])로 가는 선택지를 추가합니다.
    _create_twist_choices(twist_node, nodes[twist_node_index+1], new_branch_nodes[0])

    # 13. 새로 생성된 브랜치 내부 연결 (선형)
    _connect_nodes_with_choices(new_branch_nodes)

    return story.id

# --- 내부 로직 함수들 ---

def _match_cliche(setting):
    # DB에 있는 모든 클리셰 정보를 가져와 프롬프트에 제공
    all_cliches = Cliche.objects.select_related('genre').all()
    cliche_info = "\n".join([f"ID {c.id}: [{c.genre.name}] {c.title} - {c.summary}" for c in all_cliches])
    
    sys_prompt = "당신은 스토리 분석가입니다. 사용자 설정에 가장 적합한 클리셰 ID를 JSON으로 반환하세요."
    user_prompt = f"사용자 설정: {setting}\n\n보유 클리셰 목록:\n{cliche_info}\n\n출력형식: {{'cliche_id': 숫자}}"
    
    res = call_llm(sys_prompt, user_prompt, json_format=True)
    return Cliche.objects.get(id=res['cliche_id'])

def _generate_initial_synopsis(story, cliche):
    sys_prompt = (
        "당신은 소설가입니다. 사용자 설정과 클리셰를 결합해 2000자 내외의 시놉시스를 작성하세요. "
        "사건의 원인과 해결은 사용자 설정을 따르고, 감정선과 갈등 단계는 클리셰를 벤치마킹하세요."
    )
    user_prompt = (
        f"설정: {story.user_world_setting}\n"
        f"클리셰: {cliche.title}\n"
        f"클리셰 가이드: {cliche.structure_guide}\n"
        f"참고 작품 줄거리: {cliche.example_work_summary}"
    )
    return call_llm(sys_prompt, user_prompt)

def _update_character_db(story, text_content, is_initial=False):
    # 4번, 10번: 내면 상태 추출
    sys_prompt = (
        "텍스트를 분석하여 등장인물들의 '내면 상태'를 JSON으로 추출하세요. "
        "감정, 신뢰도, 사상, 육체적 상태 등의 변화를 구체적으로 기록해야 합니다."
    )
    res = call_llm(sys_prompt, f"분석할 텍스트:\n{text_content}", json_format=True)
    
    # DB 저장 (단순화를 위해 덮어쓰거나 누적)
    for name, state in res.items():
        char_state, created = CharacterState.objects.get_or_create(story=story, character_name=name)
        # 기존 상태와 병합 로직 (여기선 단순 갱신)
        char_state.state_data = state
        char_state.save()

def _generate_linear_nodes(story, synopsis, start_phase_index=0):
    # 5, 6번: 챕터별 2개 노드 생성 (총 4챕터 * 2 = 8노드 구조라고 가정)
    # start_phase_index가 있다면 그 지점부터 생성 (Twist 이후 생성 시)
    
    phases = ["발단", "전개", "절정", "결말"]
    nodes = []
    
    # 현재 저장된 캐릭터 상태 가져오기
    char_states = CharacterState.objects.filter(story=story)
    states_str = json.dumps({c.character_name: c.state_data for c in char_states}, ensure_ascii=False)

    current_prev_node = None
    
    # 전체 시놉시스를 4단계로 나누는 로직 (LLM에게 요청)
    sys_prompt = "시놉시스를 발단, 전개, 절정, 결말 4단계로 나누고, 각 단계를 다시 2개의 상세 장면(Scene)으로 나누어 총 8개의 줄거리(각 2000자)를 JSON 리스트로 만드세요."
    user_prompt = f"시놉시스: {synopsis}\n인물상태: {states_str}\n형식: {{'scenes': [text1, text2, ..., text8]}}"
    
    res = call_llm(sys_prompt, user_prompt, json_format=True)
    scenes = res.get('scenes', [])

    # 인덱스 조정 (부분 생성일 경우)
    target_scenes = scenes[start_phase_index * 2:] 
    
    for i, content in enumerate(target_scenes):
        phase_idx = (start_phase_index * 2 + i) // 2
        if phase_idx >= 4: break
        
        node = StoryNode.objects.create(
            story=story,
            chapter_phase=phases[phase_idx],
            content=content,
            prev_node=current_prev_node
        )
        nodes.append(node)
        current_prev_node = node
        
    return nodes

def _connect_nodes_with_choices(nodes):
    # 7, 13번: 선형 구조 연결 (선택지는 다르지만 결과 노드는 같음)
    # nodes 리스트는 순서대로 연결되어 있음 [Node1, Node2, Node3 ...]
    
    for i in range(len(nodes) - 1):
        curr = nodes[i]
        next_n = nodes[i+1]
        
        sys_prompt = (
            "현재 장면과 다음 장면을 잇는 2개의 선택지를 만드세요. "
            "각 선택지에 대해 '직후 행동/결과(result_text)'를 한 문장으로 생성하세요. "
            "어떤 선택을 하든 다음 장면으로 자연스럽게 이어져야 합니다."
        )
        user_prompt = f"현재 장면: {curr.content[-500:]}\n다음 장면: {next_n.content[:500]}\n형식: {{'choices': [{{'text': '...', 'result': '...'}}, {{'text': '...', 'result': '...'}}]}}"
        
        res = call_llm(sys_prompt, user_prompt, json_format=True)
        
        for item in res.get('choices', []):
            NodeChoice.objects.create(
                current_node=curr,
                choice_text=item['text'],
                result_text=item['result'],
                next_node=next_n
            )

def _find_twist_point(nodes):
    # 8번: 변주 지점 찾기 (랜덤 혹은 LLM 판단)
    # 마지막 노드 제외하고 중간(전개~절정 사이)에서 하나 선택
    sys_prompt = "전체 스토리 흐름에서 장르를 비틀어 새로운 전개를 시작하기 가장 좋은 지점(노드 인덱스)을 하나 찾으세요. (0부터 시작)"
    summary_list = [n.content[:200] for n in nodes]
    user_prompt = f"장면들: {summary_list}\n형식: {{'index': 3}}"
    
    res = call_llm(sys_prompt, user_prompt, json_format=True)
    idx = res.get('index', 3)
    # 안전장치
    if idx >= len(nodes) - 1 or idx < 1: idx = 3 
    return idx

def _generate_twisted_synopsis(story, accumulated_nodes):
    # 9번: 트위스트 시놉시스 생성
    current_story = "\n".join([n.content for n in accumulated_nodes])
    current_phase = accumulated_nodes[-1].chapter_phase
    
    # 어울리지 않는 다른 장르/클리셰 랜덤 매칭 (여기선 LLM에게 추천받음)
    sys_prompt = "현재까지의 이야기와 전혀 다른 반전 매력을 줄 수 있는 '새로운 장르 클리셰'를 추천하고, 이를 결합해 이후의 시놉시스를 새로 쓰세요."
    user_prompt = f"현재까지 이야기: {current_story}\n현재 단계: {current_phase}"
    
    # 시놉시스 텍스트 생성 (JSON 아님)
    new_synopsis = call_llm(sys_prompt, user_prompt)
    
    # (약식) 새 클리셰 정보는 DB에 없어도 되지만 형식상 하나 매칭하거나 생성
    # 여기선 None으로 두거나 더미 로직 사용
    return new_synopsis, None 

def _create_twist_choices(node, original_next, new_next):
    # 12번: 4개의 선택지 (2개 -> 기존, 2개 -> 신규)
    # 이미 _connect_nodes_with_choices로 기존 2개는 생성되어 있음.
    # 추가로 신규 루트로 가는 2개를 생성.
    
    sys_prompt = (
        "장르가 급격히 바뀌는 분기점입니다. "
        "기존 스토리와 다른, 새로운 전개(예: 뱀파이어 등장 등)로 이어지는 선택지 2개를 만드세요."
    )
    user_prompt = f"현재 장면: {node.content[-500:]}\n새로운 전개 시작: {new_next.content[:500]}"
    
    res = call_llm(sys_prompt, user_prompt, json_format=True)
    
    for item in res.get('choices', []):
        NodeChoice.objects.create(
            current_node=node,
            choice_text=item['text'],
            result_text=item['result'],
            next_node=new_next,
            is_twist_path=True # 이것이 변주 경로임
        )