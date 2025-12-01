import os
import json
import random
from openai import OpenAI
from django.conf import settings
from .models import Genre, Cliche, Story, CharacterState, StoryNode, NodeChoice

# API 설정 (기존 설정 유지)
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
BASE_URL = "https://api.fireworks.ai/inference/v1"
MODEL_NAME = "accounts/fireworks/models/deepseek-v3"
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=BASE_URL)

def call_llm(system_prompt, user_prompt, json_format=False):
    """LLM 호출 래퍼"""
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
        return {} if json_format else ""

# ==========================================
# [메인 파이프라인] 14단계 프로세스
# ==========================================

def create_story_pipeline(user_world_setting):
    # 1. (입력됨) user_world_setting
    
    # 2. 클리셰 매칭
    matched_cliche = _match_cliche(user_world_setting)
    
    story = Story.objects.create(
        user_world_setting=user_world_setting,
        main_cliche=matched_cliche
    )

    # 3. 초기 시놉시스 생성 (발단-전개-절정-결말)
    synopsis = _generate_synopsis(story, matched_cliche, is_twist=False)
    story.synopsis = synopsis
    story.save()

    # 4. 인물 내면 변화 추출 및 DB 저장
    _analyze_and_save_character_state(story, synopsis, context="Initial Synopsis")

    # 5 & 6. 챕터별 줄거리 생성 및 노드화 (선형 구조)
    # 총 4챕터 * 2노드 = 8개 노드 생성
    original_nodes = _create_nodes_from_synopsis(story, synopsis, start_phase_idx=0)

    # 7. 선형 노드 간 연결 (Illusion of Choice)
    _connect_linear_nodes(original_nodes)

    # 8. 클리셰 비틀기(Twist) 지점 찾기
    # 마지막 노드를 제외한 중간 지점(주로 전개~절정 사이)
    twist_node_index = _find_twist_point_index(original_nodes)
    twist_node = original_nodes[twist_node_index]
    
    story.twist_point_node_id = twist_node.id
    story.save()

    # 9. 비틀기 이후 새로운 시놉시스 생성
    # twist_node 이전까지의 내용은 유지, 이후는 새로운 장르로 전환
    accumulated_story = "\n".join([n.content for n in original_nodes[:twist_node_index+1]])
    current_phase = twist_node.chapter_phase
    
    twist_cliche, twisted_synopsis = _generate_twisted_synopsis_data(story, accumulated_story, current_phase)
    
    story.twist_cliche = twist_cliche
    story.twisted_synopsis = twisted_synopsis
    story.save()

    # 10. 인물 내면 변화 DB 업데이트 (새 시놉시스 반영)
    _analyze_and_save_character_state(story, twisted_synopsis, context="Twisted Synopsis")

    # 11 & 11-2. 비틀린 이후의 새로운 노드 생성
    # twist_node_index 이후의 단계부터 새로 생성
    new_branch_nodes = _create_nodes_from_synopsis(
        story, 
        twisted_synopsis, 
        start_phase_idx=(twist_node_index // 2) + 1, 
        is_twist_branch=True
    )

    # 12. 비틀기 지점(Twist Node)에서의 분기 처리 (4개 선택지)
    # 기존 경로(original_nodes의 다음 노드) vs 새 경로(new_branch_nodes의 첫 노드)
    if twist_node_index + 1 < len(original_nodes) and new_branch_nodes:
        original_next_node = original_nodes[twist_node_index + 1]
        new_next_node = new_branch_nodes[0]
        
        # 기존 연결(7번에서 생성됨)은 삭제하고 새로 4개를 만듭니다.
        NodeChoice.objects.filter(current_node=twist_node).delete()
        
        _create_twist_branch_choices(twist_node, original_next_node, new_next_node)

    # 13. 새로 생성된 브랜치 내부 연결 (선형)
    _connect_linear_nodes(new_branch_nodes)

    # 14. 데이터 식별 번호는 DB ID로 자동 관리됨
    return story.id


# ==========================================
# [내부 로직 함수들]
# ==========================================

def _match_cliche(setting):
    """2. 사용자 설정에 가장 적합한 클리셰 매칭"""
    all_cliches = Cliche.objects.select_related('genre').all()
    if not all_cliches.exists():
        # 데이터가 없을 경우를 대비한 더미
        return None
        
    cliche_info = "\n".join([f"ID {c.id}: [{c.genre.name}] {c.title} - {c.summary}" for c in all_cliches])
    
    sys_prompt = "당신은 스토리 분석가입니다. 사용자 설정에 가장 적합한 클리셰 ID를 JSON으로 반환하세요."
    user_prompt = f"사용자 설정: {setting}\n\n보유 클리셰 목록:\n{cliche_info}\n\n출력형식: {{'cliche_id': 숫자}}"
    
    res = call_llm(sys_prompt, user_prompt, json_format=True)
    try:
        return Cliche.objects.get(id=res['cliche_id'])
    except:
        return all_cliches.first()

def _generate_synopsis(story, cliche, is_twist=False):
    """3. 시놉시스 생성 (2000자 내외, 4단계)"""
    role = "소설가"
    task = "전체 시놉시스를 작성하세요. 발단, 전개, 절정, 결말이 명확해야 합니다."
    
    if is_twist:
        # 9번 로직에서 호출됨
        pass 
    else:
        content = (
            f"사용자 설정: {story.user_world_setting}\n"
            f"매칭된 클리셰: {cliche.title}\n"
            f"클리셰 가이드: {cliche.structure_guide}\n"
            f"참고 작품 감정선: {cliche.example_work_summary}\n\n"
            "조건: 사건의 원인과 해결은 사용자 설정을 따르고, 감정선/갈등은 클리셰를 따르세요."
            "줄거리(Plot) 형태로 작성하며 총 2000자 내외여야 합니다."
        )
        return call_llm(f"당신은 {role}입니다. {task}", content)

def _analyze_and_save_character_state(story, text_content, context):
    """4 & 10. 인물 내면 변화 추출 및 DB 저장"""
    sys_prompt = (
        "텍스트를 정밀 분석하여 각 등장인물의 '내면 상태'를 JSON으로 추출하세요. "
        "사건, 행동, 결정이 인물에게 준 감정적, 정서적, 육체적, 사상적, 신뢰적 변화를 기록해야 합니다."
    )
    user_prompt = f"분석할 텍스트:\n{text_content}\n\n출력 형식: {{ '인물이름': {{ 'emotion': '...', 'physical': '...', 'trust': '...' }} }}"
    
    res = call_llm(sys_prompt, user_prompt, json_format=True)
    
    # DB에 누적 저장 (이전 상태를 덮어쓰지 않고 새로운 레코드로 추가하여 히스토리 관리 가능)
    # 여기서는 최신 상태를 참조하기 위해 하나씩 생성
    for name, state in res.items():
        CharacterState.objects.create(
            story=story,
            character_name=name,
            state_data=state,
            update_context=context
        )

def _get_latest_character_states(story):
    """DB에서 최신 인물 상태 문자열로 가져오기"""
    states = CharacterState.objects.filter(story=story).order_by('created_at')
    # 인물별로 가장 마지막 상태만 추림
    latest_map = {}
    for s in states:
        latest_map[s.character_name] = s.state_data
    return json.dumps(latest_map, ensure_ascii=False)

def _create_nodes_from_synopsis(story, synopsis, start_phase_idx=0, is_twist_branch=False):
    """5 & 6 & 11. 시놉시스를 기반으로 챕터별 줄거리 생성 및 노드화 (각 챕터당 2노드)"""
    phases = ["발단", "전개", "절정", "결말"]
    nodes = []
    
    # 현재 인물 상태 로드 (개연성 확보용)
    char_states_str = _get_latest_character_states(story)

    # 시놉시스를 단계별로 나누기
    sys_prompt = (
        "시놉시스를 읽고 '발단', '전개', '절정', '결말' 4단계로 나누세요. "
        "그리고 각 단계를 다시 '상(파트1)', '하(파트2)' 두 부분으로 나누어 "
        "총 8개의 상세 줄거리(각 1000자~2000자)를 JSON 리스트로 만드세요."
    )
    user_prompt = (
        f"시놉시스: {synopsis}\n"
        f"현재 인물 내면 상태(개연성 필수 반영): {char_states_str}\n"
        "형식: { 'scenes': [ '발단-1 내용', '발단-2 내용', '전개-1 내용', ... '결말-2 내용' ] }"
    )
    
    res = call_llm(sys_prompt, user_prompt, json_format=True)
    scenes = res.get('scenes', [])
    
    # 필요한 부분만 슬라이싱 (Twist 이후 생성 시 start_phase_idx 사용)
    # start_phase_idx가 2라면 (절정) -> scenes 리스트의 인덱스 4부터 시작
    start_list_idx = start_phase_idx * 2
    target_scenes = scenes[start_list_idx:]
    
    current_prev_node = None
    
    for i, content in enumerate(target_scenes):
        # 현재 단계 계산
        total_idx = start_list_idx + i
        phase_name = phases[min(total_idx // 2, 3)]
        
        node = StoryNode.objects.create(
            story=story,
            chapter_phase=phase_name,
            content=content,
            # prev_node는 여기서 연결하지 않고 리스트 반환 후 별도 로직으로 연결하거나
            # 바로 연결할 수도 있음. 여기선 선형 연결을 위해 바로 저장하지 않고 리스트로 반환하여 후처리
        )
        nodes.append(node)
        
    return nodes

def _connect_linear_nodes(nodes):
    """7 & 13. 노드 간 선형 연결 (Illusion of Choice)"""
    for i in range(len(nodes) - 1):
        curr = nodes[i]
        next_n = nodes[i+1]
        
        # 순서 연결
        next_n.prev_node = curr
        next_n.save()
        
        # 선택지 생성 (2개)
        sys_prompt = (
            "현재 노드에서 다음 노드로 넘어가기 위한 선택지 2개를 만드세요. "
            "이 선택지들은 '사용자가 참여한다'는 느낌만 줄 뿐, 실제로는 같은 다음 노드로 이어집니다. "
            "각 선택지에 대해 '선택 직후의 짧은 행동/결과 묘사(result_text)'를 생성하세요."
        )
        user_prompt = (
            f"현재 장면: {curr.content[-500:]}\n"
            f"다음 장면: {next_n.content[:500]}\n"
            "형식: { 'choices': [ {'text': '선택지1', 'result': '직후묘사1'}, {'text': '선택지2', 'result': '직후묘사2'} ] }"
        )
        
        res = call_llm(sys_prompt, user_prompt, json_format=True)
        
        for item in res.get('choices', []):
            NodeChoice.objects.create(
                current_node=curr,
                choice_text=item['text'],
                result_text=item['result'],
                next_node=next_n,
                is_twist_path=False
            )

def _find_twist_point_index(nodes):
    """8. 변주 지점 찾기"""
    # 마지막은 결말이므로 제외. 보통 전개(index 2~3)나 절정(index 4~5) 사이.
    # LLM에게 텍스트를 주고 가장 적절한 곳을 고르게 함
    summaries = [f"Index {i}: {n.chapter_phase} - {n.content[:100]}..." for i, n in enumerate(nodes[:-2])]
    
    sys_prompt = (
        "이 스토리의 흐름을 보고, 장르를 비틀어 전혀 다른 이야기로 전환하기 가장 극적이고 좋은 지점(Index)을 하나 선택하세요."
        "출력은 JSON으로 { 'index': 숫자 } 만 하세요."
    )
    user_prompt = "\n".join(summaries)
    
    res = call_llm(sys_prompt, user_prompt, json_format=True)
    idx = res.get('index', 2)
    
    # 안전 장치
    if idx >= len(nodes) - 2: idx = len(nodes) - 3
    if idx < 1: idx = 1
    
    # 해당 노드를 twist point로 마킹
    nodes[idx].is_twist_point = True
    nodes[idx].save()
    
    return idx

def _generate_twisted_synopsis_data(story, accumulated_story, current_phase):
    """9. 비틀기 시놉시스 생성"""
    # 어울리지 않는 새로운 클리셰 랜덤 추천 요청
    sys_prompt = (
        "현재까지의 이야기를 바탕으로, 장르를 급격하게 전환(Twist)할 수 있는 "
        "완전히 다른 장르의 클리셰를 추천하고, "
        "그 시점 이후의 새로운 시놉시스를 작성하세요. 이전 내용은 유지해야 합니다."
    )
    user_prompt = f"현재 스토리:\n{accumulated_story}\n\n현재 단계: {current_phase}에서 전환"
    
    # 여기서는 편의상 텍스트로 통째로 받지만, 
    # 실제로는 클리셰 ID를 받거나 새로 생성하는 로직이 더 정교할 수 있음.
    # 이번 구현에서는 텍스트로 시놉시스를 받아 저장합니다.
    full_text = call_llm(sys_prompt, user_prompt)
    
    # 더미 클리셰 객체 반환 (혹은 분석해서 매칭)
    return None, full_text

def _create_twist_branch_choices(node, old_next, new_next):
    """12. 비틀기 지점의 4개 선택지 생성 (2개 -> 구, 2개 -> 신)"""
    
    sys_prompt = (
        "스토리의 중대한 분기점입니다. 4개의 선택지를 생성하세요.\n"
        "선택지 1, 2는 기존의 스토리 흐름(Original Path)으로 자연스럽게 이어져야 합니다.\n"
        "선택지 3, 4는 장르가 급변하는 새로운 흐름(Twist Path)으로 이어져야 합니다.\n"
        "각 선택지의 직후 행동 묘사(result_text)도 포함하세요."
    )
    user_prompt = (
        f"현재 장면: {node.content[-500:]}\n"
        f"기존 다음 장면(Original): {old_next.content[:500]}\n"
        f"새로운 다음 장면(Twist): {new_next.content[:500]}\n"
        "형식: { 'original_choices': [{'text':'...', 'result':'...'}, ...], 'twist_choices': [{'text':'...', 'result':'...'}, ...] }"
    )
    
    res = call_llm(sys_prompt, user_prompt, json_format=True)
    
    # 기존 경로 선택지 저장
    for item in res.get('original_choices', []):
        NodeChoice.objects.create(
            current_node=node,
            choice_text=item['text'],
            result_text=item['result'],
            next_node=old_next,
            is_twist_path=False
        )
        
    # 새로운 경로 선택지 저장
    for item in res.get('twist_choices', []):
        NodeChoice.objects.create(
            current_node=node,
            choice_text=item['text'],
            result_text=item['result'],
            next_node=new_next,
            is_twist_path=True
        )