import os
import json
import time
from openai import OpenAI
from django.conf import settings
from .models import Genre, Cliche, Story, CharacterState, StoryNode, NodeChoice

# API 설정
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
BASE_URL = "https://api.fireworks.ai/inference/v1"
MODEL_NAME = "accounts/fireworks/models/deepseek-v3p1"
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=BASE_URL)

def call_llm(system_prompt, user_prompt, json_format=False, max_retries=3):
    """LLM 호출 래퍼 (재시도 로직 포함)"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    response_format = {"type": "json_object"} if json_format else None
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                response_format=response_format,
                temperature=0.7,
                max_tokens=4000 
            )
            content = response.choices[0].message.content
            
            if json_format:
                # JSON 파싱 보정
                cleaned_content = content
                if "```json" in cleaned_content:
                    cleaned_content = cleaned_content.replace("```json", "").replace("```", "")
                elif "```" in cleaned_content:
                    cleaned_content = cleaned_content.replace("```", "")
                
                return json.loads(cleaned_content)
            
            return content

        except json.JSONDecodeError as e:
            print(f"JSON Parsing Error (Attempt {attempt+1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                print("Max retries reached. Returning empty dict.")
                return {}
            time.sleep(1) # 잠시 대기 후 재시도
            
        except Exception as e:
            print(f"LLM Critical Error: {e}")
            # 치명적 오류(네트워크 등)는 바로 종료하거나 필요시 재시도 로직 추가 가능
            return {} if json_format else ""

    return {} if json_format else ""

# ==========================================
# [메인 파이프라인] 14단계 프로세스
# ==========================================
# ... (이후 create_story_pipeline 및 내부 함수들은 기존과 동일하므로 생략) ...
# 기존 로직 유지: create_story_pipeline, _match_cliche, _generate_synopsis 등
# 위 call_llm 함수만 교체해주시면 됩니다.
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
    synopsis = _generate_synopsis(story, matched_cliche)
    story.synopsis = synopsis
    story.save()

    # 4. 인물 내면 변화 추출 및 DB 저장
    _analyze_and_save_character_state(story, synopsis, context="Initial Synopsis")

    # 5 & 6. 챕터별 줄거리 생성 및 노드화 (선형 구조, 총 8노드)
    original_nodes = _create_nodes_from_synopsis(story, synopsis, start_phase_idx=0)

    # 7. 선형 노드 간 연결 (Illusion of Choice)
    _connect_linear_nodes(original_nodes)

    # 8. 클리셰 비틀기(Twist) 지점 찾기
    twist_node_index = _find_twist_point_index(original_nodes)
    twist_node = original_nodes[twist_node_index]
    
    story.twist_point_node_id = twist_node.id
    story.save()

    # 9. 비틀기 이후 새로운 시놉시스 생성 & 새 클리셰 매칭
    accumulated_story = "\n".join([n.content for n in original_nodes[:twist_node_index+1]])
    current_phase = twist_node.chapter_phase
    
    # [수정] 실제 DB에서 Twist용 클리셰를 추천받아 매칭
    twist_cliche, twisted_synopsis = _generate_twisted_synopsis_data(story, accumulated_story, current_phase)
    
    story.twist_cliche = twist_cliche
    story.twisted_synopsis = twisted_synopsis
    story.save()

    # 10. 인물 내면 변화 DB 업데이트 (새 시놉시스 반영)
    _analyze_and_save_character_state(story, twisted_synopsis, context="Twisted Synopsis")

    # 11 & 11-2. 비틀린 이후의 새로운 노드 생성
    # twist_node_index 이후의 단계부터 새로 생성합니다.
    # 예: index 3(전개-하)에서 비틀면, index 4(절정-상)부터 새로 만듭니다.
    next_start_idx = (twist_node_index // 2) + 1
    new_branch_nodes = _create_nodes_from_synopsis(
        story, 
        twisted_synopsis, 
        start_phase_idx=next_start_idx, 
        is_twist_branch=True
    )

    # 12. 비틀기 지점(Twist Node)에서의 분기 처리 (4개 선택지)
    # 기존 다음 노드(Original) vs 새 루트 첫 노드(Twist)
    if twist_node_index + 1 < len(original_nodes) and new_branch_nodes:
        original_next_node = original_nodes[twist_node_index + 1]
        new_next_node = new_branch_nodes[0]
        
        # 기존 연결(7번에서 생성됨)은 삭제하고 새로 4개를 만듭니다.
        NodeChoice.objects.filter(current_node=twist_node).delete()
        
        _create_twist_branch_choices(twist_node, original_next_node, new_next_node)

    # 13. 새로 생성된 브랜치 내부 연결 (선형)
    _connect_linear_nodes(new_branch_nodes)

    return story.id


# ==========================================
# [내부 로직 함수들]
# ==========================================

def _match_cliche(setting):
    """2. 사용자 설정에 가장 적합한 클리셰 매칭"""
    all_cliches = Cliche.objects.select_related('genre').all()
    if not all_cliches.exists():
        return None
        
    cliche_info = "\n".join([f"ID {c.id}: [{c.genre.name}] {c.title} - {c.summary}" for c in all_cliches])
    
    sys_prompt = "당신은 스토리 분석가입니다. 사용자 설정에 가장 적합한 클리셰 ID를 JSON으로 반환하세요."
    user_prompt = f"사용자 설정: {setting}\n\n보유 클리셰 목록:\n{cliche_info}\n\n출력형식: {{'cliche_id': 숫자}}"
    
    res = call_llm(sys_prompt, user_prompt, json_format=True)
    try:
        return Cliche.objects.get(id=res['cliche_id'])
    except:
        return all_cliches.first()

def _generate_synopsis(story, cliche):
    """3. 초기 시놉시스 생성"""
    role = "소설가"
    task = "전체 시놉시스를 작성하세요. 발단, 전개, 절정, 결말이 명확해야 합니다."
    
    content = (
        f"사용자 설정: {story.user_world_setting}\n"
        f"매칭된 클리셰: {cliche.title}\n"
        f"클리셰 가이드: {cliche.structure_guide}\n"
        f"참고 작품 감정선: {cliche.example_work_summary}\n\n"
        "조건 1: 사건의 원인과 해결 방식은 '사용자 설정'을 따르세요.\n"
        "조건 2: 인물의 감정선과 갈등 단계는 '참고 작품'과 '클리셰'를 벤치마킹하세요.\n"
        "조건 3: 총 2000자 내외의 줄거리(Plot) 형태로 작성하세요."
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
    
    for name, state in res.items():
        CharacterState.objects.create(
            story=story,
            character_name=name,
            state_data=state,
            update_context=context
        )

def _get_latest_character_states(story):
    """DB에서 최신 인물 상태 가져오기"""
    states = CharacterState.objects.filter(story=story).order_by('created_at')
    latest_map = {}
    for s in states:
        latest_map[s.character_name] = s.state_data
    return json.dumps(latest_map, ensure_ascii=False)

def _create_nodes_from_synopsis(story, synopsis, start_phase_idx=0, is_twist_branch=False):
    """5, 6, 11. 챕터별 줄거리 생성 및 노드화 (각 챕터당 2노드)"""
    phases = ["발단", "전개", "절정", "결말"]
    nodes = []
    
    char_states_str = _get_latest_character_states(story)

    sys_prompt = (
        "제공된 시놉시스를 기반으로 상세 스토리 씬을 생성합니다. "
        "전체 이야기를 '발단', '전개', '절정', '결말' 4단계로 나누고, "
        "각 단계를 다시 '상(파트1)', '하(파트2)' 두 부분으로 나누어 "
        "총 8개의 상세 줄거리(각 2000자 내외)를 JSON 리스트로 만드세요. "
        "인물의 내면 상태와 행동이 모순되지 않도록 주의하세요."
    )
    
    # [중요] Twist 브랜치일 경우, 시놉시스 전체를 주되 흐름을 유지하도록 강조
    context_note = ""
    if is_twist_branch:
        context_note = "주의: 이 시놉시스는 중간에 장르가 바뀐(Twist) 이야기입니다. 전체 흐름을 고려하여 8개 씬을 모두 완성하세요."

    user_prompt = (
        f"시놉시스: {synopsis}\n"
        f"현재 인물 내면 상태: {char_states_str}\n"
        f"{context_note}\n"
        "형식: { 'scenes': [ '발단-1 내용', '발단-2 내용', '전개-1 내용', ... '결말-2 내용' ] }"
    )
    
    res = call_llm(sys_prompt, user_prompt, json_format=True)
    scenes = res.get('scenes', [])
    
    # 필요한 부분만 슬라이싱
    # start_phase_idx (0~3). 0=발단, 1=전개, 2=절정, 3=결말
    start_list_idx = start_phase_idx * 2
    
    # 만약 LLM이 8개를 다 안 주고 남은 것만 줬을 경우를 대비한 방어 로직
    if len(scenes) < 8 and is_twist_branch:
         # LLM이 똑똑해서 남은 부분만 줬다고 가정하고 그대로 씀
         target_scenes = scenes
    else:
         target_scenes = scenes[start_list_idx:]
    
    for i, content in enumerate(target_scenes):
        total_idx = start_list_idx + i
        if total_idx >= 8: break # 인덱스 초과 방지
        
        phase_name = phases[min(total_idx // 2, 3)]
        
        node = StoryNode.objects.create(
            story=story,
            chapter_phase=phase_name,
            content=content,
        )
        nodes.append(node)
        
    return nodes

def _connect_linear_nodes(nodes):
    """7 & 13. 노드 간 선형 연결 (Illusion of Choice)"""
    for i in range(len(nodes) - 1):
        curr = nodes[i]
        next_n = nodes[i+1]
        
        next_n.prev_node = curr
        next_n.save()
        
        sys_prompt = (
            "현재 노드에서 다음 노드로 넘어가기 위한 선택지 2개를 만드세요. "
            "이 선택지들은 스토리를 분기시키지 않고, 동일한 다음 노드로 이어집니다. "
            "각 선택지에 대해 '선택 직후의 짧은 행동 묘사(result_text)'를 생성하세요. "
            "result_text는 다음 노드의 첫 문장 앞에 자연스럽게 붙어야 합니다."
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
    """8. 변주 지점 찾기 (전개~절정 사이)"""
    if len(nodes) < 4: return 1
    
    # 마지막(결말)은 제외하고 중간 부분에서 탐색
    summaries = [f"Index {i}: {n.chapter_phase} - {n.content[:100]}..." for i, n in enumerate(nodes[:-2])]
    
    sys_prompt = (
        "이 스토리의 장르를 비틀어(Twist) 전혀 다른 장르(클리셰)로 전환하기 가장 극적인 지점(Index)을 하나 선택하세요."
        "출력은 JSON으로 { 'index': 숫자 } 만 하세요."
    )
    user_prompt = "\n".join(summaries)
    
    res = call_llm(sys_prompt, user_prompt, json_format=True)
    idx = res.get('index', 2)
    
    # 안전 장치
    if idx >= len(nodes) - 2: idx = len(nodes) - 3
    if idx < 1: idx = 1
    
    nodes[idx].is_twist_point = True
    nodes[idx].save()
    
    return idx

def _generate_twisted_synopsis_data(story, accumulated_story, current_phase):
    """9. 비틀기 클리셰 선정 및 시놉시스 생성"""
    
    # 9-1. 새로운 클리셰 추천 (DB에서 찾기)
    all_cliches = Cliche.objects.exclude(id=story.main_cliche.id).select_related('genre').all()
    if not all_cliches.exists():
        return None, "클리셰 데이터 부족"

    cliche_info = "\n".join([f"ID {c.id}: [{c.genre.name}] {c.title}" for c in all_cliches])
    
    rec_sys = "당신은 반전의 대가입니다. 현재까지 진행된 스토리 속에 숨겨진 '애매한 요소'나 '미해결 떡밥'을 찾으세요. "
    "이를 전혀 다른 장르의 관점에서 논리적으로 재해석할 수 있는 새로운 클리셰 ID를 추천하세요. "
    "(예: '유령(호러)'인 줄 알았으나 사실 '홀로그램(SF)'이었다 / '로맨스'인 줄 알았으나 '범죄 스릴러'의 타깃이었다)"
    rec_user = f"현재 스토리:\n{accumulated_story[-1000:]}\n\n후보 목록:\n{cliche_info}\n\n출력: {{'cliche_id': 숫자}}"
    
    rec_res = call_llm(rec_sys, rec_user, json_format=True)
    try:
        new_cliche = Cliche.objects.get(id=rec_res['cliche_id'])
    except:
        new_cliche = all_cliches.first()

    # 9-2. 새로운 시놉시스 생성 (전체 분량)
    sys_prompt = (
        "당신은 치밀한 복선 회수의 대가입니다. "
        "주어진 '현재까지의 이야기'를 유지하되, 그동안 독자가 당연하게 여겼던 사실들을 '새로운 클리셰'의 관점에서 뒤집으세요. "
        "단순한 장르 전환이 아니라, '아, 그래서 아까 그런 일이 있었구나!'라고 무릎을 탁 치게 만드는 필연적인 인과관계를 포함하여 전체 시놉시스를 재구성하세요."
    )
    user_prompt = (
        f"현재까지 이야기: {accumulated_story}\n"
        f"전환 지점: {current_phase}부터 장르 전환\n"
        f"새로운 클리셰: {new_cliche.title} ({new_cliche.summary})\n"
        f"가이드: {new_cliche.structure_guide}\n"
        "출력: 전체 시놉시스 (2000자 내외)"
    )
    
    twisted_synopsis = call_llm(sys_prompt, user_prompt)
    
    return new_cliche, twisted_synopsis

def _create_twist_branch_choices(node, old_next, new_next):
    """12. 비틀기 지점의 4개 선택지 생성"""
    
    sys_prompt = (
        "스토리의 장르가 전환되는 결정적 분기점입니다. 4개의 선택지를 생성하세요.\n"
        "선택지 1, 2 (Original): 기존 장르의 문법을 따르는 안전한 선택입니다.\n"
        "선택지 3, 4 (Twist): 주인공의 성격에 부합하는 자연스러운 행동이지만, 그 결과로 숨겨진 진실(새로운 장르)을 마주하게 되는 선택입니다.\n"
        "주의: 선택지 텍스트 자체는 너무 뜬금없지 않아야 합니다. 선택에 따른 '직후 행동 묘사(result_text)'에서 충격적인 진실이 드러나게 하세요."
    )
    user_prompt = (
        f"현재 장면: {node.content[-500:]}\n"
        f"기존 다음 장면(Original): {old_next.content[:500]}\n"
        f"새로운 다음 장면(Twist): {new_next.content[:500]}\n"
        "형식: { 'original_choices': [{'text':'...', 'result':'...'}, ...], 'twist_choices': [{'text':'...', 'result':'...'}, ...] }"
    )
    
    res = call_llm(sys_prompt, user_prompt, json_format=True)
    
    for item in res.get('original_choices', []):
        NodeChoice.objects.create(
            current_node=node,
            choice_text=item['text'],
            result_text=item['result'],
            next_node=old_next,
            is_twist_path=False
        )
        
    for item in res.get('twist_choices', []):
        NodeChoice.objects.create(
            current_node=node,
            choice_text=item['text'],
            result_text=item['result'],
            next_node=new_next,
            is_twist_path=True
        )