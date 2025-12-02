import os
import json
import time
import uuid
import random
from openai import OpenAI
from django.conf import settings
from .models import Genre, Cliche, Story, CharacterState, StoryNode, NodeChoice

from .neo4j_connection import (
    create_universe_node_neo4j, 
    sync_node_to_neo4j, 
    link_universe_to_first_scene, 
    sync_choice_to_neo4j, 
    StoryNodeData
)

# API 설정
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
BASE_URL = "https://api.fireworks.ai/inference/v1"
MODEL_NAME = "accounts/fireworks/models/deepseek-v3p1" 
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=BASE_URL)

def call_llm(system_prompt, user_prompt, json_format=False, max_retries=3):
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    response_format = {"type": "json_object"} if json_format else None
    
    if not DEEPSEEK_API_KEY:
        print("🚨 [Critical] API Key is MISSING!")
        return {} if json_format else ""

    for attempt in range(max_retries):
        try:
            # 스트리밍 비활성화 유지
            stream_option = False 
            
            response = client.chat.completions.create(
                model=MODEL_NAME, 
                messages=messages, 
                response_format=response_format, 
                temperature=0.5, # 창의성 조절값
                max_tokens=4000, 
                timeout=90,
                stream=stream_option 
            )
            
            content = response.choices[0].message.content

            if json_format:
                # JSON 파싱 전 마크다운 제거 처리 강화
                cleaned = content.replace("```json", "").replace("```", "").strip()
                return json.loads(cleaned)
            return content
        except Exception as e:
            print(f"⚠️ [LLM Error] Attempt {attempt+1}/{max_retries} Failed: {str(e)}")
            time.sleep(2)
            
    print(f"❌ [Final Fail] LLM Call Failed completely.")
    return {} if json_format else ""

# ==========================================
# [메인 파이프라인]
# ==========================================

def create_story_pipeline(user_world_setting):
    """
    스토리 생성 전체 파이프라인
    """
    universe_id = str(uuid.uuid4())
    print(f"\n🌍 [NEO4J] Creating Universe Node: {universe_id}")

    # 1. 설정 구체화 및 주인공 정의 (이름 생성 강화)
    refined_setting, protagonist_name = _refine_setting_and_protagonist(user_world_setting)
    print(f"✅ Refined Setting: {refined_setting[:50]}... / Protagonist: {protagonist_name}")

    try:
        create_universe_node_neo4j(universe_id, refined_setting)
    except Exception as e:
        print(f"Neo4j Error: {e}")

    # 2. 클리셰 매칭 (로직 개선)
    matched_cliche = _match_cliche(refined_setting)
    if not matched_cliche:
        # DB가 비어있지 않은 이상 발생하지 않아야 함
        raise ValueError("적절한 클리셰를 찾지 못했습니다.")
    print(f"✅ Matched Cliche: {matched_cliche.title}")
    
    story = Story.objects.create(user_world_setting=refined_setting, main_cliche=matched_cliche)
    
    # 3. 시놉시스 생성
    synopsis = _generate_synopsis(story, matched_cliche, protagonist_name)
    story.synopsis = synopsis
    story.save()

    # 4. 인물 내면 상태 분석
    _analyze_and_save_character_state(story, synopsis, context="Initial Synopsis")

    # 5 & 6. 초기 노드 생성 (최소 500자 보장)
    original_nodes = _create_nodes_from_synopsis(
        story, synopsis, protagonist_name, 
        start_node_index=0, 
        universe_id=universe_id,
        is_twist_branch=False
    )
    
    if not original_nodes or len(original_nodes) < 2:
        print("❌ [Error] 노드 생성 실패.")
        raise ValueError("AI가 스토리 노드를 생성하지 못했습니다.")  
    
    # Neo4j 연결 (첫 노드)
    if original_nodes:
        try:
            first_node_uid = f"{universe_id}_{original_nodes[0].id}"
            link_universe_to_first_scene(universe_id, first_node_uid)
        except Exception as e:
            print(f"Neo4j Link Error: {e}")

    # 7. 선형 연결 (주인공 이름 사용)
    _connect_linear_nodes(original_nodes, universe_id, protagonist_name)

    # 8. 비틀기(Twist) 지점 찾기
    twist_node_index = _find_twist_point_index(original_nodes)
    
    # 인덱스 안전장치
    if twist_node_index >= len(original_nodes) - 1: 
        twist_node_index = len(original_nodes) - 2
    if twist_node_index < 0: twist_node_index = 0
        
    twist_node = original_nodes[twist_node_index]
    story.twist_point_node_id = twist_node.id
    story.save()

    accumulated_content = "\n".join([n.content for n in original_nodes[:twist_node_index+1]])
    
    # 9. 비틀린 시놉시스 생성 (동일 클리셰 변주)
    twisted_synopsis = _generate_twisted_synopsis_data(story, accumulated_content, twist_node.chapter_phase)
    
    story.twisted_synopsis = twisted_synopsis
    story.save()
    
    # 10. 비틀린 시놉시스 기반 내면 분석
    _analyze_and_save_character_state(story, twisted_synopsis, context="Twisted Synopsis")

    # 11. 비틀기 노드 생성 (최소 500자)
    new_branch_nodes = _create_nodes_from_synopsis(
        story, twisted_synopsis, protagonist_name,
        start_node_index=twist_node_index+1, 
        is_twist_branch=True, 
        universe_id=universe_id
    )

    # 12. 분기 처리 (변주 선택지 추가)
    if new_branch_nodes:
        twist_next_node = new_branch_nodes[0]
        # [수정 5] 동일 상황 다른 행동 선택지 생성
        _add_twist_branch_choices_only(twist_node, twist_next_node, universe_id, protagonist_name)

    # 13. 새 브랜치 내부 연결
    _connect_linear_nodes(new_branch_nodes, universe_id, protagonist_name)

    return story.id

# ==========================================
# [내부 로직 함수들]
# ==========================================

def _refine_setting_and_protagonist(raw_setting):
    # [수정 1] 이름 생성 강화 (하드코딩 제거)
    sys_prompt = (
        "당신은 창의적인 스토리 작가입니다. 사용자의 입력을 분석하여 세계관을 확정하고 주인공을 정의하세요. "
        "**[필수] 사용자가 주인공의 이름을 지정하지 않았다면, 세계관과 분위기에 어울리는 멋진 이름을 반드시 창작하세요.** "
        "절대 '주인공', '나', '행인1' 같은 대명사나 성의 없는 이름을 사용하지 마세요. 구체적인 이름(예: 카엘, 지수, 아서 등)을 지어주세요."
    )
    user_prompt = (
        f"사용자 입력: {raw_setting}\n\n"
        "다음 형식의 JSON으로 출력하세요:\n"
        "{\n"
        "  'refined_setting': '확정된 구체적인 세계관 및 배경 설정 (텍스트)',\n"
        "  'protagonist_name': '확정되거나 창작된 주인공 이름 (문자열)',\n"
        "  'protagonist_desc': '주인공의 성격, 외모, 특징'\n"
        "}"
    )
    res = call_llm(sys_prompt, user_prompt, json_format=True)
    setting = res.get('refined_setting', raw_setting)
    name = res.get('protagonist_name', '') 
    
    # 만약 AI가 이름을 못 지었을 경우를 대비한 2차 안전장치 (하드코딩 대신 랜덤 생성 요청)
    if not name or name.strip() in ["주인공", "나", "Unknown", "미정"]:
        # 간단히 다시 요청
        name_res = call_llm("이 세계관에 어울리는 주인공 이름을 1개만 단답형으로 지어줘.", f"세계관: {setting}")
        name = name_res.strip().replace("이름:", "").replace(".", "")
        if not name: name = "이안" # 최후의 수단
        
    return setting, name

def _match_cliche(setting):
    # [수정 1] 클리셰 선택 다양화
    all_cliches = Cliche.objects.select_related('genre').all()
    if not all_cliches.exists(): return None
    
    # 목록을 셔플하여 프롬프트에 제공 (순서 편향 방지)
    cliche_list = list(all_cliches)
    random.shuffle(cliche_list)
    
    cliche_info = "\n".join([f"ID {c.id}: [{c.genre.name}] {c.title} - {c.summary}" for c in cliche_list])
    
    sys_prompt = (
        "사용자의 설정과 가장 잘 어울리는 클리셰(Cliche)를 하나 선택하세요. "
        "단순히 첫 번째 것을 고르지 말고, 사용자의 설정 내용, 장르, 분위기를 깊게 분석하여 가장 적절한 것을 찾으세요. "
        "출력은 반드시 JSON 형식입니다."
    )
    
    res = call_llm(
        sys_prompt, 
        f"사용자 설정: {setting}\n\n후보 클리셰 목록:\n{cliche_info}\n\n출력형식: {{'cliche_id': 숫자}}", 
        json_format=True
    )
    
    try: 
        selected_id = res['cliche_id']
        return Cliche.objects.get(id=selected_id)
    except: 
        # 실패 시 랜덤 선택
        print("⚠️ [Warning] 클리셰 매칭 실패. 랜덤으로 선택합니다.")
        return random.choice(all_cliches)

def _generate_synopsis(story, cliche, protagonist_name):
    # [Req 3] 시놉시스 생성
    sys_prompt = (
        "당신은 베스트셀러 작가입니다. 사용자의 설정과 선택된 클리셰를 결합하여 기승전결(발단-전개-절정-결말)이 완벽한 시놉시스를 작성하세요. "
        "1. **감정선과 갈등 구조**는 참고 작품을 벤치마킹하세요. "
        "2. **사건의 구체적인 내용, 원인, 해결 방식**은 사용자 설정(배경, 능력)을 사용하여 완전히 새롭게 창작하세요. "
        "3. 분량은 공백 포함 2000자 내외로 풍성하게 작성하세요."
    )
    user_prompt = (
        f"세계관: {story.user_world_setting}\n"
        f"주인공: {protagonist_name}\n"
        f"적용 클리셰: {cliche.title} ({cliche.summary})\n"
        f"클리셰 가이드: {cliche.structure_guide}\n"
        f"참고 작품 감정선: {cliche.example_work_summary}\n\n"
        "위 정보를 바탕으로 전체 시놉시스를 작성하세요."
    )
    return call_llm(sys_prompt, user_prompt)

def _analyze_and_save_character_state(story, text, context):
    sys_prompt = (
        "텍스트를 심층 분석하여 등장인물들의 내면 상태 변화를 추출하세요. "
        "각 사건, 행동, 결정이 인물에게 어떤 감정적, 사상적, 관계적 변화를 주었는지 구체적으로 기록해야 합니다."
    )
    user_prompt = f"텍스트: {text}\n출력 형식: {{'캐릭터이름': {{'emotion': '...', 'trust': '...', 'ideology': '...', 'relationship_change': '...'}}}}"
    
    res = call_llm(sys_prompt, user_prompt, json_format=True)
    for name, state in res.items():
        CharacterState.objects.create(
            story=story, 
            character_name=name, 
            state_data=state, 
            update_context=context
        )

def _get_latest_character_states(story):
    latest_states = {}
    states = CharacterState.objects.filter(story=story).order_by('created_at')
    for s in states:
        latest_states[s.character_name] = s.state_data
    return json.dumps(latest_states, ensure_ascii=False)

def _create_nodes_from_synopsis(story, synopsis, protagonist_name, start_node_index=0, is_twist_branch=False, universe_id=None):
    # [수정 2] 최소 500자 보장
    phases = ["발단", "전개", "절정", "결말"]
    nodes = []
    char_states_str = _get_latest_character_states(story)

    sys_prompt = (
        f"당신은 인터랙티브 스토리 게임의 작가입니다. 시놉시스를 바탕으로 플레이어가 진행할 구체적인 장면(Node)들을 생성하세요. "
        f"주인공은 '{protagonist_name}'입니다.\n"
        "**[필수 제약 사항]**\n"
        "1. 각 장면의 내용은 **공백 포함 최소 500자 이상**으로 아주 상세하고 몰입감 있게 작성해야 합니다. (너무 짧으면 안 됩니다.)\n"
        "2. **제공된 인물 내면 상태(Character State)를 반드시 반영**하여, 인물의 말과 행동이 내면과 일치하고 개연성을 가지도록 하세요.\n"
        "3. 문체는 서술형(~한다)을 사용하세요."
    )
    
    if is_twist_branch:
        sys_prompt += "\n4. **[중요] 마지막 장면에서는 이야기가 열린 결말 없이 완벽하게 종결되어야 합니다.**"

    needed_nodes = 8 - start_node_index
    
    user_prompt = (
        f"시놉시스: {synopsis}\n"
        f"현재 인물 내면 상태: {char_states_str}\n"
        f"현재 단계: {'Twist 이후 스토리' if is_twist_branch else '초기 스토리'}\n\n"
        f"다음 키를 포함한 JSON 리스트로 {needed_nodes}개의 장면을 순서대로 생성하세요:\n"
        "['title', 'description' (최소 500자 이상의 줄거리), 'setting', 'characters', 'purpose']\n"
        "형식: {'scenes': [...]}"
    )
    
    res = call_llm(sys_prompt, user_prompt, json_format=True)
    
    scenes = res.get('scenes', [])
    
    for i, scene_data in enumerate(scenes):
        current_idx = start_node_index + i
        phase_idx = min(current_idx // 2, 3)
        phase_name = phases[phase_idx]
        
        title = scene_data.get('title', '무제')
        description = scene_data.get('description', '')
        setting = scene_data.get('setting', '')
        purpose = scene_data.get('purpose', '')
        
        node = StoryNode.objects.create(story=story, chapter_phase=phase_name, content=description)
        nodes.append(node)
        
        if universe_id:
            try:
                neo4j_node_uid = f"{universe_id}_{node.id}"
                raw_chars = scene_data.get('characters', [])
                characters_str = ", ".join(raw_chars) if isinstance(raw_chars, list) else str(raw_chars)
                
                neo4j_data = StoryNodeData(
                    node_id=neo4j_node_uid,
                    phase=phase_name,
                    title=title,
                    setting=setting,
                    characters=characters_str, 
                    description=description[:200], 
                    purpose=str(purpose),
                    character_state=char_states_str
                )
                sync_node_to_neo4j(neo4j_data)
            except Exception as e:
                print(f"Neo4j Node Sync Error: {e}")

    return nodes

def _connect_linear_nodes(nodes, universe_id, protagonist_name):
    # [수정 4] 선택지에서 '주인공' 단어 사용 금지 (이름 사용)
    sys_prompt = (
        f"현재 장면에서 다음 장면으로 넘어가기 위한 선택지 2개를 생성하세요. 주인공 '{protagonist_name}'의 입장이 되어야 합니다.\n"
        "**[필수 조건]**\n"
        "1. **같은 상황(Scene)에 대한 서로 다른 행동**이어야 합니다.\n"
        "2. **선택지 텍스트('text')에는 '주인공'이라는 단어를 절대 쓰지 말고, 주인공의 이름 '{protagonist_name}'을 사용하세요.** (예: '{protagonist_name}은(는) 칼을 집어든다')\n"
        "3. 'result'(결과)는 선택지 행동의 직후 결과를 묘사하는 **완결된 문장**이어야 합니다.\n"
        "4. 다음 장면의 내용 자체는 바뀌지 않으므로, 결과 텍스트는 다음 장면의 첫 부분과 자연스럽게 이어져야 합니다."
    )
    
    for i in range(len(nodes) - 1):
        curr = nodes[i]
        next_n = nodes[i+1]
        
        next_n.prev_node = curr
        next_n.save()
        
        user_prompt = (
            f"현재 장면: {curr.content[-300:]}\n"
            f"다음 장면(이어질 내용): {next_n.content[:300]}\n\n"
            "형식: {'choices': ["
            f"{{'text': '{protagonist_name}은(는) ~한다', 'result': '그 결과 ~했다.'}}, "
            f"{{'text': '{protagonist_name}은(는) ~한다', 'result': '그 결과 ~했다.'}}"
            "]}"
        )
        
        res = call_llm(sys_prompt, user_prompt, json_format=True)

        for item in res.get('choices', []):
            choice_text = item.get('text', "다음으로")
            result_text = item.get('result', "")

            NodeChoice.objects.create(
                current_node=curr, 
                choice_text=choice_text, 
                result_text=result_text, 
                next_node=next_n, 
                is_twist_path=False
            )
            if universe_id:
                try:
                    curr_uid = f"{universe_id}_{curr.id}"
                    next_uid = f"{universe_id}_{next_n.id}"
                    sync_choice_to_neo4j(curr_uid, next_uid, choice_text, result_text, is_twist=False)
                except: pass

def _find_twist_point_index(nodes):
    if len(nodes) < 4: return 1
    summaries = [f"Idx {i}: {n.content[:100]}..." for i, n in enumerate(nodes[:-2])] 
    
    res = call_llm(
        "제공된 스토리 흐름을 보고, 클리셰를 비틀어 예상치 못한 방향(Twist)으로 이야기를 전개하기 가장 좋은 지점(Index)을 하나 고르세요.", 
        "\n".join(summaries) + "\n출력형식: {'index': 숫자}", 
        json_format=True
    )
    idx = res.get('index', 2)
    if idx < 1: idx = 1
    if idx >= len(nodes) - 2: idx = len(nodes) - 3
    
    return idx

def _generate_twisted_synopsis_data(story, accumulated_content, current_phase):
    sys_prompt = (
        "당신은 반전 스토리의 대가입니다. 지금까지 진행된 스토리의 클리셰를 유지하되, "
        "**이야기의 흐름을 비틀어(Twist) 전혀 다른 양상으로 전개되는 새로운 시놉시스**를 작성하세요. "
        "새로운 클리셰를 도입하지 말고, 현재 클리셰 안에서 사건의 해석을 달리하거나 돌발 변수를 추가하여 결말을 바꾸세요."
    )
    
    user_prompt = (
        f"현재 적용된 클리셰: {story.main_cliche.title} ({story.main_cliche.summary})\n"
        f"현재까지 진행된 줄거리: {accumulated_content[-1000:]}\n"
        f"현재 단계: {current_phase} 이후\n\n"
        "지시사항: 위 줄거리 이후부터 이어질 새로운 '전개-절정-결말' 시놉시스를 작성하세요. 인물들의 내면 변화를 반드시 반영해야 합니다."
    )
    
    twisted_synopsis = call_llm(sys_prompt, user_prompt)
    return twisted_synopsis

def _add_twist_branch_choices_only(node, new_next, universe_id, protagonist_name):
    # [수정 5] 변주 선택지 논리 강화 (동일 상황, 다른 행동, 이름 사용)
    sys_prompt = (
        f"이야기가 극적으로 갈라지는 분기점입니다. 주인공 '{protagonist_name}'의 선택에 따라 이야기가 완전히 바뀝니다. "
        f"이 선택지들은 **기존의 선택지들과 정확히 '동일한 상황'에서 시작되어야 합니다.** "
        "하지만 주인공이 기존과는 다른, 위험하거나 의외의 행동을 함으로써 'Twist Scene'으로 이어지게 만드세요.\n"
        "**[필수 조건]**\n"
        "1. 상황은 이전과 같지만, 행동이 파격적이어야 합니다.\n"
        f"2. **선택지 텍스트에는 '주인공' 대신 이름 '{protagonist_name}'을 사용하세요.**\n"
        "3. 결과(result)는 다음 장면(Twist Scene)의 첫 부분과 자연스럽게 연결되어야 합니다."
    )
    
    user_prompt = (
        f"현재 상황: {node.content[-300:]}\n"
        f"새로운 반전 장면 도입부: {new_next.content[:300]}\n\n"
        "형식: JSON\n"
        "{\n"
        "  'twist_choices': [\n"
        f"    {{'text': '{protagonist_name}은(는) ~한다 (반전 선택 1)', 'result': '그 결과 ~했다.'}},\n"
        f"    {{'text': '{protagonist_name}은(는) ~한다 (반전 선택 2)', 'result': '그 결과 ~했다.'}}\n"
        "  ]\n"
        "}"
    )
    
    res = call_llm(sys_prompt, user_prompt, json_format=True)
    
    curr_uid = f"{universe_id}_{node.id}"
    new_next_uid = f"{universe_id}_{new_next.id}"

    for item in res.get('twist_choices', []):
        text = item.get('text', '운명을 바꾸는 선택을 한다')
        result = item.get('result', '')
        
        NodeChoice.objects.create(
            current_node=node, 
            choice_text=text, 
            result_text=result, 
            next_node=new_next, 
            is_twist_path=True 
        )
        
        if universe_id:
            try: sync_choice_to_neo4j(curr_uid, new_next_uid, text, result, is_twist=True)
            except: pass