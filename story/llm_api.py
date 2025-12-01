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
            # max_tokens 5000 이상일 경우 stream=True 필수
            stream_option = True
            
            response = client.chat.completions.create(
                model=MODEL_NAME, 
                messages=messages, 
                response_format=response_format, 
                temperature=0.8,  # 창의성을 위해 온도를 약간 높임
                max_tokens=8000, 
                timeout=90,
                stream=stream_option 
            )
            
            content = ""
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    content += chunk.choices[0].delta.content

            if json_format:
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

    # 1. 설정 구체화 및 주인공 정의 (이름 자동 생성 포함)
    refined_setting, protagonist_name = _refine_setting_and_protagonist(user_world_setting)
    print(f"✅ Refined Setting: {refined_setting[:50]}... / Protagonist: {protagonist_name}")

    try:
        create_universe_node_neo4j(universe_id, refined_setting)
    except Exception as e:
        print(f"Neo4j Error: {e}")

    # 2. 클리셰 매칭 (사용자 입력 분석 강화)
    matched_cliche = _match_cliche(refined_setting)
    print(f"✅ Matched Cliche: {matched_cliche.title}")
    
    story = Story.objects.create(user_world_setting=refined_setting, main_cliche=matched_cliche)
    
    synopsis = _generate_synopsis(story, matched_cliche, protagonist_name)
    story.synopsis = synopsis
    story.save()

    # 3. 인물 내면 상태 분석
    _analyze_and_save_character_state(story, synopsis, context="Initial Synopsis")

    # 4. 초기 노드 생성
    original_nodes = _create_nodes_from_synopsis(
        story, synopsis, protagonist_name, 
        start_node_index=0, 
        universe_id=universe_id,
        is_twist_branch=False
    )
    
    if not original_nodes or len(original_nodes) < 2:
        print("❌ [Error] 노드 생성 실패. LLM 응답이 비어있거나 형식이 잘못되었습니다.")
        raise ValueError("AI가 스토리 노드를 생성하지 못했습니다.")  
    
    # 5. 첫 노드 연결 (Neo4j)
    if original_nodes:
        try:
            first_node_uid = f"{universe_id}_{original_nodes[0].id}"
            link_universe_to_first_scene(universe_id, first_node_uid)
        except Exception as e:
            print(f"Neo4j Link Error: {e}")

    # 6. 선형 연결
    _connect_linear_nodes(original_nodes, universe_id, protagonist_name)

    # 7. 비틀기(Twist) 설정
    twist_node_index = _find_twist_point_index(original_nodes)
    if twist_node_index >= len(original_nodes) - 1: 
        twist_node_index = len(original_nodes) - 2
    if twist_node_index < 0: twist_node_index = 0
        
    twist_node = original_nodes[twist_node_index]
    story.twist_point_node_id = twist_node.id
    story.save()

    accumulated = "\n".join([n.content for n in original_nodes[:twist_node_index+1]])
    
    # 8. 비틀린 시놉시스 생성 (같은 장르 내 변주)
    twist_cliche, twisted_synopsis = _generate_twisted_synopsis_data(story, accumulated, twist_node.chapter_phase)
    story.twist_cliche = twist_cliche
    story.twisted_synopsis = twisted_synopsis
    story.save()
    
    _analyze_and_save_character_state(story, twisted_synopsis, context="Twisted Synopsis")

    # 9. 비틀기 노드 생성
    new_branch_nodes = _create_nodes_from_synopsis(
        story, twisted_synopsis, protagonist_name,
        start_node_index=twist_node_index+1, 
        is_twist_branch=True, 
        universe_id=universe_id
    )

    # 10. 분기 처리
    if new_branch_nodes:
        new_next = new_branch_nodes[0]
        _add_twist_branch_choices_only(twist_node, new_next, universe_id, protagonist_name)

    # 11. 새 브랜치 내부 연결
    _connect_linear_nodes(new_branch_nodes, universe_id, protagonist_name)

    return story.id

# ==========================================
# [내부 로직 함수들]
# ==========================================

def _refine_setting_and_protagonist(raw_setting):
    # [수정] 주인공 이름 창작 지시 강화
    sys_prompt = (
        "당신은 창의적인 스토리 설정 분석가입니다. 사용자의 입력에서 모호한 설정을 확정하고, "
        "이야기의 중심이 될 주인공(Protagonist)을 정의하세요. "
        "**만약 사용자가 주인공의 이름을 지정하지 않았다면, 세계관과 분위기에 어울리는 멋진 이름을 직접 창작해서 지어주세요.** "
        "절대 '주인공', '무명', '행인1' 같은 성의 없는 이름을 사용하지 마세요."
    )
    user_prompt = (
        f"사용자 입력: {raw_setting}\n\n"
        "다음 형식의 JSON으로 출력하세요:\n"
        "{\n"
        "  'refined_setting': '확정된 구체적인 세계관 및 배경 설정 (텍스트)',\n"
        "  'protagonist_name': '확정되거나 창작된 주인공 이름 (필수)',\n"
        "  'protagonist_desc': '주인공의 성격, 외모, 특징'\n"
        "}"
    )
    res = call_llm(sys_prompt, user_prompt, json_format=True)
    setting = res.get('refined_setting', raw_setting)
    name = res.get('protagonist_name', '알렉스') # 혹시라도 비어있으면 임의의 이름
    
    # 안전장치: 여전히 '주인공'이라고 답했다면 강제로 랜덤 성씨라도 붙임 (여기선 간단히 처리)
    if name.strip() == "주인공":
        name = "이서준" # 임시 기본값
        
    return setting, name

def _match_cliche(setting):
    all_cliches = Cliche.objects.select_related('genre').all()
    if not all_cliches.exists(): return None
    
    cliche_info = "\n".join([f"ID {c.id}: [{c.genre.name}] {c.title} - {c.summary}" for c in all_cliches])
    
    # [수정] 클리셰 선정 기준 강화
    sys_prompt = (
        "당신은 최고의 스토리 기획자입니다. 사용자가 입력한 설정의 분위기, 장르적 코드, 핵심 소재를 깊이 분석하세요. "
        "제공된 클리셰 목록 중에서, 해당 설정과 결합했을 때 **가장 극적이고 흥미로운 화학 작용을 일으킬 수 있는 최적의 클리셰 하나**를 선택하세요. "
        "단순히 키워드가 일치하는 것이 아니라, 이야기의 재미를 극대화할 수 있는 것을 고르세요."
    )
    
    res = call_llm(
        sys_prompt, 
        f"사용자 설정: {setting}\n\n클리셰 목록:\n{cliche_info}\n\n출력형식: {{'cliche_id': 숫자}}", 
        json_format=True
    )
    
    try: 
        return Cliche.objects.get(id=res['cliche_id'])
    except: 
        # LLM 응답 실패 시 랜덤으로라도 다양성을 줌 (기존에는 무조건 first()였음)
        print("⚠️ [Warning] 클리셰 매칭 실패 또는 파싱 에러. 임의의 클리셰를 선택합니다.")
        return random.choice(all_cliches) if all_cliches else None

def _generate_synopsis(story, cliche, protagonist_name):
    # [수정] 한자 사용 금지 재강조
    sys_prompt = (
        "당신은 베스트셀러 소설가입니다. 다음 설정과 클리셰 구조를 바탕으로 기승전결이 확실한 시놉시스를 작성하세요. "
        "문체는 반드시 '~한다', '~했다'로 끝나는 건조한 서술체를 사용하세요. "
        "**절대 괄호 안에 한자(Chinese Characters)를 병기하지 마십시오. 오직 순수 한글로만 작성해야 합니다.**"
    )
    user_prompt = (
        f"세계관: {story.user_world_setting}\n"
        f"주인공: {protagonist_name}\n"
        f"적용 클리셰: {cliche.title}\n"
        f"가이드: {cliche.structure_guide}\n"
        f"참고 작품 감정선: {cliche.example_work_summary}\n\n"
        "지시: 사건의 원인과 해결 방식은 사용자 설정을 따르되, 감정선은 참고 작품을 벤치마킹하여 2000자 내외로 작성하세요."
    )
    return call_llm(sys_prompt, user_prompt)

def _analyze_and_save_character_state(story, text, context):
    sys_prompt = (
        "텍스트를 분석하여 등장인물들의 내면 상태(감정, 신뢰도, 사상, 육체 상태 등)를 갱신하세요. "
        "이 데이터는 이후 스토리의 개연성을 위해 사용됩니다."
    )
    user_prompt = f"텍스트: {text}\n출력 형식: {{'캐릭터이름': {{'emotion': '...', 'trust': '...', 'physical': '...'}}}}"
    
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
    phases = ["발단", "전개", "절정", "결말"]
    nodes = []
    char_states_str = _get_latest_character_states(story)

    # [수정] 한자 사용 금지 및 행동 이유 명시
    sys_prompt = (
        f"당신은 소설가입니다. 시놉시스를 바탕으로 상세한 장면(Scene)들을 생성해야 합니다. "
        f"주인공은 '{protagonist_name}'입니다. "
        "각 장면의 'content'는 반드시 **공백 포함 1000자 이상의 아주 구체적이고 묘사가 풍부한 줄거리**여야 합니다. "
        "문체는 '~한다' 체로 통일하고, **절대 한자를 사용하지 마세요.** "
        "**[중요] 인물이 특정 행동을 할 때는, 그 행동을 하는 심리적/상황적 이유를 명시적으로 서술하여 개연성을 부여하세요.**"
    )
    
    if is_twist_branch:
        sys_prompt += " 특히 마지막 장면은 이야기가 **완벽하게 종결**되어야 합니다. 열린 결말이 아닌 확실한 끝을 맺으세요."

    user_prompt = (
        f"시놉시스: {synopsis}\n"
        f"현재 인물 상태: {char_states_str}\n"
        f"주의: {'이것은 반전(Twist) 이후의 이야기입니다.' if is_twist_branch else '이것은 초기 스토리입니다.'}\n\n"
        "다음 키를 포함한 JSON 리스트로 8개의 장면을 생성하세요:\n"
        "['title', 'description' (여기에 1000자 이상 줄거리), 'setting', 'characters', 'purpose']\n"
        "형식: {'scenes': [...]}"
    )
    
    res = call_llm(sys_prompt, user_prompt, json_format=True)
    
    scenes = res.get('scenes', [])
    target_scenes = scenes[start_node_index:]
    
    for i, scene_data in enumerate(target_scenes):
        current_idx = start_node_index + i
        if current_idx >= 8: break 
        
        phase_name = phases[min(current_idx // 2, 3)]
        
        title = scene_data.get('title', '무제')
        description = scene_data.get('description', '')
        setting = scene_data.get('setting', '')
        purpose = scene_data.get('purpose', '')
        django_content = description

        node = StoryNode.objects.create(story=story, chapter_phase=phase_name, content=django_content)
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
    sys_prompt = (
        f"두 장면을 잇는 선택지를 생성하세요. 주인공 '{protagonist_name}'의 입장에서 취할 수 있는 구체적인 행동이나 대사여야 합니다. "
        "추상적인 표현(예: '갈등의 시작')은 금지입니다. "
        "각 선택지의 결과(result)는 다음 장면의 첫 문장과 자연스럽게 이어지도록 짧은 행동 묘사(~한다)로 작성하세요."
    )
    
    for i in range(len(nodes) - 1):
        curr = nodes[i]
        next_n = nodes[i+1]
        next_n.prev_node = curr
        next_n.save()
        
        user_prompt = (
            f"현재 장면 요약: {curr.content[-500:]}\n"
            f"다음 장면 요약: {next_n.content[:500]}\n\n"
            "형식: {'choices': [{'text': '주인공이 ~한다 (선택지 텍스트)', 'result': '그 결과 ~했다 (행동 묘사)'}]}"
            "선택지는 2개 생성하세요."
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
        "스토리의 장르를 비틀기에 가장 적합한 지점(Index)을 하나 고르세요.", 
        "\n".join(summaries) + "\n출력형식: {'index': 숫자}", 
        json_format=True
    )
    idx = res.get('index', 2)
    if idx < 1: idx = 1
    if idx >= len(nodes) - 2: idx = len(nodes) - 3
    
    nodes[idx].is_twist_point = True
    nodes[idx].save()
    return idx

def _generate_twisted_synopsis_data(story, accumulated, phase):
    # 같은 장르 내 다른 클리셰만 선택
    all_cliches = Cliche.objects.filter(genre=story.main_cliche.genre).exclude(id=story.main_cliche.id).all()
    if not all_cliches.exists():
        all_cliches = Cliche.objects.exclude(id=story.main_cliche.id).all()
    
    if not all_cliches: return None, ""
    
    cliche_info = "\n".join([f"ID {c.id}: {c.title}" for c in all_cliches])
    
    rec_res = call_llm(
        f"반전의 대가입니다. 현재까지의 이야기를 비틀어 **같은 장르({story.main_cliche.genre.name}) 내의 다른 클리셰**로 전환하려 합니다. 가장 적합한 클리셰 ID를 추천하세요.", 
        f"현재까지 줄거리: {accumulated[-2000:]}\n후보 목록: {cliche_info}\n출력: {{'cliche_id': 숫자}}", 
        json_format=True
    )
    try: new_cliche = Cliche.objects.get(id=rec_res['cliche_id'])
    except: new_cliche = all_cliches.first()
    
    # [수정] 한자 사용 금지
    twisted_synopsis = call_llm(
        "소설가입니다. 기존 스토리의 흐름을 유지하다가 급격하게 새로운 클리셰(같은 장르)로 전환되는 시놉시스를 작성하세요. **한자를 사용하지 마세요.**",
        f"지금까지 내용: {accumulated}\n새로운 클리셰: {new_cliche.title} ({new_cliche.summary})\n"
        "조건: 문체는 '~한다'로 통일. 구체적인 줄거리 작성."
    )
    return new_cliche, twisted_synopsis

def _add_twist_branch_choices_only(node, new_next, universe_id, protagonist_name):
    sys_prompt = (
        f"장르적 반전(Twist)이 일어나는 분기점입니다. 주인공 '{protagonist_name}'의 입장에서, "
        "완전히 새로운 전개로 이어지는 파격적인 행동 선택지 2개를 생성하세요. "
        "기존 흐름의 선택지는 이미 존재하므로 생성하지 마세요."
    )
    
    user_prompt = (
        f"현재 상황: {node.content[-500:]}\n"
        f"새로운 반전 다음 장면(Twist Scene): {new_next.content[:500]}\n\n"
        "형식: JSON\n"
        "{\n"
        "  'twist_choices': [\n"
        "    {'text': '주인공이 ~한다 (반전 선택지 1)', 'result': '그 결과 ~했다'},\n"
        "    {'text': '주인공이 ~한다 (반전 선택지 2)', 'result': '그 결과 ~했다'}\n"
        "  ]\n"
        "}"
    )
    
    res = call_llm(sys_prompt, user_prompt, json_format=True)
    
    curr_uid = f"{universe_id}_{node.id}"
    new_next_uid = f"{universe_id}_{new_next.id}"

    for item in res.get('twist_choices', []):
        text = item.get('text', '새로운 운명을 선택한다')
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