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
    update_universe_node_neo4j,
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

def call_llm(system_prompt, user_prompt, json_format=False, stream=False, max_tokens=4000, max_retries=3):
    """
    LLM 호출 함수
    """
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    response_format = {"type": "json_object"} if json_format else None
    
    if not DEEPSEEK_API_KEY:
        print("🚨 [Critical] API Key is MISSING!")
        return {} if json_format else ""

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME, 
                messages=messages, 
                response_format=response_format, 
                temperature=0.7, 
                max_tokens=max_tokens, 
                timeout=120,    
                stream=stream 
            )
            
            content = ""
            if stream:
                print(f"  [LLM] Streaming generating (Max Tokens: {max_tokens})...", end="", flush=True)
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        content += chunk.choices[0].delta.content
                print(" Done.")
            else:
                content = response.choices[0].message.content

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

    # 1. 설정 구체화 및 주인공 정의 (주인공 특징 'desc' 추가 반환)
    refined_setting, protagonist_name, protagonist_desc = _refine_setting_and_protagonist(user_world_setting)
    print(f"✅ Refined Setting: {refined_setting[:50]}... / Protagonist: {protagonist_name}")

    try:
        create_universe_node_neo4j(universe_id, refined_setting)
    except Exception as e:
        print(f"Neo4j Error: {e}")

    # 2. 클리셰 매칭
    matched_cliche = _match_cliche(refined_setting)
    if not matched_cliche:
        raise ValueError("적절한 클리셰를 찾지 못했습니다.")
    print(f"✅ Matched Cliche: {matched_cliche.title}")
    
    story = Story.objects.create(user_world_setting=refined_setting, main_cliche=matched_cliche)
    
    # 3. 시놉시스 생성 (주인공 특징 'protagonist_desc' 전달)
    print("  [Step 3] Generating Massive Synopsis (Streaming)...")
    synopsis = _generate_synopsis(story, matched_cliche, protagonist_name, protagonist_desc)
    story.synopsis = synopsis
    story.save()

    try:
        update_universe_node_neo4j(universe_id, protagonist_name, protagonist_desc, synopsis)
    except Exception as e:
        print(f"Neo4j Update Error: {e}")
        
    # 4. 인물 내면 상태 분석
    _analyze_and_save_character_state(story, synopsis, context="Initial Synopsis")

    # 5 & 6. 초기 노드 생성
    original_nodes = _create_nodes_from_synopsis(
        story, synopsis, protagonist_name, 
        start_node_index=0, 
        universe_id=universe_id,
        is_twist_branch=False
    )
    
    if not original_nodes or len(original_nodes) < 2:
        print("❌ [Error] 노드 생성 실패.")
        raise ValueError("AI가 스토리 노드를 생성하지 못했습니다.")  
    
    # Neo4j 연결
    if original_nodes:
        try:
            first_node_uid = f"{universe_id}_{original_nodes[0].id}"
            link_universe_to_first_scene(universe_id, first_node_uid)
        except Exception as e:
            print(f"Neo4j Link Error: {e}")

    # 7. 선형 연결
    _connect_linear_nodes(original_nodes, universe_id, protagonist_name)

    # 8. 비틀기(Twist) 지점 찾기
    twist_node_index = _find_twist_point_index(original_nodes)
    
    if twist_node_index >= len(original_nodes) - 1: 
        twist_node_index = len(original_nodes) - 2
    if twist_node_index < 0: twist_node_index = 0
        
    twist_node = original_nodes[twist_node_index]
    story.twist_point_node_id = twist_node.id
    story.save()

    accumulated_content = "\n".join([n.content for n in original_nodes[:twist_node_index+1]])
    
    # 9. 비틀린 시놉시스 생성
    twisted_synopsis = _generate_twisted_synopsis_data(story, accumulated_content, twist_node.chapter_phase)
    
    story.twisted_synopsis = twisted_synopsis
    story.save()
    
    # 10. 비틀린 시놉시스 기반 내면 분석
    _analyze_and_save_character_state(story, twisted_synopsis, context="Twisted Synopsis")

    # 11. 비틀기 노드 생성
    new_branch_nodes = _create_nodes_from_synopsis(
        story, twisted_synopsis, protagonist_name,
        start_node_index=twist_node_index+1, 
        is_twist_branch=True, 
        universe_id=universe_id
    )

    # 12. 분기 처리
    if new_branch_nodes:
        twist_next_node = new_branch_nodes[0]
        _add_twist_branch_choices_only(twist_node, twist_next_node, universe_id, protagonist_name)

    # 13. 새 브랜치 내부 연결
    _connect_linear_nodes(new_branch_nodes, universe_id, protagonist_name)

    return story.id

# ==========================================
# [내부 로직 함수들]
# ==========================================

def _refine_setting_and_protagonist(raw_setting):
    # [수정] 상세한 세계관 설정 가이드 적용
    sys_prompt = (
        "당신은 창의적인 스토리 작가입니다. 사용자의 입력을 분석하여 세계관을 확정하고 주인공을 정의하세요. "
        "**[필수 규칙]**\n"
        "1. 사용자가 주인공의 이름을 지정하지 않았다면, 세계관에 어울리는 멋진 이름을 반드시 창작하세요.\n"
        "2. **모든 인물의 이름은 반드시 '한글'로 표기해야 합니다.** (예: Arthur -> 아서, Jane -> 제인)\n"
        "3. 절대 '주인공', '나', '행인1' 같은 대명사나 성의 없는 이름을 사용하지 마세요."
    )
    user_prompt = (
        f"사용자 입력: {raw_setting}\n\n"
        "다음 형식의 JSON으로 출력하세요:\n"
        "{\n"
        "  'refined_setting': '확정된 구체적인 세계관 및 배경 설정. 사용자 입력 내용과 가장 관련이 깊은 설정을 중요 세계관설정으로 두고, 나머지는 그냥 세계관 설정. 다음 내용이 서로 모순을 일으키지 않고 조화롭게 하나의 세계관, 배경을 이루도록 창작할 것. [신념]: 어떤 사상이나 진리를 진실로 받아들이는 심리적 태도입니다. [가치와 윤리]: 옳고 그름, 좋음과 나쁨을 판단하는 기준이며, 도덕적, 윤리적 규범을 포함합니다. [인간 본질]: 인간이 무엇이며, 어떻게 존재해야 하는지에 대한 생각입니다. [우주와 기원]: 세계가 어떻게 시작되었고, 어떻게 작동하는지에 대한 근본적인 질문에 대한 답입니다. [사회 구조]: 가족, 공동체, 정치 시스템 등 사회를 구성하는 방식에 대한 생각입니다. [지정학]: 국가, 영토, 국경, 지리적 특징 등. [경제]: 재화, 돈, 무역 등 경제 시스템. [역사]: 과거 사건들이 현재에 미치는 영향. [종교]: 신앙 체계와 문화. [생태계]: 자연 환경, 자원, 생물권 등 (텍스트)',\n"
        "  'protagonist_name': '확정되거나 창작된 주인공 이름 (한글 표기 필수)',\n"
        "  'protagonist_desc': '주인공의 성격, 외모, 믿음, 성향, 사상'\n"
        "}"
    )
    res = call_llm(sys_prompt, user_prompt, json_format=True) 
    
    setting = res.get('refined_setting', raw_setting)
    name = res.get('protagonist_name', '') 
    desc = res.get('protagonist_desc', '') # 주인공 특징 추가 확보
    
    if not name or name.strip() in ["주인공", "나", "Unknown", "미정"]:
        # [수정] 주인공 이름 생성 프롬프트 업데이트
        name_res = call_llm("이 세계관 내용 중 중요 세계관 설정 내용과 어울리는 주인공 이름을 1개만 한글로 지어줘.", f"세계관: {setting}")
        name = name_res.strip().replace("이름:", "").replace(".", "")
        if not name: name = "이안"
        
    return setting, name, desc

def _match_cliche(setting):
    all_cliches = Cliche.objects.select_related('genre').all()
    if not all_cliches.exists(): return None
    
    cliche_list = list(all_cliches)
    random.shuffle(cliche_list)
    
    cliche_info = "\n".join([f"ID {c.id}: [{c.genre.name}] {c.title} - {c.summary}" for c in cliche_list])
    
    sys_prompt = (
        "사용자의 설정과 가장 잘 어울리는 클리셰(Cliche)를 하나 선택하세요. "
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
        print("⚠️ [Warning] 클리셰 매칭 실패. 랜덤으로 선택합니다.")
        return random.choice(all_cliches)

def _generate_synopsis(story, cliche, protagonist_name, protagonist_desc):
    # [수정] protagonist_desc 추가 반영
    sys_prompt = (
        "당신은 대서사시를 집필하는 메인 시나리오 작가입니다. "
        "사용자의 설정과 클리셰를 결합하여, **공백 포함 최소 2000자 이상의 매우 상세하고 긴 시놉시스**를 작성해야 합니다.\n\n"
        "**[필수 작성 가이드]**\n"
        "1. **분량**: 반드시 2000자를 넘기세요. 사건을 단순히 요약하지 말고, 장면 묘사, 대사, 분위기를 포함해 구체적으로 서술하세요.\n"
        "2. **구성 요소** (다음 항목들을 반드시 포함하여 순서대로 작성하세요):\n"
        "   - **[1. 주요 등장인물 소개]**: 주인공('{protagonist_name}')과 주변 인물들의 성격, 외모, 배경, 욕망을 상세히 기술. **(모든 이름은 한글로 표기)**\n"
        "   - **[2. 전체 줄거리 (기승전결)]**: 발단-전개-절정-결말의 흐름으로 사건을 서술.\n"
        "   - **[3. 인물 내면 변화 보고서]**: 스토리가 진행됨에 따라 주인공과 주요 인물이 겪는 **감정(Emotion), 신뢰(Trust), 사상(Ideology)의 변화 과정**을 단계별로 분석하여 기술.\n\n"
        "3. **감정선**: 클리셰의 전형적인 흐름을 따르되, 사용자의 설정(세계관)을 반영하여 독창적인 디테일을 추가하세요."
    )
    
    user_prompt = (
        f"세계관: {story.user_world_setting}\n"
        f"주인공: {protagonist_name}\n"
        f"주인공 특징: {protagonist_desc}\n" # [추가] 상세 특징 전달
        f"적용 클리셰: {cliche.title} ({cliche.summary})\n"
        f"클리셰 가이드: {cliche.structure_guide}\n"
        f"참고 작품 감정선: {cliche.example_work_summary}\n\n"
        "위 정보를 바탕으로 대규모 시놉시스를 작성하세요."
    )
    
    return call_llm(sys_prompt, user_prompt, stream=True, max_tokens=8000)

def _analyze_and_save_character_state(story, text, context):
    sys_prompt = (
        "텍스트를 심층 분석하여 등장인물들의 내면 상태 변화를 추출하세요. "
        "특히 시놉시스에 명시된 '인물 내면 변화' 부분을 중점적으로 참고하세요."
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
    phases = ["발단", "전개", "절정", "결말"]
    nodes = []
    char_states_str = _get_latest_character_states(story)

    sys_prompt = (
        f"당신은 인터랙티브 스토리 게임의 작가입니다. 시놉시스를 바탕으로 플레이어가 진행할 구체적인 장면(Node)들을 생성하세요. "
        f"주인공은 '{protagonist_name}'입니다.\n"
        "**[필수 제약 사항]**\n"
        "1. 각 장면의 내용은 **공백 포함 최소 500자 이상**으로 아주 상세하고 몰입감 있게 작성해야 합니다.\n"
        "2. **제공된 인물 내면 상태(Character State)를 반드시 반영**하여, 인물의 말과 행동이 내면과 일치하고 개연성을 가지도록 하세요.\n"
        "3. 문체는 서술형(~한다)을 사용하세요.\n"
        "4. **[매우 중요] 스토리의 마지막 장면(결말 단계)에서는 이야기가 확실하고 완전하게 종결되어야 합니다.** "
        "후속작을 암시하거나, '우리의 모험은 계속된다' 식의 열린 결말로 끝내지 마세요. 모든 갈등이 해소되고 상황이 종료된 명확한 엔딩을 쓰세요."
    )
    
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
                
                # Neo4j 전송 시 글자수 제한 해제 (description 전체 전송)
                neo4j_data = StoryNodeData(
                    node_id=neo4j_node_uid,
                    phase=phase_name,
                    title=title,
                    setting=setting,
                    characters=characters_str, 
                    description=description, 
                    purpose=str(purpose),
                    character_state=char_states_str
                )
                sync_node_to_neo4j(neo4j_data)
            except Exception as e:
                print(f"Neo4j Node Sync Error: {e}")

    return nodes

def _connect_linear_nodes(nodes, universe_id, protagonist_name):
    sys_prompt = (
        f"현재 장면에서 다음 장면으로 넘어가기 위한 선택지 2개를 생성하세요. 주인공 '{protagonist_name}'의 입장이 되어야 합니다.\n"
        "**[필수 조건]**\n"
        "1. **같은 상황(Scene)에 대한 서로 다른 행동**이어야 합니다.\n"
        f"2. **선택지 텍스트('text')에는 '주인공'이라는 단어를 절대 쓰지 말고, 주인공의 이름 '{protagonist_name}'을 사용하세요.**\n"
        "3. 'result'(결과)는 선택지 행동으로 인해 발생한 직후 결과를 묘사하는 **완결된 문장**이어야 합니다.\n"
        "4. 다음 장면의 내용 자체는 바뀌지 않으므로, 결과 텍스트는 다음 장면의 첫 부분과 자연스럽게 이어져야 합니다.\n"
        "5. 다음 장면의 첫 부분과 자연스럽게 이어지게 하기 위해 한 문장의 서술로는 부족하다면, 두 문장으로 서술해도 좋습니다."
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
        "새로운 클리셰를 도입하지 말고, 현재 클리셰 안에서 사건의 해석을 달리하거나 돌발 변수를 추가하여 결말을 바꾸세요.\n"
        "**주의: 생성된 시놉시스의 결말은 반드시 명확하게 매듭지어져야 합니다.**"
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