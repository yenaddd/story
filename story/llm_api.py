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
    update_universe_details_neo4j, 
    sync_node_to_neo4j, 
    link_universe_to_first_scene, 
    sync_action_to_neo4j, 
    StoryNodeData
)

# API 설정
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
BASE_URL = "https://api.fireworks.ai/inference/v1"
MODEL_NAME = "accounts/fireworks/models/deepseek-v3p1" 
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=BASE_URL)

def call_llm(system_prompt, user_prompt, json_format=False, stream=False, max_tokens=4000, max_retries=3, timeout=120):
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
                timeout=timeout,    
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
    스토리 생성 전체 파이프라인 (Action 기반)
    """
    universe_id = str(uuid.uuid4())
    print(f"\n🌍 [NEO4J] Creating Universe Node: {universe_id}")

    # 1. 설정 구체화 및 주인공 정의
    refined_setting, protagonist_info = _refine_setting_and_protagonist(user_world_setting)
    protagonist_name = protagonist_info['name']
    print(f"✅ Protagonist: {protagonist_name}")

    try:
        create_universe_node_neo4j(universe_id, refined_setting, protagonist_name)
    except Exception as e:
        print(f"Neo4j Error: {e}")

    # 2. 클리셰 매칭 (2단계 로직 적용: 장르 선정 -> 클리셰 선정)
    matched_cliche = _match_cliche(refined_setting)
    if not matched_cliche: raise ValueError("클리셰 매칭 실패")
    
    print(f"✅ Matched Cliche: [{matched_cliche.genre.name}] {matched_cliche.title}")

    story = Story.objects.create(user_world_setting=refined_setting, main_cliche=matched_cliche)
    
    # 3. 시놉시스 생성
    print("  [Step 3] Generating Synopsis...")
    synopsis = _generate_synopsis(story, matched_cliche, protagonist_name, protagonist_info['desc'])
    story.synopsis = synopsis
    story.save()

    # 3.5 주요 인물 정보 추출 및 Universe 업데이트
    print("  [Step 3.5] Extracting Characters & Universe Details...")
    universe_details = _generate_universe_details(refined_setting, synopsis)
    characters_info_json = _extract_characters_info(synopsis, protagonist_info)
    
    try:
        update_universe_details_neo4j(
            universe_id=universe_id,
            synopsis=synopsis,
            twisted_synopsis="",
            title=universe_details.get("title", "무제"),
            description=universe_details.get("description", ""),
            detail_description=universe_details.get("detail_description", ""),
            play_time=universe_details.get("play_time", "30분"),
            characters_info=characters_info_json
        )
    except Exception as e:
        print(f"Neo4j Update Error: {e}")

    # 4 & 5. 초기 노드 생성
    original_nodes = _create_nodes_from_synopsis(
        story, synopsis, protagonist_name, 
        start_node_index=0, 
        universe_id=universe_id,
        is_twist_branch=False
    )
    
    if not original_nodes: raise ValueError("노드 생성 실패")
    
    # Neo4j 연결 (Start)
    try:
        link_universe_to_first_scene(universe_id, f"{universe_id}_{original_nodes[0].id}")
    except: pass

    # 7. 선형 연결 (필수 행동 생성)
    _connect_linear_nodes(original_nodes, universe_id, protagonist_name)

    # 8. 비틀기(Twist) 지점 찾기
    twist_node_index = _find_twist_point_index(original_nodes)
    twist_node = original_nodes[twist_node_index]
    story.twist_point_node_id = twist_node.id
    story.save()
    
    accumulated_content = "\n".join([n.content for n in original_nodes[:twist_node_index+1]])
    
    print("  [Step 9] Generating Twisted Synopsis...")
    twisted_synopsis = _generate_twisted_synopsis_data(
        story, accumulated_content, twist_node.chapter_phase, protagonist_name, protagonist_info['desc']
    )
    story.twisted_synopsis = twisted_synopsis
    story.save()

    # Universe에 Twist 시놉시스 업데이트
    try:
        update_universe_details_neo4j(
            universe_id=universe_id,
            synopsis=story.synopsis,
            twisted_synopsis=twisted_synopsis,
            title=universe_details.get("title"),
            description=universe_details.get("description"),
            detail_description=universe_details.get("detail_description"),
            play_time=universe_details.get("play_time"),
            characters_info=characters_info_json
        )
    except: pass
    
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
        original_choice = twist_node.choices.first()
        original_action_text = original_choice.choice_text if original_choice else "다음으로 진행"

        _create_twist_condition(
            twist_node, 
            twist_next_node, 
            universe_id, 
            protagonist_name, 
            original_action_text
        )

    # 13. 새 브랜치 내부 연결
    _connect_linear_nodes(new_branch_nodes, universe_id, protagonist_name)

    return story.id

# ==========================================
# [내부 로직: 클리셰 매칭 개선]
# ==========================================

def _match_cliche(setting):
    """
    [2단계 매칭 로직]
    1. Genre Selection: 유저 설정에 가장 적합한 장르 1개 선정
    2. Cliche Selection: 선정된 장르 내에서 가장 적합한 클리셰 1개 선정
    """
    
    # [Step 1] 장르 선정
    all_genres = Genre.objects.all()
    if not all_genres.exists():
        return None
    
    # 장르 설명 텍스트 구성
    genre_text_list = []
    for g in all_genres:
        desc = g.description if g.description else "설명 없음"
        genre_text_list.append(f"- {g.name}: {desc}")
    genre_prompt_text = "\n".join(genre_text_list)
    
    sys_prompt_1 = (
        "당신은 장르 문학 분석가입니다. "
        "사용자의 입력(세계관 설정)을 분석하여, 아래 제공된 장르 목록 중 이를 가장 효과적으로 표현할 수 있는 **단 하나의 장르**를 선택하세요.\n"
        "반드시 JSON 형식 {'genre_name': '장르명', 'reason': '이유'} 으로만 응답하세요."
    )
    user_prompt_1 = f"사용자 설정: {setting}\n\n[장르 목록]\n{genre_prompt_text}"
    
    print("  [Step 2-1] Selecting Genre...")
    res_1 = call_llm(sys_prompt_1, user_prompt_1, json_format=True)
    selected_genre_name = res_1.get('genre_name', '판타지') # 기본값 판타지
    
    try:
        selected_genre = Genre.objects.get(name=selected_genre_name)
    except Genre.DoesNotExist:
        # LLM이 없는 이름을 뱉었을 경우 가장 유사하거나 첫 번째 장르 선택
        selected_genre = all_genres.first()
        print(f"  ⚠️ Genre '{selected_genre_name}' not found. Fallback to '{selected_genre.name}'")

    print(f"  -> Selected Genre: {selected_genre.name}")

    # [Step 2] 클리셰 선정
    cliches = Cliche.objects.filter(genre=selected_genre)
    if not cliches.exists():
        # 해당 장르에 클리셰가 없으면 전체에서 랜덤
        return Cliche.objects.first()

    # 클리셰 상세 정보 구성 (정의 및 구조)
    cliche_text_list = []
    for c in cliches:
        info = (
            f"ID: {c.id}\n"
            f"제목: {c.title}\n"
            f"정의: {c.summary}\n"
            f"구조 가이드: {c.structure_guide}\n"
        )
        cliche_text_list.append(info)
    cliche_prompt_text = "\n----------------\n".join(cliche_text_list)

    sys_prompt_2 = (
        f"당신은 '{selected_genre.name}' 장르 전문 편집자입니다. "
        "사용자의 설정과 앞서 선정된 장르를 고려하여, 해당 장르 내의 클리셰 목록 중 **가장 흥미롭고 극적인 전개가 가능한 클리셰** 하나를 선택하세요.\n"
        "각 클리셰의 정의와 구조(structure_guide)를 면밀히 분석하여 결정하세요.\n"
        "응답은 JSON 형식 {'cliche_id': ID숫자, 'reason': '선택 이유'} 만 반환하세요."
    )
    user_prompt_2 = f"사용자 설정: {setting}\n\n[선택된 장르: {selected_genre.name}]\n\n[클리셰 후보 목록]\n{cliche_prompt_text}"

    print("  [Step 2-2] Selecting Cliche...")
    res_2 = call_llm(sys_prompt_2, user_prompt_2, json_format=True)
    
    try:
        selected_id = res_2.get('cliche_id')
        if not selected_id: raise ValueError("No ID returned")
        final_cliche = Cliche.objects.get(id=selected_id)
        print(f"  -> Selected Cliche: {final_cliche.title} (Reason: {res_2.get('reason')})")
        return final_cliche
    except Exception as e:
        print(f"  ⚠️ Cliche Selection Error: {e}. Fallback to random in genre.")
        return random.choice(list(cliches))

# ==========================================
# [나머지 내부 로직 함수들 (기존 유지)]
# ==========================================

def _refine_setting_and_protagonist(raw_setting):
    sys_prompt = "세계관과 주인공을 정의하세요. 주인공 이름은 한글, 성격/믿음/사상/외모를 포함해야 합니다."
    user_prompt = (
        f"입력: {raw_setting}\n"
        "출력 JSON: {'refined_setting': '...', 'protagonist': {'name': '...', 'desc': '성격, 믿음, 사상, 외모 포함 상세 묘사'}}"
    )
    res = call_llm(sys_prompt, user_prompt, json_format=True)
    return res.get('refined_setting', raw_setting), res.get('protagonist', {'name':'이안', 'desc':'평범함'})

def _generate_synopsis(story, cliche, p_name, p_desc):
    sys_prompt = (
        "당신은 베스트셀러 웹소설 작가입니다. "
        "주어진 세계관 설정과 **지정된 필수 클리셰**를 완벽하게 조합하여 매력적인 시놉시스를 작성하세요.\n"
        "1. 분량은 2000자 이상.\n"
        "2. 기승전결 구조와 주인공의 내면 변화 포함.\n"
        "3. **선택된 클리셰의 '핵심 요약'과 '전개 가이드'를 충실히 따를 것.**"
    )
    
    cliche_detail = (
        f"제목: {cliche.title}\n"
        f"장르: {cliche.genre.name}\n"
        f"핵심 요약: {cliche.summary}\n"
        f"전개 가이드: {cliche.structure_guide}"
    )
    
    user_prompt = (
        f"세계관 설정: {story.user_world_setting}\n"
        f"주인공: {p_name} ({p_desc})\n"
        f"----------------------------------------\n"
        f"★ 필수 적용 클리셰 정보 ★\n{cliche_detail}\n"
        f"----------------------------------------\n"
        "위 내용을 바탕으로 전체 시놉시스를 작성해줘."
    )
    
    return call_llm(sys_prompt, user_prompt, stream=True, max_tokens=8000)

def _extract_characters_info(synopsis, protagonist_info):
    sys_prompt = "시놉시스에 등장하는 주요 인물들의 이름과 '성격, 믿음, 사상, 외모'를 분석하여 JSON 리스트로 추출하세요."
    user_prompt = f"시놉시스: {synopsis[:3000]}..."
    res = call_llm(sys_prompt, user_prompt, json_format=True)
    
    chars = res.get('characters', [])
    if not any(c.get('name') == protagonist_info['name'] for c in chars):
        chars.insert(0, protagonist_info)
        
    return json.dumps(chars, ensure_ascii=False)

def _create_nodes_from_synopsis(story, synopsis, protagonist_name, start_node_index=0, is_twist_branch=False, universe_id=None):
    needed_nodes = 12 - start_node_index
    phases = ["발단", "전개", "절정", "결말"]
    
    sys_prompt = (
        f"당신은 인터랙티브 스토리 작가입니다. 주인공 '{protagonist_name}'의 시점에서 장면(Node)들을 생성하세요.\n"
        "각 장면은 title, description(500자 이상), setting, purpose, characters_list, character_states, character_changes를 포함해야 합니다."
    )
    user_prompt = f"시놉시스: {synopsis}\n개수: {needed_nodes}개\nJSON 형식: {{'scenes': [...]}}"
    
    res = call_llm(sys_prompt, user_prompt, json_format=True, stream=True, max_tokens=8000)
    scenes = res.get('scenes', [])
    
    nodes = []
    for i, scene_data in enumerate(scenes):
        current_idx = start_node_index + i
        phase_name = phases[min(current_idx // 3, 3)]
        
        node = StoryNode.objects.create(
            story=story, 
            chapter_phase=phase_name, 
            content=scene_data.get('description', '')
        )
        nodes.append(node)
        
        if universe_id:
            try:
                neo4j_data = StoryNodeData(
                    node_id=f"{universe_id}_{node.id}",
                    phase=phase_name,
                    title=scene_data.get('title', '무제'),
                    setting=scene_data.get('setting', ''),
                    description=scene_data.get('description', ''),
                    purpose=scene_data.get('purpose', ''),
                    characters_list=scene_data.get('characters_list', []),
                    character_states=json.dumps(scene_data.get('character_states', {}), ensure_ascii=False),
                    character_changes=json.dumps(scene_data.get('character_changes', {}), ensure_ascii=False),
                    depth=current_idx
                )
                sync_node_to_neo4j(neo4j_data)
            except Exception as e:
                print(f"Neo4j Node Sync Error: {e}")
    return nodes

def _connect_linear_nodes(nodes, universe_id, protagonist_name):
    sys_prompt = (
       f"주인공 '{protagonist_name}'이 현재 장면에서 다음 장면으로 넘어가기 위해 취해야 할 **가장 자연스럽고 일상적인 행동(Condition Action)**을 정의하세요.\n"
        "이 게임은 유저가 선택지를 고르는 것이 아니라, 채팅창에 직접 행동을 입력하는 방식입니다.\n"
        "따라서 유저가 **별도의 힌트 없이도 상황상 자연스럽게 입력할 법한 행동**(예: '문을 연다', '대답한다', '전화를 받는다')이어야 합니다.\n"
        "너무 구체적이거나 맞추기 어려운 행동은 피하고, 다음 스토리로 자연스럽게 흘러가는 연결 고리 역할을 해야 합니다." 
    )
    
    for i in range(len(nodes) - 1):
        curr = nodes[i]
        next_n = nodes[i+1]
        
        curr.prev_node = next_n.prev_node 
        next_n.prev_node = curr
        next_n.save()
        
        user_prompt = (
            f"현재 장면 요약: {curr.content[-300:]}\n"
            f"다음 장면 요약: {next_n.content[:300]}\n\n"
            "출력 JSON: {'action': '유저가 채팅으로 입력할 법한 자연스러운 행동', 'result': '행동의 결과(다음 장면 도입부)'}"
        )
        
        res = call_llm(sys_prompt, user_prompt, json_format=True)
        action_text = res.get('action', '다음으로 이동')
        result_text = res.get('result', '')
        
        NodeChoice.objects.create(
            current_node=curr,
            choice_text=action_text,
            result_text=result_text,
            next_node=next_n,
            is_twist_path=False
        )
        
        if universe_id:
            try:
                sync_action_to_neo4j(
                    f"{universe_id}_{curr.id}", 
                    f"{universe_id}_{next_n.id}", 
                    action_text, 
                    result_text, 
                    is_twist=False
                )
            except: pass

def _find_twist_point_index(nodes):
    if len(nodes) < 4: return 1
    summaries = [f"Idx {i}: {n.content[:50]}..." for i, n in enumerate(nodes[:-2])]
    res = call_llm("비틀기 지점(Index) 선택", "\n".join(summaries), json_format=True)
    idx = res.get('index', 2)
    return max(1, min(idx, len(nodes)-3))

def _generate_twisted_synopsis_data(story, acc_content, phase, p_name, p_desc):
    sys_prompt = "반전(Twist) 시놉시스 생성. 2000자 이상."
    user_prompt = f"현재까지: {acc_content[-1000:]}\n주인공: {p_name}\n단계: {phase} 이후 변주"
    return call_llm(sys_prompt, user_prompt, stream=True, max_tokens=8000)

def _create_twist_condition(node, twist_next_node, universe_id, protagonist_name, original_action_text):
    sys_prompt = (
        f"현재 장면에서 이야기가 완전히 다른 방향(반전)으로 흐르기 위해, "
        f"주인공 '{protagonist_name}'이 수행해야 할 **돌발적이고 파격적인 조건 행동(Twist Action)**을 하나 정의하세요.\n"
        "이 게임은 유저가 채팅창에 직접 행동을 입력하는 방식입니다.\n"
        f"**중요: 기존 스토리로 이어지는 정석적인 행동은 '{original_action_text}'입니다.**\n"
        f"**반전 행동은 이 '원래 행동'과 의도나 방식이 명확히 달라야 합니다.**\n"
        "하지만 유저가 호기심이나 반항심에 시도해볼 법한, 입력 가능한 수준의 행동(예: '거절한다', '공격한다', '무시한다')이어야 합니다."
    )
    
    user_prompt = (
        f"현재 장면: {node.content[-300:]}\n"
        f"반전 장면(다음): {twist_next_node.content[:300]}\n"
        f"참고(기존 행동): {original_action_text}\n"
        "출력 JSON: {'action': '반전 행동', 'result': '행동의 결과'}"
    )
    
    res = call_llm(sys_prompt, user_prompt, json_format=True)
    action_text = res.get('action', '운명을 바꾸는 선택을 한다')
    result_text = res.get('result', '')
    
    NodeChoice.objects.create(
        current_node=node,
        choice_text=action_text,
        result_text=result_text,
        next_node=twist_next_node,
        is_twist_path=True 
    )
    
    if universe_id:
        try:
            sync_action_to_neo4j(
                f"{universe_id}_{node.id}", 
                f"{universe_id}_{twist_next_node.id}", 
                action_text, 
                result_text, 
                is_twist=True
            )
        except: pass

def _generate_universe_details(setting, synopsis):
    sys_prompt = "세계관 상세 정보 JSON 생성 (title, description, detail_description, play_time)"
    user_prompt = f"설정: {setting}\n줄거리: {synopsis[:500]}..."
    return call_llm(sys_prompt, user_prompt, json_format=True)