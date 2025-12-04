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
    update_universe_details_neo4j, # 함수명 변경
    sync_node_to_neo4j, 
    link_universe_to_first_scene, 
    sync_action_to_neo4j, # 함수명 변경
    StoryNodeData
)

# API 설정 (기존과 동일)
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
BASE_URL = "https://api.fireworks.ai/inference/v1"
MODEL_NAME = "accounts/fireworks/models/deepseek-v3p1" 
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=BASE_URL)

def call_llm(system_prompt, user_prompt, json_format=False, stream=False, max_tokens=4000, max_retries=3, timeout=120):
    # (기존 코드와 동일)
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

    # 1. 설정 구체화 및 주인공 정의 (상세 정보 포함)
    refined_setting, protagonist_info = _refine_setting_and_protagonist(user_world_setting)
    protagonist_name = protagonist_info['name']
    print(f"✅ Protagonist: {protagonist_name}")

    try:
        # Universe 생성 (이미지 필드 빈값 포함)
        create_universe_node_neo4j(universe_id, refined_setting, protagonist_name)
    except Exception as e:
        print(f"Neo4j Error: {e}")

    # 2. 클리셰 매칭
    matched_cliche = _match_cliche(refined_setting)
    if not matched_cliche: raise ValueError("클리셰 매칭 실패")
    
    story = Story.objects.create(user_world_setting=refined_setting, main_cliche=matched_cliche)
    
    # 3. 시놉시스 생성
    print("  [Step 3] Generating Synopsis...")
    synopsis = _generate_synopsis(story, matched_cliche, protagonist_name, protagonist_info['desc'])
    story.synopsis = synopsis
    story.save()

    # 3.5 주요 인물 정보 추출 및 Universe 업데이트
    print("  [Step 3.5] Extracting Characters & Universe Details...")
    universe_details = _generate_universe_details(refined_setting, synopsis)
    
    # 인물 정보 통합 (주인공 + 시놉시스 등장인물)
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

    # 8. 비틀기(Twist)
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
            twisted_synopsis=twisted_synopsis, # 업데이트
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

    # 12. 분기 처리 (기존 루트 vs 반전 루트)
    # 기존 선형 노드(original_nodes)의 다음 노드는 이미 _connect_linear_nodes에서 연결됨.
    # 여기서는 '반전 노드'로 가는 '필수 행동'을 추가하여 분기점을 만듦.
    if new_branch_nodes:
        twist_next_node = new_branch_nodes[0]
        # 분기점 노드에서 반전 노드로 가는 필수 행동 생성
        _create_twist_condition(twist_node, twist_next_node, universe_id, protagonist_name)

    # 13. 새 브랜치 내부 연결
    _connect_linear_nodes(new_branch_nodes, universe_id, protagonist_name)

    return story.id

# ==========================================
# [내부 로직 수정]
# ==========================================

def _refine_setting_and_protagonist(raw_setting):
    # (기존과 유사하나 출력 형식 보강)
    sys_prompt = "세계관과 주인공을 정의하세요. 주인공 이름은 한글, 성격/믿음/사상/외모를 포함해야 합니다."
    user_prompt = (
        f"입력: {raw_setting}\n"
        "출력 JSON: {'refined_setting': '...', 'protagonist': {'name': '...', 'desc': '성격, 믿음, 사상, 외모 포함 상세 묘사'}}"
    )
    res = call_llm(sys_prompt, user_prompt, json_format=True)
    return res.get('refined_setting', raw_setting), res.get('protagonist', {'name':'이안', 'desc':'평범함'})

def _match_cliche(setting):
    # (기존 코드 유지)
    all_cliches = Cliche.objects.select_related('genre').all()
    if not all_cliches.exists(): return None
    cliche_list = list(all_cliches)
    random.shuffle(cliche_list)
    cliche_info = "\n".join([f"ID {c.id}: [{c.genre.name}] {c.title}" for c in cliche_list])
    res = call_llm("클리셰 선택 JSON", f"설정: {setting}\n목록:\n{cliche_info}", json_format=True)
    try: return Cliche.objects.get(id=res['cliche_id'])
    except: return random.choice(all_cliches)

def _generate_synopsis(story, cliche, p_name, p_desc):
    # (기존 코드 유지 - 프롬프트만 약간 최적화 가정)
    sys_prompt = "2000자 이상의 상세 시놉시스 작성. 기승전결, 인물 내면 변화 포함."
    user_prompt = f"설정: {story.user_world_setting}, 주인공: {p_name}({p_desc}), 클리셰: {cliche.title}"
    return call_llm(sys_prompt, user_prompt, stream=True, max_tokens=8000)

def _extract_characters_info(synopsis, protagonist_info):
    """
    시놉시스에서 주요 인물 정보를 추출하고 주인공 정보와 합침 (JSON String 반환)
    Neo4j '주요 인물 정보' 필드용.
    """
    sys_prompt = "시놉시스에 등장하는 주요 인물들의 이름과 '성격, 믿음, 사상, 외모'를 분석하여 JSON 리스트로 추출하세요."
    user_prompt = f"시놉시스: {synopsis[:3000]}..."
    res = call_llm(sys_prompt, user_prompt, json_format=True)
    
    chars = res.get('characters', [])
    # 주인공 정보가 없으면 추가
    if not any(c.get('name') == protagonist_info['name'] for c in chars):
        chars.insert(0, protagonist_info)
        
    return json.dumps(chars, ensure_ascii=False)

def _create_nodes_from_synopsis(story, synopsis, protagonist_name, start_node_index=0, is_twist_branch=False, universe_id=None):
    # 노드 생성 로직. 
    # **중요**: Neo4j 요청 스펙인 '등장인물 리스트, 상태, 변화'를 함께 생성해야 함.
    
    needed_nodes = 12 - start_node_index
    phases = ["발단", "전개", "절정", "결말"]
    
    sys_prompt = (
        f"당신은 인터랙티브 스토리 작가입니다. 주인공 '{protagonist_name}'의 시점에서 장면(Node)들을 생성하세요.\n"
        "각 장면은 다음 정보를 포함해야 합니다:\n"
        "1. title: 장면 제목\n"
        "2. description: 장면 줄거리 (500자 이상)\n"
        "3. setting: 장면 배경 설명\n"
        "4. purpose: 장면의 목적\n"
        "5. characters_list: 이 장면에 등장하는 인물 이름 리스트\n"
        "6. character_states: {이름: {감정:..., 생각:..., 관계:..., 고민:...}} 형태의 상태 정보\n"
        "7. character_changes: {이름: 전 장면 대비 변화 내용} 형태의 정보\n"
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
                # Neo4j 데이터 클래스 매핑
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
    """
    선형 연결: 현재 노드 -> 다음 노드
    **변경**: 선택지 2개가 아니라, 다음 노드로 가기 위한 '필수 행동' 1개를 생성.
    """
    sys_prompt = (
        f"주인공 '{protagonist_name}'이 현재 장면에서 다음 장면으로 넘어가기 위해 수행해야 할 "
        "**단 하나의 필수적인 행동(Condition Action)**을 정의하세요.\n"
        "행동은 구체적인 대사나 지문보다는, '무엇을 한다', '어디로 간다'처럼 추상적이고 명확한 지시문이어야 합니다."
    )
    
    for i in range(len(nodes) - 1):
        curr = nodes[i]
        next_n = nodes[i+1]
        
        curr.prev_node = next_n.prev_node # LinkedList 유지
        next_n.prev_node = curr
        next_n.save()
        
        user_prompt = (
            f"현재 장면 요약: {curr.content[-200:]}\n"
            f"다음 장면 요약: {next_n.content[:200]}\n\n"
            "출력 JSON: {'action': '주인공이 해야 할 행동', 'result': '행동의 결과(다음 장면 도입부)'}"
        )
        
        res = call_llm(sys_prompt, user_prompt, json_format=True)
        action_text = res.get('action', '다음으로 이동')
        result_text = res.get('result', '')
        
        # NodeChoice 모델을 사용하여 연결 정보 저장 (이름은 Choice지만 의미는 Action)
        NodeChoice.objects.create(
            current_node=curr,
            action_text=action_text,
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
    # (기존 코드 유지)
    if len(nodes) < 4: return 1
    summaries = [f"Idx {i}: {n.content[:50]}..." for i, n in enumerate(nodes[:-2])]
    res = call_llm("비틀기 지점(Index) 선택", "\n".join(summaries), json_format=True)
    idx = res.get('index', 2)
    return max(1, min(idx, len(nodes)-3))

def _generate_twisted_synopsis_data(story, acc_content, phase, p_name, p_desc):
    # (기존 코드 유지)
    sys_prompt = "반전(Twist) 시놉시스 생성. 2000자 이상."
    user_prompt = f"현재까지: {acc_content[-1000:]}\n주인공: {p_name}\n단계: {phase} 이후 변주"
    return call_llm(sys_prompt, user_prompt, stream=True, max_tokens=8000)

def _create_twist_condition(node, twist_next_node, universe_id, protagonist_name):
    """
    분기점 처리:
    이미 선형적인 '필수 행동'이 하나 존재함 (기존 스토리로 가는 길).
    여기에 '반전 스토리'로 가는 '또 다른 필수 행동'을 추가함.
    """
    sys_prompt = (
        f"현재 장면에서 이야기가 완전히 다른 방향(반전)으로 흐르기 위해, "
        f"주인공 '{protagonist_name}'이 수행해야 할 **돌발적이고 파격적인 필수 행동**을 하나 정의하세요.\n"
        "이 행동을 하면 기존 스토리와 다른 '반전 루트'로 진입합니다."
    )
    
    user_prompt = (
        f"현재 장면: {node.content[-200:]}\n"
        f"반전 장면(다음): {twist_next_node.content[:200]}\n"
        "출력 JSON: {'action': '반전 행동', 'result': '행동의 결과'}"
    )
    
    res = call_llm(sys_prompt, user_prompt, json_format=True)
    action_text = res.get('action', '운명을 바꾸는 선택을 한다')
    result_text = res.get('result', '')
    
    NodeChoice.objects.create(
        current_node=node,
        action_text=action_text,
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
    # (기존 코드 유지)
    sys_prompt = "세계관 상세 정보 JSON 생성 (title, description, detail_description, play_time)"
    user_prompt = f"설정: {setting}\n줄거리: {synopsis[:500]}..."
    return call_llm(sys_prompt, user_prompt, json_format=True)