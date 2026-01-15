import os
import json
import time
import uuid
import random
import re
from openai import OpenAI
from django.conf import settings
from .models import Genre, Cliche, Story, CharacterState, StoryNode, NodeChoice, StoryBranch

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

# 공통 제약 조건 상수
KOREAN_ONLY_RULE = """
[필수 규칙]
[Output Rules]
1. Use ONLY Korean(Hangul). 
2. Translate ALL English words to Korean. (e.g., 'system' -> '시스템', 'pushed' -> '밀려났다')
3. Do NOT use Chinese characters.
4. Exception: Keep Proper Nouns (Names like 'V', 'Silverhand') in English if necessary.
"""

# ==========================================
# [설정 변수: 스토리 구조 제어]
# ==========================================
INITIAL_BRANCH_QUOTA = 2     
TOTAL_DEPTH_PER_PATH = 12    

def _clean_text_value(text):
    """
    [스마트 필터링] 문자열 값에서만 불필요한 외국어를 제거합니다.
    """
    if not isinstance(text, str):
        return text

    # 1. 한자(Chinese) 및 일본어 등 제거 (범위 확대)
    # \u4e00-\u9fff (한자), \u3040-\u30ff (일본어)
    text = re.sub(r'[\u4e00-\u9fff\u3040-\u30ff]+', '', text)
    
    # 2. 괄호 안의 영어 제거 (예: (System), (Love)) -> 보통 번역 후 병기하는 경우라 삭제해도 무방
    text = re.sub(r'\([A-Za-z\s]+\)', '', text)

    # 3. 소문자로 시작하는 영어 단어 제거 (동사, 일반명사 등)
    # 예: "pushed되었다" -> "되었다", "consciousness가" -> "가"
    # 예외: "V", "Silverhand" 처럼 대문자로 시작하는 고유명사는 남김
    def _remove_lowercase_english(match):
        word = match.group()
        # 첫 글자가 소문자면 삭제, 대문자면 유지
        if word[0].islower():
            return ""
        return word

    text = re.sub(r'[A-Za-z]+', _remove_lowercase_english, text)
    
    # 4. 불필요한 공백 정리
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def _clean_data_recursive(data):
    """
    JSON 데이터의 구조는 건드리지 않고, 내부의 '문자열 값'만 찾아서 청소합니다.
    """
    if isinstance(data, dict):
        return {k: _clean_data_recursive(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_clean_data_recursive(v) for v in data]
    elif isinstance(data, str):
        return _clean_text_value(data)
    else:
        return data

def call_llm(system_prompt, user_prompt, json_format=False, stream=False, max_tokens=4000, max_retries=3, timeout=300):
    # 시스템 프롬프트에 한국어 제약 조건 추가
    full_system_prompt = f"{system_prompt}\n\n{KOREAN_ONLY_RULE}"    

    messages = [{"role": "system", "content": full_system_prompt}, {"role": "user", "content": user_prompt}]
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
                    if chunk.choices and chunk.choices[0].delta.content:
                        content += chunk.choices[0].delta.content
                print(" Done.")
            else:
                content = response.choices[0].message.content
            '''
            if json_format:
                cleaned = content.replace("```json", "").replace("```", "").strip()
                return json.loads(cleaned)
            return content
            '''
            if json_format:
                # 1. 일단 JSON 파싱 (영어 키값 보존을 위해)
                cleaned_str = content.replace("```json", "").replace("```", "").strip()
                try:
                    parsed_data = json.loads(cleaned_str)
                except json.JSONDecodeError:
                    # 파싱 실패 시, 혹시 모를 문자열 끝부분 잘림 등을 보정하여 재시도
                    end_idx = cleaned_str.rfind("}")
                    if end_idx != -1:
                         parsed_data = json.loads(cleaned_str[:end_idx+1])
                    else:
                        raise

                # 2. 파싱된 데이터 내부의 값만 청소 (Recursive)
                return _clean_data_recursive(parsed_data)
            
            else:
                # 일반 텍스트는 바로 청소
                return _clean_text_value(content)

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
    except: pass

    # 2. 클리셰 매칭
    matched_cliche = _match_cliche(refined_setting)
    story = Story.objects.create(user_world_setting=refined_setting, main_cliche=matched_cliche)
    
    # 3. 메인 시놉시스 생성
    print("  [Step 3] Generating Root Synopsis...")
    root_synopsis = _generate_synopsis(story, matched_cliche, protagonist_name, protagonist_info['desc'], include_example=True)
    story.synopsis = root_synopsis
    story.save()

    # 3.5 정보 추출 및 업데이트
    universe_details = _generate_universe_details(refined_setting, root_synopsis)
    characters_info_json = _extract_characters_info(root_synopsis, protagonist_info)
    
    try:
        update_universe_details_neo4j(
            universe_id=universe_id, 
            synopsis=root_synopsis, 
            twisted_synopsis="", 
            title=universe_details.get("title", "무제"), 
            description=universe_details.get("description", ""), 
            detail_description=universe_details.get("detail_description", ""), 
            estimated_play_time_min=universe_details.get("estimated_play_time_min", 30),
            estimated_play_time_max=universe_details.get("estimated_play_time_max", 60),
            characters_info=characters_info_json
        )
    except Exception as e:
        print(f"⚠️ Neo4j Details Update Failed: {e}")
        pass

    # 4. 메인 경로 노드 생성 (엔딩까지)
    print("  [Step 4] Creating Main Path Nodes...")
    # [수정] characters_info_json 전달
    main_nodes = _generate_path_segment(
        story, root_synopsis, protagonist_name, 
        start_node=None, universe_id=universe_id,
        characters_info_json=characters_info_json 
    )
    
    if not main_nodes: raise ValueError("메인 노드 생성 실패")
    try: link_universe_to_first_scene(universe_id, f"{universe_id}_{main_nodes[0].id}")
    except: pass

    # 5. 재귀적 분기 생성 시작 (DFS)
    print(f"\n🌳 [Recursive Branching Start] Quota(n): {INITIAL_BRANCH_QUOTA}")
    
    _generate_recursive_story(
        story=story,
        current_path_nodes=main_nodes,
        quota=INITIAL_BRANCH_QUOTA,
        universe_id=universe_id,
        protagonist_name=protagonist_name,
        characters_info_json=characters_info_json,
        hierarchy_id="1"
    )

    print("\n✨ All Story Generation Completed!")
    return story.id


# ==========================================
# [핵심 로직: DFS 재귀적 스토리 생성]
# ==========================================

def _generate_recursive_story(story, current_path_nodes, quota, universe_id, protagonist_name, characters_info_json, hierarchy_id):
    if quota <= 0:
        print(f"    🚫 [Depth End] {hierarchy_id}: Quota reached 0. Stopping branch generation.")
        return

    valid_nodes = [node for node in current_path_nodes if node.chapter_phase != '결말']
    if not valid_nodes: return

    sections = _split_nodes_into_sections(valid_nodes, quota)
    
    print(f"  👉 [Processing {hierarchy_id}] Finding {quota} twist points in this path...")

    for idx, section in enumerate(sections):
        if not section: continue
        
        current_branch_num = f"{hierarchy_id}-{idx+1}"
        print(f"    🔎 [{current_branch_num}] Searching twist point in section {idx+1}/{quota}...")
        
        target_node = _select_twist_point_from_candidates(section)
        
        if not target_node:
            print("      ⚠️ No suitable twist point found.")
            continue
            
        print(f"      📌 Twist Point Found: Node {target_node.id} ({target_node.chapter_phase})")

        # [수정] 전체 히스토리(심경 변화 포함) 가져오기
        history_context = _get_full_history(target_node)
        
        twisted_synopsis = _generate_twisted_synopsis_data(
            story, history_context, target_node.chapter_phase, characters_info_json
        )
        
        StoryBranch.objects.create(
                    story=story, 
                    parent_node=target_node, 
                    synopsis=twisted_synopsis,
                    hierarchy_id=current_branch_num
                )
                
        print(f"      📝 Generating Nodes for [{current_branch_num}] ...")
        # [수정] characters_info_json 전달
        new_branch_nodes = _generate_path_segment(
            story, twisted_synopsis, protagonist_name,
            start_node=target_node, universe_id=universe_id, is_twist_branch=True,
            characters_info_json=characters_info_json
        )

        if new_branch_nodes:
            original_choice = target_node.choices.first()
            original_action = original_choice.choice_text if original_choice else "원래대로 진행"
            _create_twist_condition(target_node, new_branch_nodes[0], universe_id, protagonist_name, original_action)

            next_quota = quota - 1
            if next_quota > 0:
                print(f"      ↘️ Recursing into [{current_branch_num}] with quota {next_quota} (DFS)...")
                _generate_recursive_story(
                    story, 
                    new_branch_nodes, 
                    next_quota,
                    universe_id, 
                    protagonist_name, 
                    characters_info_json,
                    current_branch_num
                )
            else:
                print(f"      🛑 [{current_branch_num}] Leaf branch created (Next quota 0).")


# ==========================================
# [보조 함수들: 노드 생성 및 관리]
# ==========================================

# [수정] characters_info_json 인자 추가
def _generate_path_segment(story, synopsis, protagonist_name, start_node=None, universe_id=None, is_twist_branch=False, characters_info_json="[]"):
    start_depth = start_node.depth if start_node else 0
    next_depth = start_depth + 1
    
    needed_nodes = TOTAL_DEPTH_PER_PATH - start_depth
    if needed_nodes < 1: needed_nodes = 1 

    # [수정] start_node를 기반으로 '전체 히스토리(심경 변화 포함)' 추출
    initial_history = _get_full_history(start_node)

    # [수정] _create_nodes_common에 히스토리와 캐릭터 정보 전달
    nodes = _create_nodes_common(
        story, synopsis, protagonist_name, needed_nodes, next_depth, universe_id,
        initial_history=initial_history,
        characters_info_json=characters_info_json
    )
    
    if not nodes: return []

    _connect_linear_nodes(nodes, universe_id, protagonist_name)
    
    return nodes

# [수정] 전체 히스토리, 캐릭터 정보 등 모든 맥락을 입력받도록 대폭 수정
def _create_nodes_common(story, synopsis, protagonist_name, count, start_depth, universe_id, initial_history="", characters_info_json="[]"):
    phases = ["발단", "전개", "절정", "결말"]
    BATCH_SIZE = 2
    
    created_nodes = []
    generated_count = 0
    
    # [신규] 이번 세션에서 생성된 노드들의 히스토리를 누적 저장할 리스트
    current_session_history = [] 

    normal_node_count = count - 1 if count > 0 else 0
    
    print(f"    🔄 [Generation Plan] Total: {count} | Normal Batch: {normal_node_count} | Final Ending: 1")

    # --- 내부 함수: 프롬프트 생성기 (맥락 주입의 핵심) ---
    def build_prompt(batch_size, is_ending=False):
        # 1. 전체 흐름 구성 (과거 히스토리 + 현재 세션 생성분)
        full_history_text = initial_history
        if current_session_history:
            session_hist_text = "\n\n".join(current_session_history)
            if full_history_text:
                full_history_text += f"\n\n{session_hist_text}"
            else:
                full_history_text = session_hist_text
        
        # 2. 직전 상황 요약 (가장 최근 내용은 한번 더 강조)
        prev_context_snippet = ""
        if created_nodes:
            last = created_nodes[-1]
            prev_context_snippet = f"...{last.content[-500:]}"
        elif initial_history:
             prev_context_snippet = "(위 전체 줄거리 흐름의 마지막 부분 참조)"
        
        sys = (
            f"당신은 인터랙티브 스토리 작가입니다. 주인공 '{protagonist_name}'의 시점에서 장면(Node)들을 생성하세요.\n"
            "**[입력 데이터 설명]**\n"
            "1. **전체 줄거리 흐름**: 이야기의 시작부터 바로 직전까지의 모든 사건과 **인물들의 심경 변화**입니다. 이 흐름을 완벽하게 숙지하고 이어가세요.\n"
            "2. **등장인물 특성**: 인물들의 고유한 성격과 특성을 반영하여 대사와 행동을 작성하세요.\n"
            "3. **현재 시놉시스**: 이번 구간에서 진행되어야 할 핵심 줄거리입니다.\n\n"
            "**[출력 필수 항목]**\n"
            "각 장면은 title, description(500자 이상), setting, purpose, characters_list, character_states, character_changes를 포함해야 합니다.\n"
        )
        
        if is_ending:
            sys += (
                "**[엔딩 생성 모드]**\n"
                "주인공의 서사를 완벽하게 마무리하는 **마지막 엔딩 장면(1개)**을 작성하세요.\n"
                "- 확실하고 닫힌 결말(Closed Ending)\n"
                "- 500자 이상의 풍부한 분량\n"
                "- 전체 흐름과 인물의 감정선을 통합하여 감동적인 마무리를 지으세요.\n"
            )
            req_count_str = "1개 (엔딩)"
        else:
            sys += (
                f"**[일반 진행 모드]**\n"
                f"생성할 노드 개수: **정확히 {batch_size}개**\n"
                "이야기를 끝내지 말고, 시놉시스에 따라 자연스럽게 전개하세요.\n"
            )
            req_count_str = f"{batch_size}개"

        # [핵심] 모든 맥락 정보를 상세하게 주입
        user = (
            f"### [1] 등장인물 정보 및 특성\n{characters_info_json}\n\n"
            f"### [2] 현재 적용 시놉시스\n{synopsis}\n\n"
            f"### [3] 지금까지의 전체 줄거리 흐름 (심경 변화 포함)\n{full_history_text}\n\n"
            f"### [4] 직전 장면 내용 (마지막 부분)\n{prev_context_snippet}\n\n"
            f"--------------------------------------------------\n"
            f"위 모든 맥락을 반영하여 다음 장면들을 생성하세요.\n"
            f"요청 개수: {req_count_str}\n"
            f"JSON 형식: {{'scenes': [ ... ]}}"
        )
        return sys, user

    # ==========================================
    # 1. 일반 노드 배치 생성
    # ==========================================
    while generated_count < normal_node_count:
        remaining = normal_node_count - generated_count
        current_batch_size = min(BATCH_SIZE, remaining)
        
        sys_prompt, user_prompt = build_prompt(current_batch_size, is_ending=False)
        
        print(f"      runner: generating normal batch {generated_count+1}~{generated_count+current_batch_size}...")
        
        try:
            res = call_llm(sys_prompt, user_prompt, json_format=True, stream=True, max_tokens=6000, timeout=180)
            scenes = res.get('scenes', [])
        except Exception as e:
            print(f"      ⚠️ Normal batch generation failed: {e}")
            scenes = []

        if not scenes:
            print("      ⚠️ Empty response. Skipping this batch.")
            break 

        for i, scene_data in enumerate(scenes):
            current_depth = start_depth + generated_count + i
            progress_ratio = current_depth / TOTAL_DEPTH_PER_PATH
            phase_idx = int(progress_ratio * 4) 
            if phase_idx > 2: phase_idx = 2 
            phase_name = phases[phase_idx]

            node = _save_node_to_db(story, scene_data, phase_name, current_depth, universe_id)
            created_nodes.append(node)
            
            # [중요] 생성된 노드 내용을 즉시 히스토리에 반영 (심경 변화 포함)
            changes_str = json.dumps(scene_data.get('character_changes', {}), ensure_ascii=False)
            hist_entry = f"[장면 {current_depth} ({phase_name})]\n내용: {node.content}"
            if changes_str and changes_str != "{}" and changes_str != "null":
                hist_entry += f"\n(인물 심경 변화: {changes_str})"
            current_session_history.append(hist_entry)

        generated_count += len(scenes)

    # ==========================================
    # 2. 마지막 엔딩 노드 독립 생성
    # ==========================================
    if generated_count < count:
        print("      🏁 [Final Step] Generating The Ending Node independently...")
        
        sys_prompt, user_prompt = build_prompt(1, is_ending=True)

        try:
            res = call_llm(sys_prompt, user_prompt, json_format=True, stream=True, max_tokens=6000, timeout=300)
            scenes = res.get('scenes', [])
        except Exception as e:
            print(f"      ⚠️ Ending generation failed: {e}")
            scenes = []
            
        if scenes:
            scene_data = scenes[0]
            current_depth = start_depth + generated_count
            node = _save_node_to_db(story, scene_data, "결말", current_depth, universe_id)
            created_nodes.append(node)
            generated_count += 1
        else:
            print("      ⚠️ Failed to generate ending node.")

    return created_nodes

def _save_node_to_db(story, scene_data, phase_name, current_depth, universe_id):
    node = StoryNode.objects.create(
        story=story, 
        chapter_phase=phase_name, 
        content=scene_data.get('description', ''),
        depth=current_depth,
        is_twist_point=False 
    )

    changes_json = json.dumps(scene_data.get('character_changes', {}), ensure_ascii=False)
    node.temp_character_changes = changes_json
    
    if universe_id:
        try:
            neo4j_data = StoryNodeData(
                scene_id=f"{universe_id}_{node.id}",
                phase=phase_name,
                title=scene_data.get('title', '무제'),
                setting=scene_data.get('setting', ''),
                description=scene_data.get('description', ''),
                purpose=scene_data.get('purpose', ''),
                characters_list=scene_data.get('characters_list', []),
                character_states=json.dumps(scene_data.get('character_states', {}), ensure_ascii=False),
                depth=current_depth
            )
            sync_node_to_neo4j(neo4j_data)
        except Exception as e:
            print(f"Neo4j Node Sync Error: {e}")
            
    return node

# [신규 함수] 노드 역추적을 통해 전체 스토리 흐름과 심경 변화를 텍스트로 추출
def _get_full_history(node):
    if not node: return ""
    history_list = []
    curr = node
    while curr:
        changes = getattr(curr, 'temp_character_changes', '')
        info = f"[장면 {curr.depth} ({curr.chapter_phase})]\n내용: {curr.content}"
        # 심경 변화가 있다면 함께 기록
        if changes and changes != "{}" and changes != "null":
             info += f"\n(인물 심경 변화: {changes})"
        history_list.append(info)
        curr = curr.prev_node
    # 과거 -> 현재 순으로 정렬
    return "\n\n".join(reversed(history_list))

# ==========================================
# [기타 로직 함수들]
# ==========================================

def _split_nodes_into_sections(nodes, n):
    if n <= 0: return []
    if n == 1: return [nodes]
    k, m = divmod(len(nodes), n)
    return [nodes[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]

def _select_twist_point_from_candidates(candidates):
    if not candidates: return None
    candidates = [n for n in candidates if n.choices.count() < 2]
    if not candidates: return None
    if len(candidates) < 3:
        return random.choice(candidates)
        
    prompt_text = ""
    node_map = {}
    
    for n in candidates:
        if n.choices.count() >= 2: continue
        prompt_text += f"[ID: {n.id}] Phase: {n.chapter_phase} | 내용: {n.content[:60]}...\n"
        node_map[n.id] = n
    
    if not node_map:
        return None

    sys_prompt = (
        "당신은 스토리 에디터입니다. 아래 장면 목록 중, 이야기의 흐름을 비틀어(Twist) "
        "새로운 분기를 만들기에 가장 흥미롭고 개연성 있는 지점을 하나 선택하세요.\n"
        "반드시 JSON 형식 {'scene_id': ID숫자} 로 응답하세요."
    )
    user_prompt = f"후보 장면들:\n{prompt_text}"

    try:
        res = call_llm(sys_prompt, user_prompt, json_format=True)
        selected_id = res.get('scene_id')
        if selected_id and selected_id in node_map:
            return node_map[selected_id]
    except Exception as e:
        print(f"      ⚠️ Twist Point Selection Error: {e}")
        pass
    
    print("      ⚠️ No valid twist point selected by AI. Skipping branch generation.")
    return None

def _match_cliche(setting):
    all_genres = Genre.objects.all()
    if not all_genres.exists():
        print("⚠️ DB에 장르 데이터가 없습니다.")
        return None
    
    genre_text_list = []
    for g in all_genres:
        desc = g.description if g.description else "설명 없음"
        genre_text_list.append(f"- [{g.name}]: {desc}")
    
    sys_prompt_1 = (
        "당신은 장르 문학 분석가입니다. "
        "사용자의 입력을 분석하여, 아래 목록 중 가장 적합한 **단 하나의 장르**를 선택하세요.\n"
        "반드시 JSON 형식 {'genre_name': '장르명', 'reason': '이유'} 으로만 응답하세요."
    )
    user_prompt_1 = f"사용자 설정: {setting}\n\n[장르 목록]\n" + "\n".join(genre_text_list)
    
    res_1 = call_llm(sys_prompt_1, user_prompt_1, json_format=True)
    selected_genre_name = res_1.get('genre_name', '판타지')
    
    try:
        selected_genre = Genre.objects.get(name=selected_genre_name)
    except Genre.DoesNotExist:
        selected_genre = all_genres.first()

    cliches = Cliche.objects.filter(genre=selected_genre)
    if not cliches.exists(): return Cliche.objects.first()

    cliche_text_list = []
    for c in cliches:
        info = (
            f"ID: {c.id}\n제목: {c.title}\n정의: {c.summary}\n구조 가이드: {c.structure_guide}\n"
        )
        cliche_text_list.append(info)

    sys_prompt_2 = (
        f"당신은 '{selected_genre.name}' 장르 전문 편집자입니다. "
        "장르와 설정을 고려하여 **가장 흥미롭고 극적인 전개가 가능한 클리셰** 하나를 선택하세요.\n"
        "응답은 JSON 형식 {'cliche_id': ID숫자, 'reason': '선택 이유'} 만 반환하세요."
    )
    user_prompt_2 = (
        f"사용자 설정: {setting}\n\n[선택된 장르: {selected_genre.name}]\n\n"
        f"[클리셰 후보 목록]\n" + "\n----------------\n".join(cliche_text_list)
    )

    res_2 = call_llm(sys_prompt_2, user_prompt_2, json_format=True)
    
    try:
        selected_id = res_2.get('cliche_id')
        return Cliche.objects.get(id=selected_id)
    except:
        return random.choice(list(cliches))

def _refine_setting_and_protagonist(raw_setting):
    sys_prompt = "세계관과 주인공을 정의하세요. 주인공 이름은 한글, 성격/믿음/사상/외모를 포함해야 합니다."
    user_prompt = (
        f"입력: {raw_setting}\n"
        "출력 JSON: {'refined_setting': '...', 'protagonist': {'name': '...', 'desc': '성격, 믿음, 사상, 외모 포함 상세 묘사'}}"
    )
    res = call_llm(sys_prompt, user_prompt, json_format=True)
    return res.get('refined_setting', raw_setting), res.get('protagonist', {'name':'이안', 'desc':'평범함'})

def _generate_synopsis(story, cliche, p_name, p_desc, include_example=False):
    sys_prompt = (
        "당신은 베스트셀러 웹소설 작가입니다. "
        "주어진 세계관 설정과 **지정된 필수 클리셰**를 완벽하게 조합하여 매력적인 시놉시스를 작성하세요.\n"
        "1. 분량은 2000자 이상.\n"
        "2. 기승전결 구조와 주인공의 내면 변화 포함.\n"
        "3. **선택된 클리셰의 '핵심 요약'과 '전개 가이드'를 충실히 따를 것.**"
        "4. **사용자 설정 우선**: 사용자가 입력한 구체적인 설정은 크게 변경하거나 생략하지 말고 최대한 이야기에 포함시키세요.\n"
        "5. 문장은 번역투가 아닌 자연스러운 한국어 소설체로 작성하세요."
    )
    
    cliche_detail = (
        f"제목: {cliche.title}\n"
        f"장르: {cliche.genre.name}\n"
        f"핵심 요약: {cliche.summary}\n"
        f"전개 가이드: {cliche.structure_guide}"
    )
    
    if include_example and cliche.example_work_summary:
        cliche_detail += f"\n\n★ 참고용 대표 예시 작품 (영감만 받을 것) ★\n{cliche.example_work_summary}"
    
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

def _connect_linear_nodes(nodes, universe_id, protagonist_name):
    sys_prompt = (
        f"주인공 '{protagonist_name}'이 현재 장면에서 다음 장면으로 넘어가기 위해 취해야 할 **자연스럽고 일상적인 행동(Condition Action)**을 정의하세요.\n"
        "1. 유저가 별도의 힌트 없이도 상황상 자연스럽게 입력할 법한 행동이어야 합니다. 행위가 구체적이면 안됩니다.(예: '문을 연다', '대답한다', '전화를 받는다')\n"
        "2. **조건 행동의 결과(result)는 다음 장면의 시작 부분과 자연스럽게 이어져야 합니다.**"
        "3. 아주 일상적인 행동이어야 합니다. 마치 방탈출을 하는 게임 플레이어처럼 유저가 할 수 있을 법한 행동을 조건 행위로 지정해야 합니다."
    )
    
    for i in range(len(nodes) - 1):
        curr = nodes[i]
        next_n = nodes[i+1]
        
        curr.prev_node = next_n.prev_node 
        next_n.prev_node = curr
        next_n.save()
        
        user_prompt = (
            f"현재 장면(마지막 부분): ...{curr.content[-300:]}\n"
            f"다음 장면(시작 부분): {next_n.content[:300]}...\n\n"
            "위 두 장면을 연결하는 Action과 Result를 생성하세요.\n"
            "출력 JSON: {'action': '유저가 입력할 행동', 'result': '행동의 결과(다음 줄거리 도입부로 자연스럽게 연결)'}"
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
                next_changes = getattr(next_n, 'temp_character_changes', "{}")
                sync_action_to_neo4j(
                    f"{universe_id}_{curr.id}", 
                    f"{universe_id}_{next_n.id}", 
                    action_text, 
                    result_text, 
                    is_twist=False,
                    character_changes=next_changes
                )
            except: pass

def _generate_twisted_synopsis_data(story, acc_content, phase, characters_info_json):
    sys_prompt = (
        "기존 스토리의 흐름을 비틀어 새로운 결말로 향하는 'Twist Synopsis'를 작성하세요.\n"
        "1. 분량은 2000자 이상.\n"
        "2. **제공된 모든 주요 등장인물의 성격과 특성을 반영하여 입체적인 변화를 주세요.**\n"
        "3. 단순히 상황만 꼬는 것이 아니라, **확실한 결말(Closed Ending)**을 맺어야 합니다."
    )
    user_prompt = (
        f"현재까지 진행된 이야기: {acc_content[-1000:]}\n"
        f"현재 단계: {phase} (이 지점부터 이야기가 달라집니다)\n"
        f"등장인물 상세 정보: {characters_info_json}\n\n"
        "위 정보를 바탕으로 완결된 형태의 비틀린 시놉시스를 작성해주세요."
    )
    return call_llm(sys_prompt, user_prompt, stream=True, max_tokens=8000, timeout=300)

def _create_twist_condition(node, twist_next_node, universe_id, protagonist_name, original_action_text):
    sys_prompt = (
        f"현재 장면에서 이야기가 완전히 다른 방향(반전)으로 흐르기 위해, "
        f"주인공 '{protagonist_name}'이 수행해야 할 **돌발적이고 파격적인 조건 행동(Twist Action)**을 정의하세요.\n"
        "1. 기존의 정석적인 행동과는 의도가 명확히 달라야 합니다.\n"
        "2. **행동의 결과(result)는 반전된 다음 장면의 시작 부분과 자연스럽게 이어져야 합니다.**"
    )
    
    user_prompt = (
        f"현재 장면(마지막 부분): ...{node.content[-300:]}\n"
        f"반전된 다음 장면(시작 부분): {twist_next_node.content[:300]}...\n"
        f"참고(기존 정석 행동): '{original_action_text}'\n\n"
        "위 두 장면을 연결하는 반전 행동(Action)과 결과(Result)를 생성하세요.\n"
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
            twist_changes = getattr(twist_next_node, 'temp_character_changes', "{}")
            sync_action_to_neo4j(
                f"{universe_id}_{node.id}", 
                f"{universe_id}_{twist_next_node.id}", 
                action_text, 
                result_text, 
                is_twist=True,
                character_changes=twist_changes
            )
        except: pass

def _generate_universe_details(setting, synopsis):
    sys_prompt = "세계관 상세 정보 JSON 생성 (title, description, detail_description, estimated_play_time_min (int), estimated_play_time_max (int))"
    user_prompt = f"설정: {setting}\n줄거리: {synopsis[:500]}..."
    return call_llm(sys_prompt, user_prompt, json_format=True)