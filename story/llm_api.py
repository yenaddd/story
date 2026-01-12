import os
import json
import time
import uuid
import random
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
KOREAN_ONLY_RULE = "출력은 고유명사(지명, 인명 등 불가피한 경우)를 제외하고는 반드시 '한국어'로 작성해야 합니다. 영어를 섞어 쓰지 마세요."

# ==========================================
# [설정 변수: 스토리 구조 제어]
# ==========================================
INITIAL_BRANCH_QUOTA = 2     # 초기 메인 스토리에서 생성할 분기(가지)의 개수 (n)
TOTAL_DEPTH_PER_PATH = 12    # 시작부터 엔딩까지 이어지는 노드의 총 개수 (길이)

def call_llm(system_prompt, user_prompt, json_format=False, stream=False, max_tokens=4000, max_retries=3, timeout=300):
    full_system_prompt = f"{system_prompt}\n\n[중요 규칙]\n{KOREAN_ONLY_RULE}"
    
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
                    if chunk.choices[0].delta.content:
                        content += chunk.choices[0].delta.content
                print(" Done.")
            else:
                content = response.choices[0].message.content

            if json_format:
                # 가끔 마크다운 코드블록이 포함될 수 있어 제거
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
    1. 초기 시놉시스 및 전체 노드 생성 (Point 1)
    2. 재귀적 분기 생성 시작 (Point 2~10)
    """
    universe_id = str(uuid.uuid4())
    print(f"\n🌍 [NEO4J] Creating Universe Node: {universe_id}")

    # 1. 설정 구체화 및 주인공 정의
    refined_setting, protagonist_info = _refine_setting_and_protagonist(user_world_setting)
    protagonist_name = protagonist_info['name']

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
        # [수정] 추출된 상세 정보와 min/max 시간을 Neo4j로 전송
        update_universe_details_neo4j(
            universe_id=universe_id, 
            synopsis=root_synopsis, 
            twisted_synopsis="", 
            title=universe_details.get("title", "무제"), 
            description=universe_details.get("description", ""), 
            detail_description=universe_details.get("detail_description", ""), 
            estimated_play_time_min=universe_details.get("estimated_play_time_min", 30), # 정수형 기본값
            estimated_play_time_max=universe_details.get("estimated_play_time_max", 60), # 정수형 기본값
            characters_info=characters_info_json
        )
    except Exception as e:
        print(f"⚠️ Neo4j Details Update Failed: {e}")
        pass

    # 4. 메인 경로 노드 생성 (엔딩까지)
    print("  [Step 4] Creating Main Path Nodes...")
    main_nodes = _generate_path_segment(
        story, root_synopsis, protagonist_name, 
        start_node=None, universe_id=universe_id
    )
    
    if not main_nodes: raise ValueError("메인 노드 생성 실패")
    try: link_universe_to_first_scene(universe_id, f"{universe_id}_{main_nodes[0].id}")
    except: pass

    # 5. 재귀적 분기 생성 시작 (DFS)
    # 초기 n값: INITIAL_BRANCH_QUOTA
    # 계층 ID: "Root"
    print(f"\n🌳 [Recursive Branching Start] Quota(n): {INITIAL_BRANCH_QUOTA}")
    
    _generate_recursive_story(
        story=story,
        current_path_nodes=main_nodes,
        quota=INITIAL_BRANCH_QUOTA,  # 변수 n
        universe_id=universe_id,
        protagonist_name=protagonist_name,
        characters_info_json=characters_info_json,
        hierarchy_id="1" # (Point 6) 체계적 번호 부여 시작
    )

    print("\n✨ All Story Generation Completed!")
    return story.id


# ==========================================
# [핵심 로직: DFS 재귀적 스토리 생성]
# ==========================================

def _generate_recursive_story(story, current_path_nodes, quota, universe_id, protagonist_name, characters_info_json, hierarchy_id):
    """
    Point 4-10 구현:
    - 현재 흐름에서 quota(n)개의 분기점을 찾음 (개연성 기반)
    - 각 분기점에서 Twisted Synopsis 및 엔딩까지 노드 생성
    - 생성된 하위 흐름에 대해 (n-1)개의 분기를 찾으러 재귀 호출 (DFS)
    """
    
    # [Point 10] n이 0이면 종료 (Base Case)
    if quota <= 0:
        print(f"    🚫 [Depth End] {hierarchy_id}: Quota reached 0. Stopping branch generation.")
        return

    # 분기 후보군 선정 ('결말' 제외)
    valid_nodes = [node for node in current_path_nodes if node.chapter_phase != '결말']
    if not valid_nodes: return

    # [Point 2] 분기 개수(n)에 맞게 구역을 나누어 '개연성 있는' 분기점 탐색
    sections = _split_nodes_into_sections(valid_nodes, quota)
    
    print(f"  👉 [Processing {hierarchy_id}] Finding {quota} twist points in this path...")

    # [Point 7, 8] 순차적으로 분기점 처리 (DFS Loop)
    for idx, section in enumerate(sections):
        if not section: continue
        
        current_branch_num = f"{hierarchy_id}-{idx+1}"
        
        print(f"    🔎 [{current_branch_num}] Searching twist point in section {idx+1}/{quota}...")
        
        target_node = _select_twist_point_from_candidates(section)
        
        if not target_node:
            print("      ⚠️ No suitable twist point found.")
            continue
            
        print(f"      📌 Twist Point Found: Node {target_node.id} ({target_node.chapter_phase})")

        # [Point 3, 5] Twisted Synopsis 생성 및 엔딩까지 노드 생성
        history_context = _get_story_history(target_node)
        twisted_synopsis = _generate_twisted_synopsis_data(
            story, history_context, target_node.chapter_phase, characters_info_json
        )
        
        StoryBranch.objects.create(
                    story=story, 
                    parent_node=target_node, 
                    synopsis=twisted_synopsis,
                    hierarchy_id=current_branch_num
                )
                
        print(f"      📝 Generating Nodes for [{current_branch_num}] (Depth Fixed: {TOTAL_DEPTH_PER_PATH})...")
        new_branch_nodes = _generate_path_segment(
            story, twisted_synopsis, protagonist_name,
            start_node=target_node, universe_id=universe_id, is_twist_branch=True
        )

        # 분기점 연결
        if new_branch_nodes:
            original_choice = target_node.choices.first()
            original_action = original_choice.choice_text if original_choice else "원래대로 진행"
            _create_twist_condition(target_node, new_branch_nodes[0], universe_id, protagonist_name, original_action)

            # [Point 4, 7, 8] 재귀 호출 (DFS)
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
# [보조 함수들]
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

def _generate_path_segment(story, synopsis, protagonist_name, start_node=None, universe_id=None, is_twist_branch=False):
    start_depth = start_node.depth if start_node else 0
    next_depth = start_depth + 1
    
    needed_nodes = TOTAL_DEPTH_PER_PATH - start_depth
    if needed_nodes < 1: needed_nodes = 1 

    nodes = _create_nodes_common(story, synopsis, protagonist_name, needed_nodes, next_depth, universe_id)
    
    if not nodes: return []

    _connect_linear_nodes(nodes, universe_id, protagonist_name)
    
    return nodes

def _create_nodes_common(story, synopsis, protagonist_name, count, start_depth, universe_id):
    phases = ["발단", "전개", "절정", "결말"]
    BATCH_SIZE = 3
    
    created_nodes = []
    generated_count = 0
    
    print(f"    🔄 [Batch Generation Start] Total request: {count} nodes (Batch size: {BATCH_SIZE})")

    while generated_count < count:
        current_batch_size = min(BATCH_SIZE, count - generated_count)
        
        prev_context = ""
        if created_nodes:
            last_node = created_nodes[-1]
            prev_context = (
                f"\n[직전 장면 요약]\n"
                f"장면 ID: {last_node.id}\n"
                f"마지막 내용: {last_node.content[-500:]}\n"
                f"-> 위 내용에 자연스럽게 이어서 다음 장면들을 작성하세요."
            )

        sys_prompt = (
            f"당신은 인터랙티브 스토리 작가입니다. 주인공 '{protagonist_name}'의 시점에서 장면(Node)들을 생성하세요.\n"
            "각 장면은 title, description(500자 이상), setting, purpose, characters_list, character_states, character_changes를 포함해야 합니다.\n\n"
            "**[중요]**\n"
            f"생성해야 할 노드의 개수는 **정확히 {current_batch_size}개**입니다.\n"
            f"JSON 포맷이 끊기지 않도록 주의하세요.\n"
        )
        
        is_last_batch = (generated_count + current_batch_size) >= count
        if is_last_batch:
            sys_prompt += "이번 배치의 마지막 장면은 이야기의 **확실한 결말(Ending)**을 맺어야 합니다.\n"

        user_prompt = (
            f"전체 시놉시스: {synopsis}\n"
            f"{prev_context}\n"
            f"요청 생성 개수: {current_batch_size}개\n"
            f"JSON 형식: {{'scenes': [ ... ({current_batch_size}개의 장면 객체) ... ]}}"
        )
        
        print(f"      runner: generating batch {generated_count+1}~{generated_count+current_batch_size}...")
        try:
            res = call_llm(sys_prompt, user_prompt, json_format=True, stream=True, max_tokens=8000)
            scenes = res.get('scenes', [])
        except Exception as e:
            print(f"      ⚠️ Batch generation failed: {e}")
            scenes = []

        if not scenes:
            print("      ⚠️ Empty response received. Retrying or stopping.")
            break

        for i, scene_data in enumerate(scenes):
            current_depth = start_depth + generated_count + i
            
            progress_ratio = current_depth / TOTAL_DEPTH_PER_PATH
            phase_idx = int(progress_ratio * 4) 
            if phase_idx > 3: phase_idx = 3
            phase_name = phases[phase_idx]

            node = StoryNode.objects.create(
                story=story, 
                chapter_phase=phase_name, 
                content=scene_data.get('description', ''),
                depth=current_depth,
                is_twist_point=False 
            )

            changes_json = json.dumps(scene_data.get('character_changes', {}), ensure_ascii=False)
            node.temp_character_changes = changes_json
            
            created_nodes.append(node)
            
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

        generated_count += len(scenes)
        
        if len(scenes) < current_batch_size:
             print("      ⚠️ LLM generated fewer nodes than requested.")
             pass

    return created_nodes
    
def _get_story_history(target_node):
    path_contents = []
    curr = target_node
    while curr:
        path_contents.append(curr.content)
        curr = curr.prev_node 
    return "\n".join(reversed(path_contents))

# ==========================================
# [기타 로직 함수들]
# ==========================================

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
        "1. 유저가 별도의 힌트 없이도 상황상 자연스럽게 입력할 법한 행동이어야 합니다. 행위가 구체적이면 안됩니다.\n"
        "2. **조건 행동의 결과(result)는 다음 장면의 시작 부분과 자연스럽게 이어져야 합니다.**"
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
            "출력 JSON: {'action': '유저가 입력할 행동', 'result': '행동의 결과'}"
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
    return call_llm(sys_prompt, user_prompt, stream=True, max_tokens=8000)

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
    # [수정] play_time 대신 min/max(int)를 요청하도록 프롬프트 변경
    sys_prompt = "세계관 상세 정보 JSON 생성 (title, description, detail_description, estimated_play_time_min (int)(분 단위), estimated_play_time_max (int)(분 단위))"
    user_prompt = f"설정: {setting}\n줄거리: {synopsis[:500]}..."
    return call_llm(sys_prompt, user_prompt, json_format=True)