import os
import json
import time
import uuid
import random
from openai import OpenAI
from django.conf import settings
# StoryBranch 추가 임포트
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
# [설정 변수: 이 값을 변경하여 스토리 규모 조절]
# ==========================================
TARGET_BRANCH_COUNT = 2      # 추가로 생성할 분기(엔딩)의 수. (기본 1개 + 추가 2개 = 총 3개 엔딩. 3으로 하면 총 4개)
TOTAL_DEPTH_PER_PATH = 12    # 시작부터 엔딩까지 이어지는 노드의 총 개수 (길이)

def call_llm(system_prompt, user_prompt, json_format=False, stream=False, max_tokens=4000, max_retries=3, timeout=120):
    # 시스템 프롬프트에 한국어 제약 조건 추가
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
    스토리 생성 전체 파이프라인 (재귀적 분기 생성 구조 적용)
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

    # 2. 클리셰 매칭
    matched_cliche = _match_cliche(refined_setting)
    if not matched_cliche: raise ValueError("클리셰 매칭 실패")
    
    print(f"✅ Matched Cliche: [{matched_cliche.genre.name}] {matched_cliche.title}")

    story = Story.objects.create(user_world_setting=refined_setting, main_cliche=matched_cliche)
    
    # 3. 메인 시놉시스 생성 (Main Stem)
    print("  [Step 3] Generating Root Synopsis...")
    root_synopsis = _generate_synopsis(story, matched_cliche, protagonist_name, protagonist_info['desc'])
    story.synopsis = root_synopsis
    story.save()

    # 3.5 주요 인물 정보 추출 및 Universe 업데이트
    print("  [Step 3.5] Extracting Characters & Universe Details...")
    universe_details = _generate_universe_details(refined_setting, root_synopsis)
    characters_info_json = _extract_characters_info(root_synopsis, protagonist_info)
    
    try:
        update_universe_details_neo4j(
            universe_id=universe_id,
            synopsis=root_synopsis,
            twisted_synopsis="", # 초기엔 없음
            title=universe_details.get("title", "무제"),
            description=universe_details.get("description", ""),
            detail_description=universe_details.get("detail_description", ""),
            play_time=universe_details.get("play_time", "30분"),
            characters_info=characters_info_json
        )
    except Exception as e:
        print(f"Neo4j Update Error: {e}")

    # 4. 메인 경로 노드 생성 (Start -> Ending)
    print("  [Step 4] Creating Main Path Nodes...")
    main_nodes = _generate_path_segment(
        story, root_synopsis, protagonist_name, 
        start_node=None, universe_id=universe_id
    )
    
    if not main_nodes: raise ValueError("메인 노드 생성 실패")
    
    # Neo4j 연결 (Start)
    try:
        link_universe_to_first_scene(universe_id, f"{universe_id}_{main_nodes[0].id}")
    except: pass

    # 5. 분기 생성 루프 (목표 개수만큼 반복)
    current_branches = 0
    
    while current_branches < TARGET_BRANCH_COUNT:
        print(f"\n🌿 [Branching {current_branches + 1}/{TARGET_BRANCH_COUNT}] Generating Twist...")

        # 5-1. 분기할 후보 노드 선정 (LLM 활용)
        target_node = _select_branch_node_with_llm(story)
        
        if not target_node:
            print("⚠️ 더 이상 분기할 적절한 노드가 없습니다.")
            break
            
        print(f"  -> Selected Twist Point: Node {target_node.id} ({target_node.chapter_phase})")

        # 5-2. 해당 지점까지의 스토리 맥락 복원
        history_context = _get_story_history(target_node)
        
        # 5-3. 비틀기 시놉시스 생성
        # [중요] 기존의 _generate_twisted_synopsis_data 함수와 프롬프트를 그대로 사용
        twisted_synopsis = _generate_twisted_synopsis_data(
            story, history_context, target_node.chapter_phase, characters_info_json
        )
        
        # 분기 정보 저장
        StoryBranch.objects.create(story=story, parent_node=target_node, synopsis=twisted_synopsis)

        # 5-4. 새로운 가지 노드 생성 (Target Node 뒤부터 엔딩까지)
        print("  -> Creating Branch Nodes...")
        new_branch_nodes = _generate_path_segment(
            story, twisted_synopsis, protagonist_name,
            start_node=target_node, universe_id=universe_id, is_twist_branch=True
        )

        # 5-5. 분기점 연결 (Twist Action 생성)
        if new_branch_nodes:
            # 기존 선택지 텍스트를 참고용으로 가져옴
            original_choice = target_node.choices.first()
            original_action_text = original_choice.choice_text if original_choice else "다음으로 진행"

            # [중요] 기존의 _create_twist_condition 로직 사용
            _create_twist_condition(
                target_node, 
                new_branch_nodes[0], 
                universe_id, 
                protagonist_name, 
                original_action_text
            )
            
        current_branches += 1

    return story.id


# ==========================================
# [내부 로직: 재귀적 생성 지원 함수]
# ==========================================

def _generate_path_segment(story, synopsis, protagonist_name, start_node=None, universe_id=None, is_twist_branch=False):
    """
    특정 지점(start_node)부터 엔딩까지 이어지는 노드들을 생성하고 선형으로 연결합니다.
    """
    # 1. 시작 깊이 계산
    start_depth = start_node.depth if start_node else 0
    next_depth = start_depth + 1
    
    # 2. 필요한 노드 수 계산 (전체 길이 - 현재 깊이)
    needed_nodes = TOTAL_DEPTH_PER_PATH - start_depth
    if needed_nodes < 1: needed_nodes = 1 # 최소 1개는 생성

    # 3. 노드 생성 (LLM 호출)
    nodes = _create_nodes_common(story, synopsis, protagonist_name, needed_nodes, next_depth, universe_id)
    
    if not nodes: return []

    # 4. 생성된 노드들끼리 선형 연결 (Linear Connection)
    # [중요] 기존의 _connect_linear_nodes 사용
    _connect_linear_nodes(nodes, universe_id, protagonist_name)
    
    return nodes

def _create_nodes_common(story, synopsis, protagonist_name, count, start_depth, universe_id):
    """
    _create_nodes_from_synopsis의 로직을 일반화하여, 지정된 개수(count)만큼 노드를 생성합니다.
    """
    phases = ["발단", "전개", "절정", "결말"]
    
    # [중요] 기존 프롬프트 원본 유지 (needed_nodes 변수만 count로 대체)
    sys_prompt = (
        f"당신은 인터랙티브 스토리 작가입니다. 주인공 '{protagonist_name}'의 시점에서 장면(Node)들을 생성하세요.\n"
        "각 장면은 title, description(500자 이상), setting, purpose, characters_list, character_states, character_changes를 포함해야 합니다.\n\n"
        "**[중요]**\n"
        f"생성해야 할 노드의 개수는 총 {count}개입니다.\n"
        "마지막 노드(Last Node)는 반드시 이야기의 **확실한 끝(Ending)**을 맺어야 합니다.\n"
        "어물쩍 넘어가거나 다음 이야기가 있는 것처럼 끝내지 말고, 확실한 결말을 지으세요."
    )
    # 기존 프롬프트 형식 유지
    user_prompt = f"시놉시스: {synopsis}\n생성 개수: {count}개\nJSON 형식: {{'scenes': [...]}}"
    
    res = call_llm(sys_prompt, user_prompt, json_format=True, stream=True, max_tokens=8000)
    scenes = res.get('scenes', [])
    
    created_nodes = []
    for i, scene_data in enumerate(scenes):
        current_depth = start_depth + i
        
        # 단계(Phase)를 전체 길이에 비례하여 계산 (비율 매핑)
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

        # 임시 데이터 저장 (연결 시 사용)
        changes_json = json.dumps(scene_data.get('character_changes', {}), ensure_ascii=False)
        node.temp_character_changes = changes_json

        created_nodes.append(node)
        
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
                    depth=current_depth
                )
                sync_node_to_neo4j(neo4j_data)
            except Exception as e:
                print(f"Neo4j Node Sync Error: {e}")
    return created_nodes


def _select_branch_node_with_llm(story):
    """
    현재 존재하는 노드 중 분기하기 가장 좋은 노드를 선택합니다.
    """
    # 후보군: '결말'이 아니고, 아직 선택지(Child)가 2개 미만인 노드 (이미 분기된 곳 제외)
    candidates = StoryNode.objects.filter(story=story, choices__count__lt=2).exclude(chapter_phase='결말')
    
    candidate_list = list(candidates)
    if not candidate_list: return None
    
    # 너무 적으면 바로 랜덤 반환
    if len(candidate_list) < 3:
        return random.choice(candidate_list)

    # 샘플링 (너무 초반이나 후반보다는 중간 위주)
    sampled_candidates = sorted(candidate_list, key=lambda n: n.id)[1:-1]
    if len(sampled_candidates) > 10:
        sampled_candidates = random.sample(sampled_candidates, 10)
    elif not sampled_candidates:
        sampled_candidates = candidate_list

    prompt_text = ""
    node_map = {}
    for n in sampled_candidates:
        prompt_text += f"[ID: {n.id}] Phase: {n.chapter_phase} | 내용: {n.content[:60]}...\n"
        node_map[n.id] = n

    sys_prompt = (
        "당신은 스토리 에디터입니다. 아래 장면 목록 중, 이야기의 흐름을 비틀어(Twist) "
        "새로운 분기를 만들기에 가장 흥미롭고 극적인 지점을 하나 선택하세요.\n"
        "반드시 JSON 형식 {'node_id': ID숫자} 로 응답하세요."
    )
    user_prompt = f"후보 장면들:\n{prompt_text}"

    try:
        res = call_llm(sys_prompt, user_prompt, json_format=True)
        selected_id = res.get('node_id')
        if selected_id and selected_id in node_map:
            return node_map[selected_id]
    except:
        pass
    
    return random.choice(sampled_candidates)

def _get_story_history(target_node):
    """
    루트 노드부터 target_node까지 거슬러 올라가며 내용을 복원함.
    """
    path_contents = []
    curr = target_node
    while curr:
        path_contents.append(curr.content)
        curr = curr.prev_node # 역추적
    
    # 역순이므로 뒤집어서 결합
    return "\n".join(reversed(path_contents))

# ==========================================
# [기존 로직 함수들 (프롬프트 원본 유지)]
# ==========================================

def _match_cliche(setting):
    """
    [2단계 매칭 로직] - 원본 프롬프트 복원
    """
    all_genres = Genre.objects.all()
    if not all_genres.exists():
        print("⚠️ DB에 장르 데이터가 없습니다.")
        return None
    
    # [Step 1] 장르 선정
    genre_text_list = []
    for g in all_genres:
        desc = g.description if g.description else "설명 없음"
        genre_text_list.append(f"- [{g.name}]: {desc}")
    
    genre_prompt_text = "\n".join(genre_text_list)
    
    sys_prompt_1 = (
        "당신은 장르 문학 분석가입니다. "
        "사용자의 입력(세계관 설정)을 분석하여, 아래 제공된 장르 목록 중 이를 가장 효과적으로 표현할 수 있는 **단 하나의 장르**를 선택하세요.\n"
        "반드시 JSON 형식 {'genre_name': '장르명', 'reason': '이유'} 으로만 응답하세요."
    )
    
    user_prompt_1 = f"사용자 설정: {setting}\n\n[장르 목록]\n{genre_prompt_text}"
    
    print("  [Step 1] Selecting Genre...")
    res_1 = call_llm(sys_prompt_1, user_prompt_1, json_format=True)
    selected_genre_name = res_1.get('genre_name', '판타지')
    
    try:
        selected_genre = Genre.objects.get(name=selected_genre_name)
    except Genre.DoesNotExist:
        selected_genre = all_genres.first()
        print(f"  ⚠️ Genre '{selected_genre_name}' not found. Fallback to '{selected_genre.name}'")

    print(f"  -> Selected Genre: {selected_genre.name}")

    # [Step 2] 클리셰 선정
    cliches = Cliche.objects.filter(genre=selected_genre)
    
    if not cliches.exists():
        return Cliche.objects.first()

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
        "각 클리셰의 '정의'와 '구조 가이드'를 면밀히 분석하여 결정하세요.\n"
        "응답은 JSON 형식 {'cliche_id': ID숫자, 'reason': '선택 이유'} 만 반환하세요."
    )
    
    user_prompt_2 = (
        f"사용자 설정: {setting}\n\n"
        f"[선택된 장르: {selected_genre.name}]\n\n"
        f"[클리셰 후보 목록]\n{cliche_prompt_text}"
    )

    print("  [Step 2] Selecting Cliche...")
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

def _connect_linear_nodes(nodes, universe_id, protagonist_name):
    # [중요] 원본 프롬프트 유지
    sys_prompt = (
        f"주인공 '{protagonist_name}'이 현재 장면에서 다음 장면으로 넘어가기 위해 취해야 할 **자연스럽고 일상적인 행동(Condition Action)**을 정의하세요.\n"
        "1. 유저가 별도의 힌트 없이도 상황상 자연스럽게 입력할 법한 행동이어야 합니다. (예: '문을 연다', '대답한다', '전화를 받는다') 행위가 구체적이면 안됩니다. 아주 일상적인 행동이어야 합니다. 마치 방탈출을 하는 게임 플레이어처럼 유저가 할 수 있을 법한 행동을 조건 행위로 지정해야 합니다.\n"
        "2. **조건 행동의 결과(result)는 다음 장면의 시작 부분과 자연스럽게 이어져야 합니다.**\n"
        "   - 시간적 흐름: Action(행동) -> Result(결과) -> Next Scene Start(다음 장면)\n"
        "   - 예시: 행동 '문을 연다' -> 결과 '문이 열리자 차가운 바람이 불어왔다.' -> 다음 장면 '방 안에는 아무도 없었다...'"
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
            "출력 JSON: {'action': '유저가 입력할 행동', 'result': '행동의 결과 (다음 장면 도입부로 자연스럽게 연결)'}"
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
    # [중요] 원본 프롬프트 유지 + 약간의 보강(인물 정보)
    sys_prompt = (
        "기존 스토리의 흐름을 비틀어 새로운 결말로 향하는 'Twist Synopsis'를 작성하세요.\n"
        "1. 분량은 2000자 이상.\n"
        "2. **제공된 모든 주요 등장인물의 성격과 특성을 반영하여 입체적인 변화를 주세요.**\n"
        "3. 단순히 상황만 꼬는 것이 아니라, **확실한 결말(Closed Ending)**을 맺어야 합니다.\n"
        "   - 열린 결말이나 흐지부지한 엔딩 금지.\n"
        "   - 비극이든 희극이든 이야기가 완결되어야 함."
    )
    user_prompt = (
        f"현재까지 진행된 이야기: {acc_content[-1000:]}\n"
        f"현재 단계: {phase} (이 지점부터 이야기가 달라집니다)\n"
        f"등장인물 상세 정보: {characters_info_json}\n\n"
        "위 정보를 바탕으로 완결된 형태의 비틀린 시놉시스를 작성해주세요."
    )
    return call_llm(sys_prompt, user_prompt, stream=True, max_tokens=8000)

def _create_twist_condition(node, twist_next_node, universe_id, protagonist_name, original_action_text):
    # [중요] 원본 프롬프트 유지
    sys_prompt = (
        f"현재 장면에서 이야기가 완전히 다른 방향(반전)으로 흐르기 위해, "
        f"주인공 '{protagonist_name}'이 수행해야 할 **돌발적이고 파격적인 조건 행동(Twist Action)**을 정의하세요.\n"
        "1. 기존의 정석적인 행동과는 의도가 명확히 달라야 합니다.\n"
        "2. **행동의 결과(result)는 반전된 다음 장면의 시작 부분과 자연스럽게 이어져야 합니다.**\n"
        "   - 시간적 흐름: Twist Action -> Result -> Twist Next Scene Start"
    )
    
    user_prompt = (
        f"현재 장면(마지막 부분): ...{node.content[-300:]}\n"
        f"반전된 다음 장면(시작 부분): {twist_next_node.content[:300]}...\n"
        f"참고(기존 정석 행동): '{original_action_text}'\n\n"
        "위 두 장면을 연결하는 반전 행동(Action)과 결과(Result)를 생성하세요.\n"
        "출력 JSON: {'action': '반전 행동', 'result': '행동의 결과 (다음 장면 도입부로 자연스럽게 연결)'}"
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
    sys_prompt = "세계관 상세 정보 JSON 생성 (title, description, detail_description, play_time)"
    user_prompt = f"설정: {setting}\n줄거리: {synopsis[:500]}..."
    return call_llm(sys_prompt, user_prompt, json_format=True)