# story/llm_api.py

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

GENRE_NAMING_GUIDE = {
    "로맨스": (
        "Create trendy, sentimental, and sophisticated modern Korean names "
        "typical of protagonists in K-Dramas or Romance Webtoons. "
        "Avoid old-fashioned names. (e.g., 'Seon-jae', 'Yi-seo', 'Gu-won')"
    ),
    "판타지": (
        "Use elegant, aristocratic Western-style names often found in "
        "'Romance Fantasy' (RoFan) webtoons or Western fantasy novels. "
        "They should sound noble and graceful. (e.g., 'Callisto', 'Penelope', 'Arwin')"
    ),
    "무협": (
        "Use weighty Sino-Korean names or prestigious clan names "
        "typical of traditional Wuxia (Murim) novels. "
        "They should sound strong and classical. (e.g., 'Cheong-myeong', 'Hwa-san', 'Namgung')"
    ),
    "SF": (
        "Use names with a multinational feel or mix with code names/aliases, "
        "typical of Cyberpunk games or Sci-Fi movies. (e.g., 'V', 'K', 'David')"
    ),
    "추리/미스터리": (
        "Use realistic Korean names that sound ordinary but imply a hidden backstory or secrets, "
        "like characters in Korean thriller movies or crime dramas."
    ),
    "호러": (
        "Use realistic Korean names that evoke a somewhat chilly, sensitive, or nervous atmosphere, "
        "suitable for a horror setting."
    ),
}

def _clean_text_value(text):
    if not isinstance(text, str): return text
    text = re.sub(r'[\u4e00-\u9fff\u3040-\u30ff]+', '', text)
    text = re.sub(r'\([A-Za-z\s]+\)', '', text)
    def _remove_lowercase_english(match):
        word = match.group()
        if word[0].islower(): return ""
        return word
    text = re.sub(r'[A-Za-z]+', _remove_lowercase_english, text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def _clean_data_recursive(data):
    if isinstance(data, dict):
        return {k: _clean_data_recursive(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_clean_data_recursive(v) for v in data]
    elif isinstance(data, str):
        return _clean_text_value(data)
    else:
        return data

def call_llm(system_prompt, user_prompt, json_format=False, stream=False, max_tokens=4000, max_retries=3, timeout=300, temperature=0.7):
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
                temperature=temperature, 
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

            if json_format:
                cleaned_str = content.replace("```json", "").replace("```", "").strip()
                try:
                    parsed_data = json.loads(cleaned_str)
                except json.JSONDecodeError:
                    end_idx = cleaned_str.rfind("}")
                    if end_idx != -1:
                         parsed_data = json.loads(cleaned_str[:end_idx+1])
                    else:
                        raise
                return _clean_data_recursive(parsed_data)
            else:
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
    1. 유저 인풋 기반 클리셰 매칭
    2. [주인공], [인물1]... 임시 이름을 사용하여 시놉시스 생성
    3. [신규] 시놉시스와 인풋을 기반으로 세계관 세부 설정(시공간, 규칙, 문화) 생성
    4. 생성된 시놉시스와 세계관 설정을 반영하여 인물 상세 프로필 생성
    5. 시놉시스의 임시 이름을 실제 이름으로 치환하여 최종 저장
    """
    universe_id = str(uuid.uuid4())
    print(f"\n🌍 [NEO4J] Creating Universe Node: {universe_id}")

    # [Step 1] 클리셰 및 장르 매칭
    print("  [Step 1] Analyzing Genre & Matching Cliche...")
    matched_cliche = _match_cliche(user_world_setting)
    current_genre_name = matched_cliche.genre.name
    print(f"  -> Matched Genre: {current_genre_name} / Cliche: {matched_cliche.title}")

    # [Step 2] 임시 이름을 사용한 시놉시스 생성
    print("  [Step 2] Generating Synopsis with Temporary Names...")
    temp_synopsis = _generate_temp_synopsis(user_world_setting, matched_cliche)
    print(f"  -> Temp Synopsis Generated (Length: {len(temp_synopsis)})")

    # [Step 2.5] 세계관 세부 설정 생성 (신규 추가)
    print("  [Step 2.5] Generating Detailed World Settings...")
    detailed_world_setting = _generate_detailed_world_setting(user_world_setting, temp_synopsis)
    
    # Neo4j 저장을 위해 JSON 문자열로 변환
    world_setting_json_str = json.dumps(detailed_world_setting, ensure_ascii=False)
    print("  -> World Settings Generated.")

    # [Step 3] 시놉시스 + 세계관 설정 기반 캐릭터 상세 생성
    print("  [Step 3] Creating Detailed Characters based on Synopsis & World Context...")
    # 인자에 detailed_world_setting 추가
    character_map_list = _generate_character_mapping(temp_synopsis, current_genre_name, detailed_world_setting)
    
    # 데이터 처리 및 주인공 정보 추출
    protagonist_name = "주인공"
    name_map = {}
    characters_info_for_db = []
    
    for char in character_map_list:
        placeholder = char.get('placeholder')
        real_name = char.get('real_name')
        profile = char.get('profile', {}) 
        
        # 1. 이름 치환 맵핑
        if placeholder and real_name:
            name_map[placeholder] = real_name
            if placeholder == "[주인공]":
                protagonist_name = real_name

        # 2. DB/Neo4j 저장용 리스트 구성
        characters_info_for_db.append({
            "name": real_name,
            "role": placeholder,
            "profile": profile
        })

    print(f"  -> Protagonist Defined: {protagonist_name}")

    # [Step 4] 시놉시스 내 임시 이름 치환
    print("  [Step 4] Replacing Names in Synopsis...")
    final_synopsis = temp_synopsis
    for placeholder, real_name in name_map.items():
        final_synopsis = final_synopsis.replace(placeholder, real_name)
    
    # Django DB 저장
    story = Story.objects.create(user_world_setting=user_world_setting, main_cliche=matched_cliche)
    story.synopsis = final_synopsis
    story.save()

    # Neo4j Universe 생성 (world_setting 대신 상세 설정 JSON 전달)
    try:
        # 기존 user_world_setting 대신 상세 생성된 world_setting_json_str을 저장
        create_universe_node_neo4j(universe_id, world_setting_json_str, protagonist_name)
    except Exception as e:
        print(f"⚠️ Neo4j Universe Creation Failed: {e}")
        pass

    # 캐릭터 정보 JSON 변환
    characters_info_json = json.dumps(characters_info_for_db, ensure_ascii=False)

    # 3.5 Neo4j 업데이트 (Universe Details)
    universe_details = _generate_universe_details(user_world_setting, final_synopsis)
    try:
        update_universe_details_neo4j(
            universe_id=universe_id, 
            synopsis=final_synopsis, 
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

    # 4. 메인 경로 노드 생성
    print("  [Step 5] Creating Main Path Nodes...")
    main_nodes = _generate_path_segment(
        story, final_synopsis, protagonist_name, 
        start_node=None, universe_id=universe_id,
        characters_info_json=characters_info_json 
    )
    
    if not main_nodes: raise ValueError("메인 노드 생성 실패")
    try: link_universe_to_first_scene(universe_id, f"{universe_id}_{main_nodes[0].id}")
    except: pass

    # 5. 재귀적 분기 생성 시작
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
# 시놉시스 , 세계관, 캐릭터 생성 로직
# ==========================================

def _generate_temp_synopsis(setting, cliche):
    """
    구체적인 이름 없이 [주인공], [인물1], [인물2]... 형태의 플레이스홀더를 사용하여 시놉시스를 작성합니다.
    """
    sys_prompt = (
        "당신은 베스트셀러 웹소설 작가입니다. "
        "주어진 설정과 클리셰를 바탕으로 기승전결이 완벽한 시놉시스를 작성하세요.\n\n"
        "**[중요 규칙: 임시 이름 사용]**\n"
        "1. 등장인물의 이름을 **절대** 짓지 마세요.\n"
        "2. 주인공은 반드시 **'[주인공]'**이라고 표기하세요.\n"
        "3. 그 외 등장인물은 **'[인물1]', '[인물2]', '[인물3]'...** 순서로 표기하세요.\n"
        "   (예: [주인공]은 [인물1]을 만나 사랑에 빠지지만, [인물2]의 방해를 받는다.)\n"
        "4. 이 규칙을 어기고 임의로 이름을 지으면 안 됩니다.\n\n"
        "**[작성 가이드]**\n"
        "1. 분량은 6000자 이상.\n"
        "2. 클리셰의 '핵심 요약'과 '전개 가이드'를 충실히 따를 것.\n"
        "3. 사용자의 세계관 설정을 최대한 반영할 것."
    )
    
    cliche_detail = (
        f"장르: {cliche.genre.name}\n"
        f"클리셰 제목: {cliche.title}\n"
        f"핵심 요약: {cliche.summary}\n"
        f"전개 가이드: {cliche.structure_guide}"
    )
    
    user_prompt = (
        f"★ 사용자 세계관 설정: {setting}\n"
        f"----------------------------------------\n"
        f"★ 필수 적용 클리셰 정보 ★\n{cliche_detail}\n"
        f"----------------------------------------\n"
        "위 내용을 바탕으로 '[주인공]'과 번호 붙은 임시 인물명('[인물1]', '[인물2]'... 이러한 형식)을 사용하여 전체 시놉시스를 작성해줘."
    )
    
    # 일반 텍스트 모드로 생성
    return call_llm(sys_prompt, user_prompt, stream=True, max_tokens=16000)

def _generate_detailed_world_setting(user_input, synopsis):
    """
    시놉시스와 유저 인풋을 기반으로 구체적인 세계관 세부 사항을 JSON으로 생성합니다.
    """
    sys_prompt = (
        "당신은 세계관 설정(World Building) 전문가입니다. "
        "사용자의 기초 설정과 앞서 생성된 시놉시스를 철저히 분석하여, 이야기가 진행될 세계의 **구체적이고 입체적인 세부 설정**을 확장하세요.\n\n"
        "**[생성해야 할 3가지 카테고리]**\n"
        "1. **시공간 설정 (time_space)**: 시대적 배경(역사), 지리적 특성(지역명, 기후, 생태계), 해당 세계만의 물리 법칙 등.\n"
        "2. **법과 규칙 (laws_rules)**: 통치 체제, 핵심 법률, 마법/기술의 구동 원리 및 한계, 사회 계급 구조.\n"
        "3. **문화 및 가치관 (culture_values)**: 종교, 지배적인 철학, 도덕 관념, 전통/풍습, 종족적 특성, 세계를 바라보는 관점.\n\n"
        "**[출력 규칙]**\n"
        "- 반드시 아래 JSON 포맷을 준수하세요.\n"
        "- 내용은 추상적이지 않고 구체적이어야 합니다. (예: '마법이 있다' -> '마나를 혈관에 순환시켜 발동하며 과다 사용 시 혈관이 파열된다')\n"
        "- 시놉시스 사건의 개연성을 뒷받침해야 합니다."
    )
    
    user_prompt = (
        f"### [1] 사용자 기초 설정\n{user_input}\n\n"
        f"### [2] 전체 시놉시스\n{synopsis}\n\n"
        f"----------------------------------------\n"
        f"위 내용을 바탕으로 세계관 세부 설정을 JSON으로 생성하세요.\n"
        f"JSON 키: time_space, laws_rules, culture_values"
    )

    return call_llm(sys_prompt, user_prompt, json_format=True, max_tokens=4000)


def _generate_character_mapping(synopsis, genre_name, world_details):
    """
    시놉시스와 '세계관 세부 설정'을 반영하여 인물 프로필을 생성합니다.
    """
    naming_style = GENRE_NAMING_GUIDE.get(genre_name, "장르에 어울리는 매력적인 이름")
    world_context_str = json.dumps(world_details, ensure_ascii=False, indent=2)

    sys_prompt = (
        "당신은 캐릭터 메이킹 전문가입니다. "
        "시놉시스와 **확장된 세계관 설정**을 깊이 있게 분석하여, "
        "등장인물([주인공], [인물1]...)들에게 **실제 이름**과 **상세 프로필**을 부여하세요.\n\n"
        "**[중요: 세계관 반영]**\n"
        "캐릭터의 직업, 신념, 특기 등은 반드시 **제공된 세계관(법칙, 문화, 종족 특성)**과 밀접하게 연관되어야 합니다.\n"
        "(예: 마법이 금지된 세계라면 -> 신념: '마법 자유화', 특기: '숨겨진 마나 제어')\n\n"
        "**[필수 상세 항목 (10가지)]**\n"
        "1. **이름**: 장르에 어울리는 이름 (가이드 참고)\n"
        "2. **성격**: MBTI나 구체적인 성향 묘사\n"
        "3. **신념**: 삶을 살아가는 원동력 (세계관 가치관 반영)\n"
        "4. **목표**: 스토리 내 목표\n"
        "5. **가치관**: 중요하게 여기는 가치\n"
        "6. **인간관계 스타일**: 타인을 대하는 태도\n"
        "7. **좋아하는 것**\n"
        "8. **싫어하는 것**\n"
        "9. **취미**: 세계관 내에서 가능한 취미\n"
        "10. **특기**: 스토리 해결 능력 (세계관 기술/마법 반영)\n\n"
        f"**[작명 가이드]**: {naming_style}\n\n"
        "**[출력 형식]**\n"
        "반드시 아래 JSON 리스트 포맷을 준수하세요:\n"
        "[\n"
        "  {\n"
        "    'placeholder': '[주인공]',\n"
        "    'real_name': '이름',\n"
        "    'profile': {\n"
        "      'personality': '...',\n"
        "      'beliefs': '...',\n"
        "      'goal': '...',\n"
        "      'values': '...',\n"
        "      'relationship_style': '...',\n"
        "      'likes': '...',\n"
        "      'dislikes': '...',\n"
        "      'hobbies': '...',\n"
        "      'specialties': '...'\n"
        "    }\n"
        "  }, ...\n"
        "]"
    )
    
    user_prompt = (
        f"### [1] 세계관 세부 설정 (Context)\n{world_context_str}\n\n"
        f"### [2] 시놉시스\n{synopsis}\n\n"
        "위 세계관 속에서 살아 숨 쉬는 입체적인 캐릭터들을 생성하세요."
    )
    
    # JSON 모드로 호출
    res = call_llm(sys_prompt, user_prompt, json_format=True, max_tokens=4000)
    
    if isinstance(res, dict) and 'characters' in res:
        return res['characters']
    elif isinstance(res, list):
        return res
    else:
        return []


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

# _refine_setting_and_protagonist 함수는 더 이상 사용되지 않으므로 삭제하거나 무시해도 됩니다.
# _generate_synopsis 함수도 _generate_temp_synopsis로 대체되었습니다.

# [핵심 로직: DFS 재귀적 스토리 생성], [보조 함수들] 등은 기존 코드 유지
def _generate_recursive_story(story, current_path_nodes, quota, universe_id, protagonist_name, characters_info_json, hierarchy_id, twist_synopsis=None):
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
        new_branch_nodes = _generate_path_segment(
            story, twisted_synopsis, protagonist_name,
            start_node=target_node, universe_id=universe_id, is_twist_branch=True,
            characters_info_json=characters_info_json
        )

        if new_branch_nodes:
            original_choice = target_node.choices.first()
            original_action = original_choice.choice_text if original_choice else "원래대로 진행"
            _create_twist_condition(
                target_node, 
                new_branch_nodes[0], 
                universe_id, 
                protagonist_name, 
                original_action,
                twist_synopsis=twisted_synopsis 
            )
            
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

# characters_info_json 인자 추가
def _generate_path_segment(story, synopsis, protagonist_name, start_node=None, universe_id=None, is_twist_branch=False, characters_info_json="[]"):
    start_depth = start_node.depth if start_node else 0
    next_depth = start_depth + 1
    
    needed_nodes = TOTAL_DEPTH_PER_PATH - start_depth
    if needed_nodes < 1: needed_nodes = 1 

    # start_node를 기반으로 '전체 히스토리(심경 변화 포함)' 추출
    initial_history = _get_full_history(start_node)

    # _create_nodes_common에 히스토리와 캐릭터 정보 전달
    nodes = _create_nodes_common(
        story, synopsis, protagonist_name, needed_nodes, next_depth, universe_id,
        initial_history=initial_history,
        characters_info_json=characters_info_json
    )
    
    if not nodes: return []

    _connect_linear_nodes(nodes, universe_id, protagonist_name)
    
    return nodes

# 노드 생성 공통 함수: 직전 장면 전문 전달
def _create_nodes_common(story, synopsis, protagonist_name, count, start_depth, universe_id, initial_history="", characters_info_json="[]"):
    phases = ["발단", "전개", "절정", "결말"]
    BATCH_SIZE = 1
    
    created_nodes = []
    generated_count = 0
    
    # 이번 세션에서 생성된 노드들의 히스토리를 누적 저장할 리스트
    current_session_history = [] 

    normal_node_count = count - 1 if count > 0 else 0
    
    print(f"    🔄 [Generation Plan] Total: {count} | Normal Batch: {normal_node_count} | Final Ending: 1")

    # --- 내부 함수: 프롬프트 생성기 ---
    def build_prompt(batch_size, is_ending=False):
        # 1. 전체 흐름 구성 (Action 포함된 히스토리)
        full_history_text = initial_history
        if current_session_history:
            session_hist_text = "\n\n".join(current_session_history)
            if full_history_text:
                full_history_text += f"\n\n{session_hist_text}"
            else:
                full_history_text = session_hist_text
        
        # 2. 직전 상황 전달: 요약/발췌 없이 '전문(Full Text)' 전달
        prev_context_full = ""
        if created_nodes:
            last = created_nodes[-1]
            # 전체 내용 전달
            prev_context_full = last.content 
        elif initial_history:
             # initial_history의 마지막 부분이 직전 노드의 전체 내용임
             prev_context_full = "(위 '전체 줄거리 흐름'의 가장 마지막 장면을 전체 내용으로 참고하세요.)"
        
        sys = (
            f"당신은 인터랙티브 스토리 작가입니다. 주인공 '{protagonist_name}'의 시점에서 장면(Node)들을 생성하세요.\n"
            "**[입력 데이터 설명]**\n"
            "1. **전체 줄거리 흐름**: 이야기의 시작부터 직전까지의 모든 사건, **수행한 필수 행동(Action)**, 인물 심경 변화가 **요약 없이** 포함되어 있습니다. 흐름을 완벽히 숙지하세요.\n"
            "2. **직전 장면**: 바로 앞 장면의 **전체 내용**입니다. 문맥이 끊기지 않게 자연스럽게 이어가세요.\n"
            "3. **현재 시놉시스**: 이번 구간의 핵심 목표입니다.\n\n"
            "**[출력 필수 항목]**\n"
            "각 장면은 title, description(2000자 이상), setting, purpose, characters_list, character_states, character_changes를 포함해야 합니다.\n"
            "**분량 엄수**: description은 반드시 공백 포함 2000자 이상이어야 합니다. 풍부한 묘사와 대사를 포함하세요.\n"
        )
        
        if is_ending:
            sys += "**[엔딩 생성 모드]** 확실하고 닫힌 결말(Closed Ending)을 1개 작성하세요.\n"
            req_count_str = "1개 (엔딩)"
        else:
            sys += f"**[일반 진행 모드]** 정확히 {batch_size}개의 장면을 이어서 작성하세요.\n"
            req_count_str = f"{batch_size}개"

        user = (
            f"### [1] 등장인물 정보 및 특성\n{characters_info_json}\n\n"
            f"### [2] 현재 적용 시놉시스\n{synopsis}\n\n"
            f"### [3] 전체 줄거리 흐름 (행동/심경 변화 포함, 요약 없음)\n{full_history_text}\n\n"
            f"### [4] 직전 장면 내용 (전문, Full Text)\n{prev_context_full}\n\n"
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
            res = call_llm(sys_prompt, user_prompt, json_format=True, stream=True, max_tokens=16000, timeout=300)
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

            # 세션 히스토리 누적 시에도 '요약 없이' 전체 내용 저장
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
            res = call_llm(sys_prompt, user_prompt, json_format=True, stream=True, max_tokens=16000, timeout=300)
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

def _get_full_history(node):
    if not node: return ""
    history_list = []
    curr = node
    while curr:
        changes = getattr(curr, 'temp_character_changes', '')
        
        # 1. 현재 노드로 오기 위해 수행한 '필수 행동' 조회
        action_text = ""
        if curr.prev_node:
            try:
                # curr.prev_node에서 curr로 연결된 선택지(Action) 찾기
                choice = NodeChoice.objects.filter(current_node=curr.prev_node, next_node=curr).first()
                if choice:
                    action_text = f"\n[▼ 수행한 행동: {choice.choice_text}]"
            except Exception:
                pass
        
        # 2. 요약 없는 전체 내용 구성
        info = f"{action_text}\n[장면 {curr.depth} ({curr.chapter_phase})]\n내용: {curr.content}"
        
        # 심경 변화가 있다면 함께 기록
        if changes and changes != "{}" and changes != "null":
             info += f"\n(인물 심경 변화: {changes})"
             
        history_list.append(info)
        curr = curr.prev_node
        
    # 과거 -> 현재 순으로 정렬하여 반환
    return "\n\n".join(reversed(history_list))

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

def _connect_linear_nodes(nodes, universe_id, protagonist_name):
    sys_prompt = (
        f"주인공 '{protagonist_name}'이 **'현재 장면'의 서술이 모두 끝난 직후**, 다음 장면으로 넘어가기 위해 취해야 할 행동을 정의하세요. 꼭 '{protagonist_name}'이 주체여야 합니다.\n"
        "1. **[중요] 시점 원칙**: '현재 장면'에 서술된 내용은 이미 다 일어난 일입니다. 행동은 그 **이후**에 벌어질 일이어야 합니다.\n"
        "2. **[중요] 중복 금지**: 현재 장면 본문에 이미 묘사된 행위(예: 짐을 풀었다, 대화를 나눴다 등)를 다시 행동으로 제시하지 마세요.\n"
        "3. 행동은 구체적이지 않고 단순하고 직관적이어야 합니다. (예: '방을 나선다', '대답한다', '주위를 살핀다')\n"
        "4. 행동의 결과(result)는 다음 장면의 첫 문장과 내용상 자연스럽게 이어져야 합니다."
    )
    
    for i in range(len(nodes) - 1):
        curr = nodes[i]
        next_n = nodes[i+1]
        
        curr.prev_node = next_n.prev_node 
        next_n.prev_node = curr
        next_n.save()
        
        user_prompt = (
            f"### [1] 현재 장면 (이미 완료된 상황)\n{curr.content}\n"
            f"(설명: 위 내용은 이미 진행되었습니다. 주인공은 이 상황 끝에 놓여 있습니다.)\n\n"
            f"### [2] 다음 장면 (이어질 내용)\n{next_n.content}...\n\n"
            f"--------------------------------------------------\n"
            f"위 두 장면 사이를 연결하는 '유저 행동(Action)'과 '그 직후의 결과(Result)'를 생성하세요.\n"
            f"Q: 현재 장면의 상황이 모두 끝난 후, 주인공이 무엇을 해야 다음 장면으로 넘어갑니까?\n"
            f"출력 JSON: {{'action': '유저가 할 행동', 'result': '행동 직후 묘사(다음 장면 도입부와 이어지는 내용)'}}"
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
        "1. 분량은 6000자 이상.\n"
        "2. **제공된 모든 주요 등장인물의 성격과 특성을 전부 수정사항 없이 반영하여 스토리 흐름의 입체적인 변화를 주세요.**\n"
        "3. 단순히 상황만 꼬는 것이 아니라, **확실한 결말(Closed Ending)**을 맺어야 합니다.\n"
        "4. 등장인물의 특성을 임의로 변경하면 안됩니다. twist synopsis는 모든 등장인물의 성격, 특성을 전부 고려하였을 때 말이 되도록 작성해야 합니다. (ex.재벌결혼을 주장하는 아버지가 갑자기 진정한 사랑이라는 이유로 결혼을 허락하면 안됨. 인물의 신념에 위배.)"
    )
    user_prompt = (
        f"현재까지 진행된 이야기: {acc_content}\n"
        f"현재 단계: {phase} (이 지점부터 이야기가 달라집니다)\n"
        f"등장인물 상세 정보: {characters_info_json}\n\n"
        "위 정보를 바탕으로 완결된 형태의 비틀린 시놉시스를 작성해주세요."
        "기존의 시놉시스에서 과도하게 벗어나지 말고, 약간만 결과를 바꿔주세요."
    )
    return call_llm(sys_prompt, user_prompt, stream=True, max_tokens=16000, timeout=300)

def _create_twist_condition(node, twist_next_node, universe_id, protagonist_name, original_action_text, twist_synopsis=None):
    sys_prompt = (
        f"현재 장면이 끝난 시점에서, 이야기가 완전히 다른 방향(반전)으로 흐르기 위해 "
        f"주인공 '{protagonist_name}'이 수행해야 할 **돌발적인 조건 행동(Twist Action)**을 정의하세요.\n"
        "1. '현재 장면'에 이미 나온 내용은 행동으로 쓰지 마세요. 행동은 현재 장면이 끝난 **다음**에 발생합니다.\n"
        "2. 기존의 정석적인 행동과는 의도가 명확히 달라야 합니다.\n"
        "3. 행동의 결과(result)는 반전된 다음 장면의 시작 부분과 자연스럽게 이어져야 합니다."
    )
    
    user_prompt = (
        f"### [1] 현재 장면 (완료된 상황): ...{node.content[-500:]}\n"
        f"### [2] 반전된 다음 장면 (시작 부분): {twist_next_node.content[:300]}...\n"
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
                character_changes=twist_changes,
                twist_synopsis=twist_synopsis 
            )
        except: pass

def _generate_universe_details(setting, synopsis):
    """
    매력적이고 마케팅적인 제목과 설명 생성
    """
    sys_prompt = (
        "당신은 베스트셀러 소설의 편집자이자 천재적인 마케터입니다. "
        "주어진 세계관과 전체 시놉시스를 분석하여, 독자(플레이어)의 호기심을 강하게 자극하는 매력적인 정보를 JSON으로 생성하세요.\n\n"
        "**[작성 가이드]**\n"
        "1. **title (제목)**:\n"
        "   - 촌스러운 설명조(예: '철수의 모험', '조선시대 좀비물')는 절대 금지입니다.\n"
        "   - **은유적, 상징적, 시적인 표현**을 사용하여 여운과 임팩트를 주세요.\n"
        "   - 모순된 단어의 조합이나 강렬한 이미지를 사용하세요. (예: '달빛이 닿지 않는 왕좌', '기계 심장의 고동', '내일이 없는 소녀')\n\n"
        "2. **description (한 줄 소개)**:\n"
        "   - 유저가 홀린 듯이 플레이 버튼을 누르게 만드는 **강력한 훅(Hook)** 문장입니다. (100자 이내)\n"
        "   - 주인공이 처한 아이러니한 상황이나, 이야기의 가장 흥미로운 딜레마를 질문형이나 권유형으로 던지세요.\n\n"
        "3. **detail_description (상세 소개)**:\n"
        "   - 줄거리를 건조하게 요약하지 마세요. **영화 예고편의 내레이션**처럼 작성하세요.\n"
        "   - 세계관의 독특한 분위기(Atmosphere)와 주인공의 시련을 강조하여 긴장감을 조성하세요.\n\n"
        "4. **JSON 필드**: title, description, detail_description, estimated_play_time_min (int), estimated_play_time_max (int)"
    )
    
    # 요약본이 아닌 '전체 시놉시스'를 전달하여 맥락 전체 파악 유도
    user_prompt = f"세계관 설정: {setting}\n\n전체 시놉시스(Full Text): {synopsis}"
    
    return call_llm(sys_prompt, user_prompt, json_format=True, temperature=0.8)