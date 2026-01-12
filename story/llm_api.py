import os
import json
import time
import uuid
import random
import re
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

# 공통 제약 조건 상수
#KOREAN_ONLY_RULE = "출력은 고유명사(지명, 인명 등 불가피한 경우)를 제외하고는 반드시 '한국어'로 작성해야 합니다. 영어를 섞어 쓰지 마세요."
KOREAN_ONLY_RULE = """
[필수 규칙]
[Output Rules]
1. Use ONLY Korean(Hangul). 
2. Translate ALL English words to Korean. (e.g., 'system' -> '시스템', 'pushed' -> '밀려났다')
3. Do NOT use Chinese characters.
4. Exception: Keep Proper Nouns (Names like 'V', 'Silverhand') in English if necessary.
"""
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

def call_llm(system_prompt, user_prompt, json_format=False, stream=False, max_tokens=4000, max_retries=3, timeout=120):
    # 시스템 프롬프트에 한국어 제약 조건 추가
    full_system_prompt = f"{system_prompt}\n\n{KOREAN_ONLY_RULE}"
    #ull_system_prompt = f"{user_prompt}\n\n----------------\n{KOREAN_ONLY_RULE}"
    
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
    except Exception as e:
        print(f"Neo4j Error: {e}")

    # 2. 클리셰 매칭
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
    # [수정] 모든 캐릭터 정보를 넘겨줍니다.
    twisted_synopsis = _generate_twisted_synopsis_data(
        story, accumulated_content, twist_node.chapter_phase, characters_info_json
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
    """
    
    # [Step 1] 장르 선정
    all_genres = Genre.objects.all()
    if not all_genres.exists():
        print("⚠️ DB에 장르 데이터가 없습니다.")
        return None
    
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

# ==========================================
# [나머지 내부 로직 함수들]
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
        "3. **선택된 클리셰의 '핵심 요약'과 '전개 가이드'를 충실히 따를 것.**\n"
        "4. **사용자 설정 우선**: 사용자가 입력한 구체적인 설정은 크게 변경하거나 생략하지 말고 최대한 이야기에 포함시키세요.\n"
        "5. 문장은 번역투가 아닌 자연스러운 한국어 소설체로 작성하세요."
    )
    
    cliche_detail = (
        f"제목: {cliche.title}\n"
        f"장르: {cliche.genre.name}\n"
        f"핵심 요약: {cliche.summary}\n"
        f"전개 가이드: {cliche.structure_guide}"
    )
    
    user_prompt = (
        f"★ 사용자 세계관 설정 (최우선 반영): {story.user_world_setting}\n"
        f"주인공: {p_name} ({p_desc})\n"
        f"----------------------------------------\n"
        f"★ 필수 적용 클리셰 정보 ★\n{cliche_detail}\n"
        f"----------------------------------------\n"
        "위 내용을 바탕으로 사용자의 설정을 충실히 반영한 전체 시놉시스를 작성해줘."
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
    NODES_PER_PHASE = 3  
    TOTAL_NODES = NODES_PER_PHASE * 4  # 예: 5 * 4 = 20개

    # 총 생성 개수 계산
    needed_nodes = TOTAL_NODES - start_node_index

    phases = ["발단", "전개", "절정", "결말"]
    
    sys_prompt = (
        f"당신은 인터랙티브 스토리 작가입니다. 주인공 '{protagonist_name}'의 시점에서 장면(Node)들을 생성하세요.\n"
        "각 장면은 title, description(500자 이상), setting, purpose, characters_list, character_states, character_changes를 포함해야 합니다.\n\n"
        "**[중요]**\n"
        f"생성해야 할 노드의 개수는 총 {needed_nodes}개입니다.\n"
        "마지막 노드(Last Node)는 반드시 이야기의 **확실한 끝(Ending)**을 맺어야 합니다.\n"
        "어물쩍 넘어가거나 다음 이야기가 있는 것처럼 끝내지 말고, 확실한 결말을 지으세요."
    )
    user_prompt = f"시놉시스: {synopsis}\n생성 개수: {needed_nodes}개\nJSON 형식: {{'scenes': [...]}}"
    
    res = call_llm(sys_prompt, user_prompt, json_format=True, stream=True, max_tokens=8000)
    scenes = res.get('scenes', [])
    
    nodes = []
    for i, scene_data in enumerate(scenes):
        current_idx = start_node_index + i
        phase_name = phases[min(current_idx // NODES_PER_PHASE, 3)]
        
        node = StoryNode.objects.create(
            story=story, 
            chapter_phase=phase_name, 
            content=scene_data.get('description', '')
        )

        changes_json = json.dumps(scene_data.get('character_changes', {}), ensure_ascii=False)
        node.temp_character_changes = changes_json

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
                    depth=current_idx
                )
                sync_node_to_neo4j(neo4j_data)
            except Exception as e:
                print(f"Neo4j Node Sync Error: {e}")
    return nodes

def _connect_linear_nodes(nodes, universe_id, protagonist_name):
    # [수정] 자연스러운 연결을 위한 프롬프트 강화
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

def _find_twist_point_index(nodes):
    if len(nodes) < 4:
        return 0 if len(nodes) < 2 else 1

    summaries = [f"Idx {i}: {n.content[:50]}..." for i, n in enumerate(nodes[:-2])]
    res = call_llm("비틀기 지점(Index) 선택", "\n".join(summaries), json_format=True)
    idx = res.get('index', 2)
    return max(1, min(idx, len(nodes)-3))

def _generate_twisted_synopsis_data(story, acc_content, phase, characters_info_json):
    # [수정] 전체 캐릭터 정보를 반영하고 확실한 결말을 요구
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
    # [수정] 결과 연결성 강화
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