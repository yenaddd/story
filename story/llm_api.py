import os
import json
import time
import uuid
from openai import OpenAI
from django.conf import settings
from .models import Genre, Cliche, Story, CharacterState, StoryNode, NodeChoice

# [추가] 수정된 Neo4j 함수 임포트
from .neo4j_connection import (
    create_universe_node_neo4j, 
    sync_node_to_neo4j, 
    link_universe_to_first_scene, 
    sync_choice_to_neo4j, 
    StoryNodeData
)

# API 설정 (기존 유지)
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
BASE_URL = "https://api.fireworks.ai/inference/v1"
MODEL_NAME = "accounts/fireworks/models/deepseek-v3p1" 
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=BASE_URL)

def call_llm(system_prompt, user_prompt, json_format=False, max_retries=3):
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    response_format = {"type": "json_object"} if json_format else None
    if not DEEPSEEK_API_KEY:
        print("🚨 [Critical] API Key is MISSING! (.env 파일을 확인해주세요)")
        return {} if json_format else ""

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME, 
                messages=messages, 
                response_format=response_format, 
                temperature=0.7, 
                max_tokens=4000, # DeepSeek V3는 컨텍스트가 길므로 여유 있게 설정
                timeout=60       # 응답 대기 시간
            )
            content = response.choices[0].message.content
            
            # JSON 포맷팅 응답 처리
            if json_format:
                cleaned = content.replace("```json", "").replace("```", "").strip()
                return json.loads(cleaned)
            return content

        except Exception as e:
            print(f"⚠️ [LLM Error] Attempt {attempt+1}/{max_retries} Failed: {str(e)}")
            time.sleep(1)
            
    print(f"❌ [Final Fail] LLM Call Failed completely.")
    return {} if json_format else ""

# ==========================================
# [메인 파이프라인]
# ==========================================

def create_story_pipeline(user_world_setting):
    # [Neo4j] 1. Universe ID 생성 및 세계관 노드 생성
    universe_id = str(uuid.uuid4())
    print(f"\n🌍 [NEO4J] Creating Universe Node: {universe_id}")
    try:
        create_universe_node_neo4j(universe_id, user_world_setting)
    except Exception as e:
        print(f"Neo4j Error: {e}")

    # (Django 로직: 클리셰 매칭 ~ 시놉시스 생성 - 기존 동일)
    matched_cliche = _match_cliche(user_world_setting)
    story = Story.objects.create(user_world_setting=user_world_setting, main_cliche=matched_cliche)
    synopsis = _generate_synopsis(story, matched_cliche)
    story.synopsis = synopsis
    story.save()
    _analyze_and_save_character_state(story, synopsis, context="Initial Synopsis")

    # [Neo4j & Django] 2. 첫 번째 챕터 노드들 생성
    original_nodes = _create_nodes_from_synopsis(story, synopsis, start_node_index=0)
    if not original_nodes or len(original_nodes) < 2:
        print(f"❌ [Error] 노드 생성 실패. 생성된 노드 수: {len(original_nodes)}")
        # 실패 시 예외를 발생시켜 로그에 명확히 남기고 중단
        raise ValueError("AI가 스토리 노드를 생성하지 못했습니다. (API 응답 오류 또는 파싱 실패)")  
    
    # [Neo4j] 3. 첫 번째 노드(Root Node)를 Universe와 연결
    if original_nodes:
        try:
            link_universe_to_first_scene(universe_id, original_nodes[0].id)
        except Exception as e:
            print(f"Neo4j Link Error: {e}")

    # [Neo4j & Django] 4. 선형 연결
    _connect_linear_nodes(original_nodes)

    # (비틀기 로직 - 기존 동일)
    twist_node_index = _find_twist_point_index(original_nodes)

    if twist_node_index >= len(original_nodes):
        twist_node_index = len(original_nodes) - 1
        
    twist_node = original_nodes[twist_node_index]
    twist_node = original_nodes[twist_node_index]
    story.twist_point_node_id = twist_node.id
    story.save()

    accumulated = "\n".join([n.content for n in original_nodes[:twist_node_index+1]])
    twist_cliche, twisted_synopsis = _generate_twisted_synopsis_data(story, accumulated, twist_node.chapter_phase)
    story.twist_cliche = twist_cliche
    story.twisted_synopsis = twisted_synopsis
    story.save()
    _analyze_and_save_character_state(story, twisted_synopsis, context="Twisted Synopsis")

    # [Neo4j & Django] 5. 비틀기 이후 노드 생성
    new_branch_nodes = _create_nodes_from_synopsis(story, twisted_synopsis, start_node_index=twist_node_index+1, is_twist_branch=True)

    # [Neo4j & Django] 6. 분기 처리 (선택지 관계 전송 포함)
    if twist_node_index + 1 < len(original_nodes) and new_branch_nodes:
        original_next = original_nodes[twist_node_index + 1]
        new_next = new_branch_nodes[0]
        NodeChoice.objects.filter(current_node=twist_node).delete()
        _create_twist_branch_choices(twist_node, original_next, new_next)

    # [Neo4j & Django] 7. 새 브랜치 연결
    _connect_linear_nodes(new_branch_nodes)

    return story.id

# ==========================================
# [내부 로직 함수들]
# ==========================================

# ... (_match_cliche, _generate_synopsis, _analyze_and_save_character_state, _get_latest_character_states 기존 유지) ...
def _match_cliche(setting):
    all_cliches = Cliche.objects.select_related('genre').all()
    if not all_cliches.exists(): return None
    cliche_info = "\n".join([f"ID {c.id}: [{c.genre.name}] {c.title} - {c.summary}" for c in all_cliches])
    res = call_llm("스토리 분석가입니다. ID JSON 반환.", f"설정: {setting}\n목록:\n{cliche_info}\n출력: {{'cliche_id': 숫자}}", json_format=True)
    try: return Cliche.objects.get(id=res['cliche_id'])
    except: return all_cliches.first()

def _generate_synopsis(story, cliche):
    return call_llm("소설가입니다.", f"설정: {story.user_world_setting}\n클리셰: {cliche.title}\n가이드: {cliche.structure_guide}\n줄거리 작성.")

def _analyze_and_save_character_state(story, text, context):
    res = call_llm("인물 내면 상태 분석 JSON.", f"텍스트: {text}", json_format=True)
    for name, state in res.items():
        CharacterState.objects.create(story=story, character_name=name, state_data=state, update_context=context)

def _get_latest_character_states(story):
    states = CharacterState.objects.filter(story=story).order_by('created_at')
    latest_map = {}
    for s in states: latest_map[s.character_name] = s.state_data
    return json.dumps(latest_map, ensure_ascii=False)

def _create_nodes_from_synopsis(story, synopsis, start_node_index=0, is_twist_branch=False):
    phases = ["발단", "전개", "절정", "결말"]
    nodes = []
    char_states_str = _get_latest_character_states(story)

    sys_prompt = "상세 스토리 씬 8개를 JSON 리스트로 생성하세요."
    context_note = "주의: Twist Branch입니다." if is_twist_branch else ""
    user_prompt = f"시놉시스: {synopsis}\n상태: {char_states_str}\n{context_note}\n형식: {{'scenes': [...]}}"
    
    res = call_llm(sys_prompt, user_prompt, json_format=True)

    # [디버깅용 출력 추가]
    print(f"🔍 [Debug] LLM Response for Nodes: {res}") 

    scenes = res.get('scenes', [])
    
    if not scenes:
        print("⚠️ [Warning] 'scenes' key not found in response or list is empty.")

    target_scenes = scenes[start_node_index:]
    
    for i, content in enumerate(target_scenes):
        current_idx = start_node_index + i
        if current_idx >= 8: break 
        
        phase_name = phases[min(current_idx // 2, 3)]
        
        # 1. Django DB 저장
        node = StoryNode.objects.create(story=story, chapter_phase=phase_name, content=content)
        nodes.append(node)
        
        # 2. Neo4j 전송 (노드 생성만 담당, UniverseID 제거)
        try:
            neo4j_data = StoryNodeData(
                node_id=node.id,
                phase=phase_name,
                content=content,
                character_state=char_states_str
            )
            sync_node_to_neo4j(neo4j_data)
        except Exception as e:
            print(f"Neo4j Node Sync Error: {e}")

    return nodes

def _connect_linear_nodes(nodes):
    for i in range(len(nodes) - 1):
        curr = nodes[i]
        next_n = nodes[i+1]
        next_n.prev_node = curr
        next_n.save()
        
        sys_prompt = "다음 노드 연결 선택지 2개 생성. result_text는 주인공 주어 완결 문장."
        user_prompt = f"현재: {curr.content[-500:]}\n다음: {next_n.content[:500]}\n형식: JSON"
        res = call_llm(sys_prompt, user_prompt, json_format=True)
        
        for item in res.get('choices', []):
            NodeChoice.objects.create(
                current_node=curr, choice_text=item['text'], result_text=item['result'], 
                next_node=next_n, is_twist_path=False
            )
            try:
                sync_choice_to_neo4j(curr.id, next_n.id, item['text'], item['result'], is_twist=False)
            except: pass

def _find_twist_point_index(nodes):
    if len(nodes) < 4: return 1
    summaries = [f"Idx {i}: {n.content[:50]}" for i, n in enumerate(nodes[:-2])]
    res = call_llm("비틀기 지점 인덱스 선택 JSON", "\n".join(summaries), json_format=True)
    idx = res.get('index', 2)
    if idx >= len(nodes)-2: idx = len(nodes)-3
    if idx < 1: idx = 1
    nodes[idx].is_twist_point = True
    nodes[idx].save()
    return idx

def _generate_twisted_synopsis_data(story, accumulated, phase):
    # (복선 회수 프롬프트 유지)
    all_cliches = Cliche.objects.exclude(id=story.main_cliche.id).all()
    if not all_cliches: return None, ""
    cliche_info = "\n".join([f"ID {c.id}: {c.title}" for c in all_cliches])
    
    rec_res = call_llm("반전의 대가. 미해결 떡밥 재해석할 클리셰 추천.", f"스토리: {accumulated[-1000:]}\n후보: {cliche_info}", json_format=True)
    try: new_cliche = Cliche.objects.get(id=rec_res['cliche_id'])
    except: new_cliche = all_cliches.first()

    twisted_synopsis = call_llm("치밀한 복선 회수. 시놉시스 재구성.", f"스토리: {accumulated}\n새 클리셰: {new_cliche.title}")
    return new_cliche, twisted_synopsis

def _create_twist_branch_choices(node, old_next, new_next):
    sys_prompt = "장르 전환 분기점. 선택지 1,2(Original), 3,4(Twist) 생성. result_text 완결 문장."
    user_prompt = f"현재: {node.content[-500:]}\n기존 다음: {old_next.content[:500]}\n새 다음: {new_next.content[:500]}\n형식: JSON"
    res = call_llm(sys_prompt, user_prompt, json_format=True)
    
    for item in res.get('original_choices', []):
        NodeChoice.objects.create(
            current_node=node, choice_text=item['text'], result_text=item['result'], 
            next_node=old_next, is_twist_path=False
        )
        try: sync_choice_to_neo4j(node.id, old_next.id, item['text'], item['result'], is_twist=False)
        except: pass
        
    for item in res.get('twist_choices', []):
        NodeChoice.objects.create(
            current_node=node, choice_text=item['text'], result_text=item['result'], 
            next_node=new_next, is_twist_path=True
        )
        try: sync_choice_to_neo4j(node.id, new_next.id, item['text'], item['result'], is_twist=True)
        except: pass