import os
import json
import time
from openai import OpenAI
from django.conf import settings
from .models import Genre, Cliche, Story, CharacterState, StoryNode, NodeChoice
from .neo4j_connection import sync_node_to_neo4j, sync_choice_to_neo4j, StoryNodeData

# API 설정
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
BASE_URL = "https://api.fireworks.ai/inference/v1"
# 모델명은 사용 가능한 최신 모델로 설정 (예: llama-v3-70b-instruct 등)
MODEL_NAME = "accounts/fireworks/models/llama-v3-70b-instruct" 
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=BASE_URL)

def call_llm(system_prompt, user_prompt, json_format=False, max_retries=3):
    """LLM 호출 래퍼 (재시도 로직 포함)"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    response_format = {"type": "json_object"} if json_format else None
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                response_format=response_format,
                temperature=0.7,
                max_tokens=4000 
            )
            content = response.choices[0].message.content
            
            if json_format:
                # JSON 파싱 보정
                cleaned_content = content
                if "```json" in cleaned_content:
                    cleaned_content = cleaned_content.replace("```json", "").replace("```", "")
                elif "```" in cleaned_content:
                    cleaned_content = cleaned_content.replace("```", "")
                
                return json.loads(cleaned_content)
            
            return content

        except json.JSONDecodeError as e:
            print(f"JSON Parsing Error (Attempt {attempt+1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                print("Max retries reached. Returning empty dict.")
                return {}
            time.sleep(1) # 잠시 대기 후 재시도
            
        except Exception as e:
            print(f"LLM Critical Error: {e}")
            return {} if json_format else ""

    return {} if json_format else ""

# ==========================================
# [메인 파이프라인] 14단계 프로세스
# ==========================================

def create_story_pipeline(user_world_setting):
    # 1. (입력됨) user_world_setting
    
    # 2. 클리셰 매칭
    matched_cliche = _match_cliche(user_world_setting)
    
    story = Story.objects.create(
        user_world_setting=user_world_setting,
        main_cliche=matched_cliche
    )

    # 3. 초기 시놉시스 생성 (발단-전개-절정-결말)
    synopsis = _generate_synopsis(story, matched_cliche)
    story.synopsis = synopsis
    story.save()

    # 4. 인물 내면 변화 추출 및 DB 저장
    _analyze_and_save_character_state(story, synopsis, context="Initial Synopsis")

    # 5 & 6. 챕터별 줄거리 생성 및 노드화 (선형 구조, 총 8노드)
    # 초기 생성은 처음(0)부터 시작
    original_nodes = _create_nodes_from_synopsis(story, synopsis, start_node_index=0)

    # 7. 선형 노드 간 연결 (Illusion of Choice)
    _connect_linear_nodes(original_nodes)

    # 8. 클리셰 비틀기(Twist) 지점 찾기
    twist_node_index = _find_twist_point_index(original_nodes)
    twist_node = original_nodes[twist_node_index]
    
    story.twist_point_node_id = twist_node.id
    story.save()

    # 9. 비틀기 이후 새로운 시놉시스 생성 & 새 클리셰 매칭
    accumulated_story = "\n".join([n.content for n in original_nodes[:twist_node_index+1]])
    current_phase = twist_node.chapter_phase
    
    twist_cliche, twisted_synopsis = _generate_twisted_synopsis_data(story, accumulated_story, current_phase)
    
    story.twist_cliche = twist_cliche
    story.twisted_synopsis = twisted_synopsis
    story.save()

    # 10. 인물 내면 변화 DB 업데이트 (새 시놉시스 반영)
    _analyze_and_save_character_state(story, twisted_synopsis, context="Twisted Synopsis")

    # 11. 비틀기 이후 노드 생성 + [Neo4j 전송]
    new_branch_nodes = _create_nodes_from_synopsis(story, twisted_synopsis, start_node_index=twist_node_index+1, is_twist_branch=True)

    # 12. 분기 처리 + [Neo4j 전송]
    if twist_node_index + 1 < len(original_nodes) and new_branch_nodes:
        original_next = original_nodes[twist_node_index + 1]
        new_next = new_branch_nodes[0]
        NodeChoice.objects.filter(current_node=twist_node).delete()
        _create_twist_branch_choices(twist_node, original_next, new_next)

    # 13. 새 브랜치 연결 + [Neo4j 전송]
    _connect_linear_nodes(new_branch_nodes)

    return story.id

# ==========================================
# [내부 로직 함수들]
# ==========================================

def _match_cliche(setting):
    """2. 사용자 설정에 가장 적합한 클리셰 매칭"""
    all_cliches = Cliche.objects.select_related('genre').all()
    if not all_cliches.exists():
        return None
        
    cliche_info = "\n".join([f"ID {c.id}: [{c.genre.name}] {c.title} - {c.summary}" for c in all_cliches])
    
    sys_prompt = "당신은 스토리 분석가입니다. 사용자 설정에 가장 적합한 클리셰 ID를 JSON으로 반환하세요."
    user_prompt = f"사용자 설정: {setting}\n\n보유 클리셰 목록:\n{cliche_info}\n\n출력형식: {{'cliche_id': 숫자}}"
    
    res = call_llm(sys_prompt, user_prompt, json_format=True)
    try:
        return Cliche.objects.get(id=res['cliche_id'])
    except:
        return all_cliches.first()

def _generate_synopsis(story, cliche):
    """3. 초기 시놉시스 생성"""
    role = "소설가"
    task = "전체 시놉시스를 작성하세요. 발단, 전개, 절정, 결말이 명확해야 합니다."
    
    content = (
        f"사용자 설정: {story.user_world_setting}\n"
        f"매칭된 클리셰: {cliche.title}\n"
        f"클리셰 가이드: {cliche.structure_guide}\n"
        f"참고 작품 감정선: {cliche.example_work_summary}\n\n"
        "조건 1: 사건의 원인과 해결 방식은 '사용자 설정'을 따르세요.\n"
        "조건 2: 인물의 감정선과 갈등 단계는 '참고 작품'과 '클리셰'를 벤치마킹하세요.\n"
        "조건 3: 총 2000자 내외의 줄거리(Plot) 형태로 작성하세요."
    )
    return call_llm(f"당신은 {role}입니다. {task}", content)

def _analyze_and_save_character_state(story, text_content, context):
    """4 & 10. 인물 내면 변화 추출 및 DB 저장"""
    sys_prompt = (
        "텍스트를 정밀 분석하여 각 등장인물의 '내면 상태'를 JSON으로 추출하세요. "
        "사건, 행동, 결정이 인물에게 준 감정적, 정서적, 육체적, 사상적, 신뢰적 변화를 기록해야 합니다."
    )
    user_prompt = f"분석할 텍스트:\n{text_content}\n\n출력 형식: {{ '인물이름': {{ 'emotion': '...', 'physical': '...', 'trust': '...' }} }}"
    
    res = call_llm(sys_prompt, user_prompt, json_format=True)
    
    for name, state in res.items():
        CharacterState.objects.create(
            story=story,
            character_name=name,
            state_data=state,
            update_context=context
        )

def _get_latest_character_states(story):
    """DB에서 최신 인물 상태 가져오기"""
    states = CharacterState.objects.filter(story=story).order_by('created_at')
    latest_map = {}
    for s in states:
        latest_map[s.character_name] = s.state_data
    return json.dumps(latest_map, ensure_ascii=False)

def _create_nodes_from_synopsis(story, synopsis, start_node_index=0, is_twist_branch=False):
    phases = ["발단", "전개", "절정", "결말"]
    nodes = []
    char_states_str = _get_latest_character_states(story)

    sys_prompt = "상세 스토리 씬 8개를 JSON 리스트로 생성하세요."
    context_note = "주의: Twist Branch입니다." if is_twist_branch else ""
    user_prompt = f"시놉시스: {synopsis}\n상태: {char_states_str}\n{context_note}\n형식: {{'scenes': [...]}}"
    
    res = call_llm(sys_prompt, user_prompt, json_format=True)
    scenes = res.get('scenes', [])
    target_scenes = scenes[start_node_index:]
    
    for i, content in enumerate(target_scenes):
        current_idx = start_node_index + i
        if current_idx >= 8: break 
        
        phase_name = phases[min(current_idx // 2, 3)]
        
        # 1. Django DB 저장
        node = StoryNode.objects.create(story=story, chapter_phase=phase_name, content=content)
        nodes.append(node)
        
        # 2. Neo4j 전송 (데이터 클래스 활용)
        try:
            neo4j_data = StoryNodeData(
                node_id=node.id,  # Django ID 사용
                phase=phase_name,
                content=content,
                character_state=char_states_str
            )
            sync_node_to_neo4j(neo4j_data)
        except Exception as e:
            print(f"Neo4j Sync Error: {e}")

    return nodes

def _connect_linear_nodes(nodes):
    for i in range(len(nodes) - 1):
        curr = nodes[i]
        next_n = nodes[i+1]
        next_n.prev_node = curr
        next_n.save()
        
        sys_prompt = "다음 노드로 가는 선택지 2개 생성. result_text는 주인공 주어 완결 문장."
        user_prompt = f"현재: {curr.content[-500:]}\n다음: {next_n.content[:500]}\n형식: JSON"
        res = call_llm(sys_prompt, user_prompt, json_format=True)
        
        for item in res.get('choices', []):
            # 1. Django 저장
            NodeChoice.objects.create(
                current_node=curr, choice_text=item['text'], result_text=item['result'], 
                next_node=next_n, is_twist_path=False
            )
            # 2. Neo4j 전송 (관계 생성)
            sync_choice_to_neo4j(curr.id, next_n.id, item['text'], item['result'], is_twist=False)

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
    # (기존 복선 회수 프롬프트 유지)
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
        sync_choice_to_neo4j(node.id, old_next.id, item['text'], item['result'], is_twist=False)
        
    for item in res.get('twist_choices', []):
        NodeChoice.objects.create(
            current_node=node, choice_text=item['text'], result_text=item['result'], 
            next_node=new_next, is_twist_path=True
        )
        sync_choice_to_neo4j(node.id, new_next.id, item['text'], item['result'], is_twist=True)