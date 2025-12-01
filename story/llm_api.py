import os
import json
import uuid 
from openai import OpenAI
from typing import List, Dict, Any, Optional
from django.db import transaction, close_old_connections 
from .models import StoryNode, NodeChoice
import concurrent.futures 

# [수정] update_scene_choices 함수 import 추가
from .neo4j_connection import create_scene_node, SceneNode, create_universe_node, UniverseNode, update_scene_choices

# --- GLOBAL CONFIGURATION ---
GLOBAL_STORY_CONFIG = {
    "WORLD_SETTING": "",
    "ARC_INFO": {},
    "BRANCH_CONFIG": [], 
    "MAX_DEPTH": 0,
    "STORY_TEXT_LENGTH": 7000,
    "MAX_CONCURRENT_WORKERS": 3,
    "OVERALL_STORY_PLOT": "",
    "UNIVERSE_ID": "1" 
}

GPT_MODEL = "gpt-4o-mini"
MAX_RETRIES = 3
CRITIQUE_SCORE_THRESHOLD = 80 

def get_openai_client():
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        raise ValueError("OpenAI API Key is missing in environment variables.")
    return OpenAI()

def set_global_config(world_setting: str, arc_type: str, branches: List[int], max_workers: int):
    GLOBAL_STORY_CONFIG["WORLD_SETTING"] = world_setting
    GLOBAL_STORY_CONFIG["ARC_INFO"] = {"arc": arc_type}
    GLOBAL_STORY_CONFIG["BRANCH_CONFIG"] = branches
    GLOBAL_STORY_CONFIG["MAX_DEPTH"] = len(branches)
    GLOBAL_STORY_CONFIG["MAX_CONCURRENT_WORKERS"] = max_workers
    
    new_universe_id = str(uuid.uuid4())
    GLOBAL_STORY_CONFIG["UNIVERSE_ID"] = new_universe_id
    
    print(f"Global Story Config Set: Depth={GLOBAL_STORY_CONFIG['MAX_DEPTH']}, UniverseID={new_universe_id}")

def _get_freytag_stage_for_depth(current_depth: int, max_depth: int) -> str:
    STAGE_NAMES = {
        "Exposition": "발단 (Exposition)",
        "Inciting Incident": "발단부의 갈등 (Inciting Incident)",
        "Rising Action": "상승 (Rising Action)",
        "Climax": "절정 (Climax)",
        "Falling Action": "하강 (Falling Action)",
        "Resolution": "결말 (Resolution)"
    }
    depth_stage_map_eng = [] 

    if max_depth <= 0: return "Unknown"
    elif max_depth == 1: depth_stage_map_eng = ["Exposition"]
    elif max_depth == 2: depth_stage_map_eng = ["Climax", "Resolution"]
    elif max_depth == 3: depth_stage_map_eng = ["Rising Action", "Climax", "Resolution"]
    elif max_depth == 4: depth_stage_map_eng = ["Exposition", "Rising Action", "Climax", "Resolution"]
    elif max_depth == 5: depth_stage_map_eng = ["Exposition", "Rising Action", "Climax", "Climax", "Resolution"]
    elif max_depth >= 6:
        stages_in_order = ["Exposition", "Inciting Incident", "Rising Action", "Climax", "Falling Action", "Resolution"]
        scene_counts = {stage: 0 for stage in stages_in_order}
        base_scenes = max_depth // 6
        remainder_scenes = max_depth % 6
        remainder_priority = ["Climax", "Rising Action", "Falling Action", "Inciting Incident", "Exposition"]

        for stage in stages_in_order: scene_counts[stage] = base_scenes
        for i in range(remainder_scenes):
            if i < len(remainder_priority): scene_counts[remainder_priority[i]] += 1
        for stage in stages_in_order:
            depth_stage_map_eng.extend([stage] * scene_counts[stage])
    
    if not depth_stage_map_eng: return "Unknown"
    stage_key = depth_stage_map_eng[current_depth] if 0 <= current_depth < len(depth_stage_map_eng) else depth_stage_map_eng[-1]
    return STAGE_NAMES.get(stage_key, "Unknown Stage")

def call_llm_flow_definer(world_setting: str, arc_type: str) -> str:
    print("--- 전체흐름정의자(Flow Definer) AI 호출 ---")
    try:
        client = get_openai_client()
        system_prompt = (
            "당신은 천재적인 소설가이자 스토리 '전체흐름정의자'입니다. "
            "사용자의 [세계관/인물 설정]과 [캐릭터 아크]를 바탕으로, 기승전결이 명확하고 "
            "일관성 있는 '전체 스토리의 핵심 줄거리'를 30000자 내외로 생성해야 합니다. "
            "생성되는 줄거리는 많은 문학작품 검색과 문학작품의 작품성 관련 연구결과 검색을 통해, 독자가 흥미를 느낄만한 요소가 많고 이야기의 개연성과 감동이 있어야합니다."
        )
        user_prompt = f"""
        [세계관/인물 설정]: {world_setting}
        [적용된 캐릭터 아크 이론]: {arc_type}
        위 설정을 바탕으로, 주인공의 여정이 담긴 30000자 내외의 '전체 스토리 줄거리'를 생성하십시오.
        """
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Flow Definer LLM Error: {e}")
        return "전체 줄거리 생성 중 오류 발생"

def call_llm_initial_setup(world_setting: str) -> Dict[str, Any]:
    print("--- 초기 설정 분석가(Setup Analyzer) AI 호출 ---")
    try:
        client = get_openai_client()
        system_prompt = (
            "당신은 소설 설정 분석가입니다. 사용자가 입력한 [세계관/인물 설정] 텍스트를 분석하여 "
            "JSON 형식으로 추출해야 합니다.\n"
            "1. relationships: { '등장인물 이름': '관계 및 태도 설명(문자열)' } 형태의 딕셔너리. (중첩 객체 금지)\n"
            "2. state: 주인공의 심리 상태 (문자열)\n"
            "3. title: 소설 제목 (문자열)"
        )
        user_prompt = f"[세계관/인물 설정]: {world_setting}"
        response = client.chat.completions.create(
            model=GPT_MODEL, response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        )
        return json.loads(response.choices[0].message.content)
    except Exception:
        return {"relationships": {"시스템": "데이터 추출 실패"}, "state": "시작", "title": "생성된 스토리"}

def _get_story_history(node_id: int) -> str:
    if node_id <= 1: return "이야기 시작 지점"
    try:
        history_parts = []
        current_node = StoryNode.objects.get(id=node_id)
        while current_node is not None:
            history_parts.append(f"[선택]: {current_node.parent_choice_text}")
            history_parts.append(f"[장면 Depth {current_node.depth}]: {current_node.story_text}")
            if current_node.parent_node_id is None: break
            current_node = StoryNode.objects.get(id=current_node.parent_node_id)
        history_parts.reverse()
        return "\n-> ".join(history_parts)
    except Exception as e:
        return f"히스토리 생성 오류: {e}"

def call_llm_architect(depth: int, context_node_data: Optional[Dict[str, Any]], last_critique: str, overall_story_plot: str, story_history: str) -> Dict[str, Any]:
    config = GLOBAL_STORY_CONFIG
    max_depth = config["MAX_DEPTH"]
    
    try:
        client = get_openai_client()
        if context_node_data is None:
             context_node_data = {'relationships': {}, 'state': '초기 상태', 'choice_text': '이야기 시작'}
             
        current_relationships = json.dumps(context_node_data.get('relationships', {}), ensure_ascii=False)
        protagonist_state = context_node_data.get('state', '시작 상태')
        parent_choice_text = context_node_data.get('choice_text', '없음')
        stage = _get_freytag_stage_for_depth(depth, max_depth)
        
        system_prompt = "당신은 최고의 스토리 기획자입니다. JSON 형식으로 응답하십시오."
        user_prompt = f"""
        [세계관]: {config["WORLD_SETTING"]}
        [전체 줄거리]: {overall_story_plot}
        [스토리 히스토리]: {story_history}
        [현재 상태]: 행동='{parent_choice_text}', 심리='{protagonist_state}', 관계='{current_relationships}', Depth={depth}
        [피드백]: {last_critique}
        [임무]: Freytag 단계({stage})에 맞는 다음 장면 기획.
        
        응답 형식: 
        {{
            "freytag_stage": "...", 
            "plot_summary": "...", 
            "new_relationships": {{ "캐릭터이름": "관계변화설명(문자열)" }}, 
            "new_protagonist_state": "..."
        }}
        * 주의: relationships의 값은 반드시 객체가 아닌 '문자열'이어야 합니다.
        """
        response = client.chat.completions.create(
            model=GPT_MODEL, response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        )
        result = json.loads(response.choices[0].message.content)
        result['freytag_stage'] = stage
        return result
    except Exception as e:
        print(f"Architect Error: {e}")
        return {"freytag_stage": "Error", "plot_summary": "Error", "new_relationships": {}, "new_protagonist_state": ""}

def call_llm_critic(architect_result: Dict[str, Any], context: Dict[str, Any], overall_story_plot: str, story_history: str) -> Dict[str, Any]:
    try:
        client = get_openai_client()
        system_prompt = "당신은 스토리 기획안의 개연성을 평가하는 감독입니다. 점수(0-100)와 지침을 JSON으로 반환하십시오."
        user_prompt = f"""
        [전체 줄거리]: {overall_story_plot}
        [히스토리]: {story_history}
        [기획안]: {json.dumps(architect_result, ensure_ascii=False)}
        응답 형식: {{"score": 85, "critique": "..."}}
        """
        response = client.chat.completions.create(
            model=GPT_MODEL, response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        )
        data = json.loads(response.choices[0].message.content)
        return {"score": int(data.get('score', 0)), "critique": data.get('critique', '')}
    except Exception:
        return {"score": 0, "critique": "Error"}

def call_llm_choices(plot_summary_text: str, num_choices: int, overall_story_plot: str) -> List[str]:
    try:
        client = get_openai_client()
        system_prompt = f"당신은 분기점 설계자입니다. {num_choices}개의 상반된 선택지를 JSON 배열로 생성하십시오."
        user_prompt = f"[전체 줄거리]: {overall_story_plot}\n[현재 상황]: {plot_summary_text}\n응답 형식: {{\"choices\": [\"선택1\", \"선택2\"]}}"
        response = client.chat.completions.create(
            model=GPT_MODEL, response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        )
        return json.loads(response.choices[0].message.content).get('choices', [])
    except Exception:
        return [f"선택지 {i+1}" for i in range(num_choices)]


def _generate_single_node(parent_node_id: int, choice_text: str, parent_relationships=None, parent_state=None) -> StoryNode:
    config = GLOBAL_STORY_CONFIG
    overall_story_plot = config.get("OVERALL_STORY_PLOT", "")
    universe_id = config.get("UNIVERSE_ID", "1") 
    
    story_history = _get_story_history(parent_node_id)
    story_title = "새로운 스토리"

    if parent_node_id == 0:
        new_node_depth = 0
        setup_data = call_llm_initial_setup(config["WORLD_SETTING"])
        context_data = {
            'relationships': setup_data.get('relationships', {}),
            'state': setup_data.get('state', '시작'),
            'choice_text': choice_text
        }
        story_title = setup_data.get('title', '생성된 스토리')
    else:
        if parent_relationships is None:
            parent_node = StoryNode.objects.get(id=parent_node_id)
            parent_relationships = parent_node.current_relationships
            parent_state = parent_node.protagonist_state
        context_data = {'relationships': parent_relationships, 'state': parent_state, 'choice_text': choice_text}
        new_node_depth = StoryNode.objects.get(id=parent_node_id).depth + 1

    last_critique = "첫 시도"
    for attempt in range(1, MAX_RETRIES + 1):
        architect_result = call_llm_architect(new_node_depth, context_data, last_critique, overall_story_plot, story_history)
        if architect_result['plot_summary'] == "Error": continue

        critique_result = call_llm_critic(architect_result, context_data, overall_story_plot, story_history)
        score = critique_result['score']
        last_critique = critique_result['critique']
        print(f"Depth {new_node_depth} 시도 {attempt}: {score}점")

        if score >= CRITIQUE_SCORE_THRESHOLD or attempt == MAX_RETRIES:
            new_node = StoryNode.objects.create(
                depth=new_node_depth,
                freytag_stage=architect_result.get('freytag_stage'),
                parent_node_id=parent_node_id if parent_node_id != 0 else None,
                parent_choice_text=choice_text,
                current_relationships=architect_result.get('new_relationships'),
                protagonist_state=architect_result.get('new_protagonist_state'),
                story_text=architect_result.get('plot_summary'),
                critique_score=score
            )
            
            # Neo4j Sync
            try:
                # [수정 포인트] ID 충돌 방지를 위한 "고유 ID(Composite Key)" 생성
                # 예: "uuid-1234-5678_1" (우주ID_노드ID)
                unique_node_id = f"{universe_id}_{new_node.id}"
                
                # 부모 노드 ID도 같은 방식으로 변환 (루트가 아닐 경우)
                unique_parent_id = f"{universe_id}_{parent_node_id}" if parent_node_id else None
                neo_scene = SceneNode(
                    universe_id=universe_id, 
                    depth=new_node.depth, 
                    node_id=unique_node_id,
                    freytag_stage=new_node.freytag_stage,
                    parent_node_id=unique_parent_id,
                    parent_choice_text=choice_text,
                    story_text=new_node.story_text,
                    protagonist_state=new_node.protagonist_state,
                    current_relationships=new_node.current_relationships,
                    critique_score=new_node.critique_score
                )
                if new_node.depth == 0:
                    create_universe_node(UniverseNode(universe_id=universe_id, title=story_title))
                create_scene_node(neo_scene)
                print(f"  -> Neo4j Sync 완료: Node {new_node.id}")
            except Exception as e:
                print(f"  -> Neo4j Sync 실패: {e}")
            
            return new_node
    raise Exception("노드 생성 실패")

def generate_full_story_tree(parent_node_id: int, choice_text: str, parent_relationships=None, parent_state=None) -> StoryNode:
    close_old_connections()
    
    try:
        config = GLOBAL_STORY_CONFIG
        max_depth = config["MAX_DEPTH"]
        branch_config = config["BRANCH_CONFIG"]
        max_workers = config.get("MAX_CONCURRENT_WORKERS", 4)
        overall_story_plot = config.get("OVERALL_STORY_PLOT", "")
        
        # 현재 노드 생성
        current_node = _generate_single_node(parent_node_id, choice_text, parent_relationships, parent_state)
        new_node_depth = current_node.depth
        
        # [수정됨] 종료 조건 완화: Depth가 max_depth와 같아질 때까지 진행
        # 기존: new_node_depth >= (max_depth - 1)  <-- Depth 2에서 멈춤
        # 수정: new_node_depth >= max_depth        <-- Depth 3까지 감
        is_ending_node = new_node_depth >= max_depth

        if parent_node_id != 0:
            with transaction.atomic():
                try:
                    choice = NodeChoice.objects.get(parent_node_id=parent_node_id, choice_text=choice_text)
                    choice.next_node = current_node
                    choice.save()
                except Exception: pass

        if not is_ending_node:
            try: num_choices = branch_config[new_node_depth]
            except: num_choices = 2
            
            choice_texts = call_llm_choices(current_node.story_text, num_choices, overall_story_plot)
            
            # Neo4j에 선택지 업데이트
            try:
                universe_id = config.get("UNIVERSE_ID", "1")
                unique_node_id = f"{universe_id}_{current_node.id}"
                
                update_scene_choices(unique_node_id, choice_texts)
            except Exception as e:
                print(f"Neo4j Choice Update Failed: {e}")

            choices_to_process = []
            
            with transaction.atomic():
                for text in choice_texts:
                    NodeChoice.objects.create(parent_node=current_node, choice_text=text)
                    choices_to_process.append(text)

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for text in choices_to_process:
                    future = executor.submit(
                        generate_full_story_tree, 
                        parent_node_id=current_node.id, 
                        choice_text=text,
                        parent_relationships=current_node.current_relationships,
                        parent_state=current_node.protagonist_state
                    )
                    futures.append(future)
                concurrent.futures.wait(futures)
        
        return current_node
        
    finally:
        close_old_connections()