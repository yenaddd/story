from neo4j import GraphDatabase
import json
from dataclasses import dataclass, asdict

# Neo4j 연결 설정
URI = "neo4j+ssc://32adcd36.databases.neo4j.io"
AUTH = ("neo4j", "sKyJKxvWChIunry20Sk2cA-Wi-d-0oZH75LWcZz6zUg")

_DRIVER = None

def get_driver():
    global _DRIVER
    if _DRIVER is None:
        _DRIVER = GraphDatabase.driver(URI, auth=AUTH)
    return _DRIVER

def close_driver():
    global _DRIVER
    if _DRIVER is not None:
        _DRIVER.close()
        _DRIVER = None

def run_cypher(query: str, params: dict = None):
    if params is None: params = {}
    try:
        driver = get_driver() 
        with driver.session(database="neo4j") as session:
            session.run(query, **params)
    except Exception as e:
        print(f"Cypher 쿼리 실행 중 오류 발생: {e}")

# -----------------------------------------------------------
# [1] 세계관(Universe) 노드
# -----------------------------------------------------------
def create_universe_node_neo4j(universe_id: str, world_setting: str, protagonist_name: str):
    """
    Universe 노드 생성
    """
    query = """
    MERGE (u:Universe {universe_id: $universe_id})
    ON CREATE SET 
        u.setting = $world_setting,
        u.protagonist_name = $protagonist_name,
        u.experimental = false,
        u.image_url = "",  // 이미지 링크 빈 필드
        u.created_at = timestamp()
    ON MATCH SET 
        u.setting = $world_setting
    """
    run_cypher(query, {
        "universe_id": universe_id, 
        "world_setting": world_setting,
        "protagonist_name": protagonist_name
    })
    print(f"  [Neo4j] Universe Node Created: {universe_id}")

def update_universe_details_neo4j(universe_id: str, synopsis: str, twisted_synopsis: str,
                                  title: str, description: str, detail_description: str, 
                                  play_time: str, characters_info: str):
    """
    Universe 노드 상세 정보 업데이트 (요청 사항 반영)
    - 기존/분기 스토리, 주요 인물 정보, 예상 플레이 시간 등
    """
    query = """
    MATCH (u:Universe {universe_id: $universe_id})
    SET 
        u.synopsis = $synopsis,
        u.twisted_synopsis = $twisted_synopsis,
        u.title = $title,
        u.description = $description,           // 간단한 소개
        u.detail_description = $detail_description, // 상세 소개
        u.play_time = $play_time,
        u.characters_info = $characters_info    // 주요 인물 정보 (JSON 문자열)
    """
    run_cypher(query, {
        "universe_id": universe_id, 
        "synopsis": synopsis,
        "twisted_synopsis": twisted_synopsis,
        "title": title,
        "description": description,
        "detail_description": detail_description,
        "play_time": play_time,
        "characters_info": characters_info
    })
    print(f"  [Neo4j] Universe Details Updated")

# -----------------------------------------------------------
# [2] 스토리(Scene) 노드
# -----------------------------------------------------------
@dataclass
class StoryNodeData:
    node_id: str          
    phase: str            # 기승전결
    title: str            # 장면 제목
    setting: str          # 어떤 장면인지 설정 텍스트
    description: str      # 장면 줄거리
    purpose: str          # 장면의 목적
    
    # 등장인물 관련
    characters_list: list # 등장인물 리스트
    character_states: str # 등장인물 상태 (감정, 생각, 관계 등) JSON string
    
    depth: int = 0        # 장면 깊이

    def to_dict(self):
        return asdict(self)

def sync_node_to_neo4j(data: StoryNodeData):
    props = data.to_dict()
    query = """
    MERGE (n:Scene {node_id: $props.node_id})
    ON CREATE SET 
        n.title = $props.title,
        n.phase = $props.phase,
        n.setting = $props.setting,
        n.description = $props.description,
        n.purpose = $props.purpose,
        n.characters = $props.characters_list,
        n.character_states = $props.character_states,
        n.depth = $props.depth,
        n.created_at = timestamp()
    ON MATCH SET 
        n.title = $props.title,
        n.description = $props.description,
        n.character_states = $props.character_states,
    """
    run_cypher(query, {"props": props})

# -----------------------------------------------------------
# [3] 연결 관계 (조건 행동)
# -----------------------------------------------------------

def link_universe_to_first_scene(universe_id: str, first_node_id: str):
    query = """
    MATCH (u:Universe {universe_id: $universe_id})
    MATCH (n:Scene {node_id: $node_id})
    MERGE (u)-[r:HAS_START]->(n)
    """
    run_cypher(query, {"universe_id": universe_id, "node_id": first_node_id})

def sync_action_to_neo4j(curr_id: str, next_id: str, action_text: str, result_text: str, is_twist=False, character_changes: str = "{}"):
    """
    Scene 간의 연결은 '필수 행동(REQUIRED_ACTION)'으로 정의
    """
    rel_type = "TWIST_ACTION" if is_twist else "REQUIRED_ACTION"
    query = f"""
    MATCH (curr:Scene {{node_id: $curr_id}})
    MATCH (next:Scene {{node_id: $next_id}})
    MERGE (curr)-[r:{rel_type} {{action_text: $action_text}}]->(next)
    SET r.result_text = $result_text
        r.character_changes = $character_changes
    """
    run_cypher(query, {
        "curr_id": curr_id, 
        "next_id": next_id, 
        "action_text": action_text, 
        "result_text": result_text,
        "character_changes": character_changes # 전 노드 대비 변화 JSON string

    })