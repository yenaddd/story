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
# [1] 세계관(Universe) 노드 생성
# -----------------------------------------------------------
def create_universe_node_neo4j(universe_id: str, world_setting: str):
    """
    최상위 부모 노드인 Universe 노드를 생성합니다.
    """
    query = """
    MERGE (u:Universe {universe_id: $universe_id})
    ON CREATE SET 
        u.setting = $world_setting,
        u.created_at = timestamp()
    ON MATCH SET 
        u.setting = $world_setting
    """
    run_cypher(query, {"universe_id": universe_id, "world_setting": world_setting})
    print(f"  [Neo4j] Universe Node Created: {universe_id}")

# -----------------------------------------------------------
# [2] 스토리(Scene) 노드 생성
# -----------------------------------------------------------
@dataclass
class StoryNodeData:
    node_id: int          # Django ID
    phase: str            # 단계 (발단, 전개...)
    content: str          # 내용
    character_state: str  # 내면 상태 JSON

    def to_dict(self):
        return asdict(self)

def sync_node_to_neo4j(data: StoryNodeData):
    props = data.to_dict()
    query = """
    MERGE (n:Scene {node_id: $props.node_id})
    ON CREATE SET 
        n.content = $props.content,
        n.phase = $props.phase,
        n.character_state = $props.character_state,
        n.created_at = timestamp()
    ON MATCH SET 
        n.content = $props.content,
        n.phase = $props.phase,
        n.character_state = $props.character_state
    """
    run_cypher(query, {"props": props})

# -----------------------------------------------------------
# [3] 연결 관계 생성 (Universe -> Start, Scene -> Scene)
# -----------------------------------------------------------

def link_universe_to_first_scene(universe_id: str, first_node_id: int):
    """
    세계관(Universe)과 이야기의 시작점(첫 Scene)을 연결합니다.
    (Universe)-[:HAS_START]->(Scene)
    """
    query = """
    MATCH (u:Universe {universe_id: $universe_id})
    MATCH (n:Scene {node_id: $node_id})
    MERGE (u)-[r:HAS_START]->(n)
    """
    run_cypher(query, {"universe_id": universe_id, "node_id": first_node_id})
    print(f"  [Neo4j] Linked Universe -> First Scene ({first_node_id})")

def sync_choice_to_neo4j(curr_id, next_id, choice_text, result_text, is_twist=False):
    """
    선택지에 따른 Scene 간 연결 관계를 생성합니다.
    (Scene)-[:CHOICE]->(Scene)
    """
    rel_type = "TWIST_CHOICE" if is_twist else "CHOICE"
    query = f"""
    MATCH (curr:Scene {{node_id: $curr_id}})
    MATCH (next:Scene {{node_id: $next_id}})
    MERGE (curr)-[r:{rel_type} {{choice_text: $choice_text}}]->(next)
    SET r.result_text = $result_text
    """
    run_cypher(query, {
        "curr_id": curr_id, 
        "next_id": next_id, 
        "choice_text": choice_text, 
        "result_text": result_text
    })