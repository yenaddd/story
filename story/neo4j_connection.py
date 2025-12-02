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
# [1] 세계관(Universe) 노드 생성 & 업데이트
# -----------------------------------------------------------
def create_universe_node_neo4j(universe_id: str, world_setting: str):
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

def update_universe_node_neo4j(universe_id: str, protagonist_name: str, protagonist_desc: str, synopsis: str, 
                               world_title: str, description: str, detail_description: str, play_time: str):
    """
    초기 시놉시스 및 세계관 상세 정보(제목, 소개, 플레이타임) 업데이트
    """
    query = """
    MATCH (u:Universe {universe_id: $universe_id})
    SET 
        u.protagonist_name = $protagonist_name,
        u.protagonist_desc = $protagonist_desc,
        u.synopsis = $synopsis,
        u.title = $world_title,
        u.description = $description,
        u.detail_description = $detail_description,
        u.play_time = $play_time
    """
    run_cypher(query, {
        "universe_id": universe_id, 
        "protagonist_name": protagonist_name,
        "protagonist_desc": protagonist_desc,
        "synopsis": synopsis,
        "world_title": world_title,
        "description": description,
        "detail_description": detail_description,
        "play_time": play_time
    })
    print(f"  [Neo4j] Universe Node Updated with Details (Title: {world_title})")
    
def update_universe_twist_neo4j(universe_id: str, twisted_synopsis: str):
    """[New] 변주 시놉시스 별도 필드 저장"""
    query = """
    MATCH (u:Universe {universe_id: $universe_id})
    SET 
        u.twisted_synopsis = $twisted_synopsis
    """
    run_cypher(query, {
        "universe_id": universe_id, 
        "twisted_synopsis": twisted_synopsis
    })
    print(f"  [Neo4j] Universe Node Updated with Twisted Synopsis")

# -----------------------------------------------------------
# [2] 스토리(Scene) 노드 생성
# -----------------------------------------------------------
@dataclass
class StoryNodeData:
    node_id: str          
    phase: str            
    title: str
    setting: str
    characters: list      
    description: str
    purpose: str
    character_state: str  

    def to_dict(self):
        return asdict(self)

def sync_node_to_neo4j(data: StoryNodeData):
    props = data.to_dict()
    query = """
    MERGE (n:Scene {node_id: $props.node_id})
    ON CREATE SET 
        n.title = $props.title,
        n.setting = $props.setting,
        n.characters = $props.characters,
        n.description = $props.description,
        n.purpose = $props.purpose,
        n.phase = $props.phase,
        n.character_state = $props.character_state,
        n.created_at = timestamp()
    ON MATCH SET 
        n.title = $props.title,
        n.description = $props.description,
        n.character_state = $props.character_state
    """
    run_cypher(query, {"props": props})

# -----------------------------------------------------------
# [3] 연결 관계 생성
# -----------------------------------------------------------

def link_universe_to_first_scene(universe_id: str, first_node_id: str):
    query = """
    MATCH (u:Universe {universe_id: $universe_id})
    MATCH (n:Scene {node_id: $node_id})
    MERGE (u)-[r:HAS_START]->(n)
    """
    run_cypher(query, {"universe_id": universe_id, "node_id": first_node_id})
    print(f"  [Neo4j] Linked Universe -> First Scene ({first_node_id})")

def sync_choice_to_neo4j(curr_id: str, next_id: str, choice_text, result_text, is_twist=False):
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