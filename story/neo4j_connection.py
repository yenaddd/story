from neo4j import GraphDatabase
import json
from dataclasses import dataclass, asdict

# Neo4j 연결 설정 (기존 유지)
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
# [1] 노드 데이터 전송 (내용, 단계, ID, 인물 내면)
# -----------------------------------------------------------
@dataclass
class StoryNodeData:
    node_id: int          # Django ID와 일치시켜 식별
    phase: str            # 발단, 전개, 절정, 결말
    content: str          # 노드 줄거리 내용
    character_state: str  # 인물 내면 상태 (JSON String)

    def to_dict(self):
        return asdict(self)

def sync_node_to_neo4j(data: StoryNodeData):
    """
    Django의 StoryNode가 생성될 때 호출.
    Neo4j에 Scene 노드를 생성하거나 업데이트합니다.
    """
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
    # print(f"  [Neo4j] Node {props['node_id']} synced.")

# -----------------------------------------------------------
# [2] 선택지 및 연결 데이터 전송 (선택지 내용, 결과 행동, 연결)
# -----------------------------------------------------------
def sync_choice_to_neo4j(current_node_id, next_node_id, choice_text, result_text, is_twist=False):
    """
    Django의 NodeChoice가 생성될 때 호출.
    두 노드 사이를 잇는 CHOICE 관계(Relationship)를 생성합니다.
    """
    params = {
        "curr_id": current_node_id,
        "next_id": next_node_id,
        "choice_text": choice_text,
        "result_text": result_text,
        "type": "TWIST_CHOICE" if is_twist else "CHOICE"
    }

    # 관계(Relationship)에 속성(선택지 텍스트, 결과 행동)을 저장합니다.
    query = f"""
    MATCH (curr:Scene {{node_id: $curr_id}})
    MATCH (next:Scene {{node_id: $next_id}})
    MERGE (curr)-[r:{params['type']} {{choice_text: $choice_text}}]->(next)
    SET r.result_text = $result_text
    """
    run_cypher(query, params)
    # print(f"  [Neo4j] Link: {current_node_id} -> {next_node_id} ({choice_text})")