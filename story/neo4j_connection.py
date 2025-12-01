from neo4j import GraphDatabase
import os
import uuid
import json
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List

@dataclass
class SceneNode:
    universe_id: str
    depth: int
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    freytag_stage: str = ""
    parent_node_id: Optional[str] = None
    parent_choice_text: str = ""
    # [명세서 반영] 선택지 텍스트들을 담을 리스트 필드
    choice_text: List[str] = field(default_factory=list)
    current_relationships: Dict[str, Any] = field(default_factory=dict)
    protagonist_state: str = ""
    story_text: str = ""
    critique_score: Optional[int] = None
    
    def __str__(self):
        return f"Node {self.node_id} (Depth: {self.depth}, Stage: {self.freytag_stage})"

@dataclass
class UniverseNode:
    universe_id: str
    title: str=""
    description: str = ""

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
    records = []
    try:
        driver = get_driver() 
        with driver.session(database="neo4j") as session:
            result = session.run(query, **params)
            for record in result:
                records.append(record.data())
    except Exception as e:
        print(f"Cypher 쿼리 실행 중 오류 발생: {e}")
    return records

def create_scene_node(scene_node: SceneNode) -> dict:
    """
    Scene 노드를 생성합니다.
    (이 시점에는 아직 다음 선택지가 생성되기 전이므로 choice_text 리스트는 비어있을 수 있습니다.)
    """
    node_dict = asdict(scene_node)
    
    # 관계 설정용 필드는 노드 속성에서 제거 (깔끔하게 저장하기 위해)
    node_dict.pop('parent_choice_text', None)
    node_dict.pop('parent_node_id', None) 
    
    # 딕셔너리(관계 등)는 JSON 문자열로 변환
    if 'current_relationships' in node_dict and isinstance(node_dict['current_relationships'], dict):
        node_dict['current_relationships'] = json.dumps(node_dict['current_relationships'], ensure_ascii=False)
    
    # 1. 노드 생성 (MERGE 사용)
    query_create_node = """
    MERGE (n:Scene {node_id: $props.node_id})
    ON CREATE SET n = $props
    ON MATCH SET n += $props
    RETURN n
    """
    params_create = {"props": node_dict}
    run_cypher(query_create_node, params_create)
    
    # 2. 부모 관계 연결
    if scene_node.depth == 0:
        # 루트 노드면 Universe와 연결
        query_create_start = """
        MATCH (u:Universe {universe_id: $universe_id})
        MATCH (s:Scene {node_id: $child_node_id})
        MERGE (u)-[r:HAS_START]->(s)
        """
        params_start = {
            "universe_id": scene_node.universe_id,
            "child_node_id": scene_node.node_id
        }
        run_cypher(query_create_start, params_start)

    elif scene_node.parent_node_id: 
        # 일반 노드면 부모 Scene과 연결
        query_create_choice = """
        MATCH (parent:Scene {node_id: $parent_node_id})
        MATCH (child:Scene {node_id: $child_node_id})
        MERGE (parent)-[r:CHOICE]->(child)
        SET r.choice_text = $choice_text 
        """
        # 참고: 명세서에는 화살표 속성에 대한 언급이 없지만, 
        # 그래프 시각화에서 '어떤 선이 어떤 선택인지' 구별하기 위해 화살표에도 텍스트를 남겨두는 것이 안전합니다.
        params_choice = {
            "parent_node_id": scene_node.parent_node_id,
            "child_node_id": scene_node.node_id,
            "choice_text": scene_node.parent_choice_text
        }
        run_cypher(query_create_choice, params_choice)

def update_scene_choices(node_id: str, choices: List[str]):
    """
    [신규] 명세서 준수: Scene 노드 안에 선택지 리스트(List<String>)를 업데이트합니다.
    스토리 생성 로직상, 노드가 먼저 생성되고 나중에 선택지가 결정되므로 이 함수가 필요합니다.
    """
    query = """
    MATCH (n:Scene {node_id: $node_id})
    SET n.choice_text = $choices
    """
    params = {
        "node_id": node_id,
        "choices": choices
    }
    run_cypher(query, params)
    # print(f"  -> Neo4j Update: Node {node_id} updated with choices {choices}")

def create_universe_node(universe_node_obj: UniverseNode) -> dict:
    node_data = asdict(universe_node_obj)
    query = """
    MERGE (n:Universe {universe_id: $props.universe_id})
    ON CREATE SET n = $props
    ON MATCH SET n += $props
    RETURN n
    """
    params = {"props": node_data}
    result = run_cypher(query, params)
    return result[0] if result else {}