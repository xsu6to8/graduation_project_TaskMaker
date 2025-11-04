"""
Ambiguity Resolver Module

불특정 표현/대명사를 구체적 객체로 해결하고 의미적 불일치를 감지하여 수정
LLM 기반 Hybrid 선택 지원
"""

import os
import json
import math
from typing import Dict, List, Optional, Any

# LLM Integration
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

class AmbiguityResolver:
    """불특정 표현, 대명사, 의미적 불일치 해결"""
    
    # 불특정 표현 키워드
    AMBIGUOUS_TARGETS = {
        'anything', 'something', 'whatever', 'anywhere',
        'any object', 'some object', 'null', None
    }
    
    PICKABLE_OBJECTS = {
        'book', 'books', 'note', 'penholder', 'pencil',
        'pinkhighlighter', 'stapler', 'tape', 'thesispaper', 'yellowhighlighter',
        'trash', 'garbage'
    }
    
    UNPICKABLE_OBJECTS = {
        'laptop', 'monitor', 'tv', 'board', 'door',
        'chair', 'counter', 'desk', 'waterdispensor',
        'switch', 'bin', 'trashcan',
        'lab', 'classroom', 'library', 'hallway'
    }
    
    PLACEABLE_PRIORITY = {
        'primary': ['desk', 'counter'],
        'secondary': ['bin', 'trashcan'],
        'tertiary': ['penholder'],
        'fallback': ['books', 'monitor', 'waterdispensor']
    }
    
    PLACEABLE_OBJECTS = set(
        PLACEABLE_PRIORITY['primary'] +
        PLACEABLE_PRIORITY['secondary'] +
        PLACEABLE_PRIORITY['tertiary'] +
        PLACEABLE_PRIORITY['fallback']
    )
    
    ROOMS = ['lab', 'classroom', 'library', 'hallway']
    
    TRASH_COMMANDS = {
        'trash', 'waste', 'throw', 'discard', 
        'throw away', 'get rid of', 'dispose'
    }
    
    CLEAN_COMMANDS = {
        'clean', 'clear', 'tidy', 'organize', 'tidy up'
    }
    
    PRONOUNS = {
        'it', 'that', 'this', 'them', 'these', 'those'
    }
    
    def __init__(self, okb_path: str = None):
        # OKB 경로 설정
        if okb_path is None:
            base_path = os.path.dirname(os.path.abspath(__file__))
            self.okb_path = os.path.join(base_path, "..", "OKB")
        else:
            self.okb_path = okb_path
        
        # 데이터 저장소
        self.objects_metadata = {}  # lab_objects.json 데이터
        self.environment = {}  # lab_env.json 데이터
        self.relations = []  # relation_information 데이터
        
        # OKB 데이터 로드
        self._load_okb_data()
        
        # 최근 상호작용한 객체 히스토리 (LIFO - 최신순)
        self.interaction_history = []  # [{"object": "book_01", "command": "pick", "timestamp": ...}, ...]
        self.max_history_size = 10  # 최대 히스토리 크기
        
        # ========================================================================
        # LLM Integration (Hybrid 방식)
        # ========================================================================
        # OpenAI API 키 로드
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.llm_enabled = OPENAI_AVAILABLE and self.openai_api_key is not None
        
        # LLM 설정
        self.llm_model = "gpt-4o-mini"  # 빠르고 저렴
        self.llm_temperature = 0.3  # 일관성 중시
        self.llm_max_tokens = 200  # 짧은 응답
        
        # 통계
        self.stats = {
            "total_resolutions": 0,
            "successful_resolutions": 0,
            "failed_resolutions": 0,
            "pronoun_resolutions": 0,
            "semantic_mismatch_resolutions": 0,  # 의미적 불일치 해결 횟수
            "llm_calls": 0,  # LLM 호출 횟수
            "llm_successes": 0,  # LLM 성공 횟수
            "llm_failures": 0  # LLM 실패 횟수
        }
        
        
        # LLM 상태 출력
        if self.llm_enabled:
            pass  # LLM enabled
        else:
            if not OPENAI_AVAILABLE:
                pass  # OpenAI not available
            else:
                pass  # API key not set
    
    def _load_okb_data(self):
        """OKB 데이터 로드 (lab_objects.json, lab_env.json)"""
        try:
            # lab_objects.json 로드
            objects_path = os.path.join(self.okb_path, "lab_objects.json")
            with open(objects_path, 'r', encoding='utf-8') as f:
                self.objects_metadata = json.load(f)
            
            # lab_env.json 로드
            env_path = os.path.join(self.okb_path, "lab_env.json")
            with open(env_path, 'r', encoding='utf-8') as f:
                self.environment = json.load(f)
            
            self._load_relations()
            
        except (FileNotFoundError, json.JSONDecodeError, Exception):
            self.objects_metadata = {}
            self.environment = {}
            self.relations = []
    
    def reload_environment(self):
        """환경 데이터 재로드 (plan seq 성공 후, Unity 업데이트 반영)"""
        try:
            env_path = os.path.join(self.okb_path, "lab_env.json")
            with open(env_path, 'r', encoding='utf-8') as f:
                self.environment = json.load(f)
            self._load_relations()
        except Exception:
            self.relations = []
    
    def _load_relations(self):
        """relation_information 로드 및 파싱"""
        try:
            # relation_information 키가 있는지 확인
            relations_data = self.environment.get('relation_information', [])
            
            # 데이터가 없으면 빈 리스트
            if not relations_data:
                self.relations = []
                return
            
            # 각 relation 안전하게 파싱
            valid_relations = []
            for rel in relations_data:
                try:
                    # 필수 키 확인
                    if not isinstance(rel, dict):
                        continue
                    
                    subject = rel.get('subject')
                    # Unity: predicate, 기존: relation
                    relation = rel.get('predicate', rel.get('relation'))
                    # Unity: target, 기존: object
                    obj = rel.get('target', rel.get('object'))
                    
                    # 필수 값 검증
                    if not subject or not relation or not obj:
                        continue
                    
                    # 유효한 relation 추가
                    valid_relations.append({
                        'subject': str(subject),
                        'relation': str(relation).lower(),  # 소문자로 정규화
                        'object': str(obj)
                    })
                    
                except Exception as e:
                    # 개별 relation 파싱 실패는 무시하고 계속
                    continue
            
            self.relations = valid_relations
        except Exception:
            self.relations = []
    
    def get_objects_in_relation(self, relation_type: str, target_object: str) -> List[str]:
        """특정 관계를 가진 객체들 반환 (예: desk 위의 책들)"""
        if not self.relations:
            return []
        
        try:
            relation_type_lower = relation_type.lower()
            results = []
            
            for rel in self.relations:
                # relation 타입과 object가 일치하는지 확인
                if rel['relation'] == relation_type_lower and rel['object'] == target_object:
                    results.append(rel['subject'])
            
            return results
        except Exception:
            return []
    
    def get_object_relation(self, subject: str) -> Optional[Dict[str, str]]:
        """특정 객체의 관계 정보 반환"""
        if not self.relations:
            return None
        
        try:
            for rel in self.relations:
                if rel['subject'] == subject:
                    return {
                        'relation': rel['relation'],
                        'object': rel['object']
                    }
            
            return None
        except Exception:
            return None
    
    def _parse_spatial_info(self, spatial_info: str) -> Optional[Dict[str, str]]:
        """spatial_info 파싱 (예: "on desk" → {"relation": "on", "object": "desk"})"""
        if not spatial_info:
            return None
        
        try:
            # 공백으로 분리
            parts = spatial_info.strip().lower().split()
            
            if len(parts) < 2:
                return None
            
            # 첫 단어가 전치사(relation), 나머지는 객체
            relation = parts[0]
            obj = ' '.join(parts[1:])
            
            # 알려진 전치사만 허용
            known_relations = ['on', 'in', 'at', 'near', 'under', 'above', 'beside', 'behind', 'by']
            if relation not in known_relations:
                return None
            
            return {
                "relation": relation,
                "object": obj
            }
            
        except Exception:
            return None
    
    def filter_by_spatial_relation(
        self, 
        candidates: List[str], 
        relation_type: str, 
        reference_object: str
    ) -> List[str]:
        """공간 관계로 후보 필터링 (예: desk 위의 book만)"""
        if not self.relations:
            return candidates
        
        try:
            # reference_object의 모든 인스턴스 찾기
            reference_instances = []
            all_objects = self.environment.get("objects", [])
            ref_class = reference_object.lower()
            
            for obj in all_objects:
                obj_name = obj.get("name", "")
                obj_class = self._extract_class_from_name(obj_name)
                if obj_class == ref_class:
                    reference_instances.append(obj_name)
            
            if not reference_instances:
                return candidates
            
            # 각 참조 객체에 대해 relation_type으로 연결된 객체 찾기
            related_objects = set()
            for ref_instance in reference_instances:
                objs = self.get_objects_in_relation(relation_type, ref_instance)
                related_objects.update(objs)
            
            if not related_objects:
                return candidates
            
            # candidates 중 related_objects에 포함된 것만 필터링
            filtered = [c for c in candidates if c in related_objects]
            
            if not filtered:
                return candidates
            
            return filtered
        except Exception:
            return candidates
    
    def add_to_history(self, obj_instance: str, command: str, context: Dict = None):
        """객체 상호작용 히스토리에 추가"""
        import time
        
        # 히스토리 엔트리 생성
        entry = {
            "object": obj_instance,
            "command": command,
            "timestamp": time.time(),
            "object_class": self._extract_class_from_name(obj_instance)
        }
        
        # 컨텍스트 추가 (있을 경우)
        if context:
            entry["context"] = context
        
        # 히스토리 맨 앞에 추가 (최신순)
        self.interaction_history.insert(0, entry)
        
        # 최대 크기 유지
        if len(self.interaction_history) > self.max_history_size:
            self.interaction_history = self.interaction_history[:self.max_history_size]
        
    
    def get_latest_object(self, command_filter: str = None) -> Optional[str]:
        """가장 최근 상호작용한 객체 반환"""
        if not self.interaction_history:
            return None
        
        # 필터링 없이 가장 최근 객체 반환
        if command_filter is None:
            return self.interaction_history[0]["object"]
        
        # 특정 명령으로 필터링
        for entry in self.interaction_history:
            if entry["command"] == command_filter:
                return entry["object"]
        
        return None
    
    def get_latest_pickable_object(self) -> Optional[str]:
        """가장 최근 상호작용한 pickable 객체 반환"""
        for entry in self.interaction_history:
            obj_class = entry.get("object_class", "")
            if self._is_pickable(obj_class):
                return entry["object"]
        
        return None
    
    def get_latest_placeable_object(self) -> Optional[str]:
        """가장 최근 상호작용한 placeable 객체 반환"""
        for entry in self.interaction_history:
            obj_class = entry.get("object_class", "")
            if self._is_placeable(obj_class):
                return entry["object"]
        
        return None
    
    def clear_history(self):
        """세션 히스토리 초기화 (새 세션 시작 시 사용)"""
        self.interaction_history = []
    
    def print_history(self):
        """현재 히스토리 출력 (디버깅용)"""
        for i, entry in enumerate(self.interaction_history, 1):
            pass  # History display removed for release
    
    def is_ambiguous(self, target: str) -> bool:
        # None 체크
        if target is None:
            return True
        
        # 문자열 소문자 변환하여 체크
        target_lower = str(target).lower()
        
        # AMBIGUOUS_TARGETS에 포함되어 있는지 확인
        return target_lower in self.AMBIGUOUS_TARGETS
    
    # ========================================================================
    # Semantic Mismatch Detection (의미적 불일치 감지)
    # ========================================================================
    def detect_semantic_mismatch(self, command: str, target: str) -> bool:
        """
        command와 target의 의미적 불일치 감지
        
        Args:
            command: 명령어 (예: switchoff, pick, place)
            target: 대상 객체 (예: library, desk)
            
        Returns:
            bool: 의미적 불일치가 있으면 True
            
        예시:
            - switchoff + room → True (switch만 가능)
            - pick + room → True (pickable 객체만 가능)
            - place + pickable_object → True (placeable 표면만 가능)
        """
        if not command or not target:
            return False
        
        target_lower = target.lower()
        obj_class = self._extract_class_from_name(target)
        
        # switchon/switchoff 명령: switch만 가능, room은 불가
        if command in ["switchon", "switchoff", "turn_on", "turn_off"]:
            if obj_class in self.ROOMS or target_lower in self.ROOMS:
                return True
            if obj_class != "switch" and "switch" not in obj_class:
                return True
        
        # pick/grab/take 명령: pickable 객체만 가능
        elif command in ["pick", "grab", "take"]:
            if obj_class in self.ROOMS or target_lower in self.ROOMS:
                return True
            if not self._is_pickable(obj_class):
                return True
        
        # place/put 명령: placeable 표면만 가능
        elif command in ["place", "put", "drop"]:
            if obj_class in self.ROOMS or target_lower in self.ROOMS:
                return True
            if self._is_pickable(obj_class):
                return True
        
        return False
    
    def resolve_semantic_mismatch_with_llm(
        self, 
        command: str, 
        target: str, 
        original_input: str
    ) -> Optional[str]:
        """
        LLM을 사용하여 의미적 불일치 해결
        """
        if not self.llm_enabled:
            return None
        
        self.stats["llm_calls"] += 1
        
        try:
            # 1. target이 room인 경우 해당 room의 객체들 찾기
            target_lower = target.lower()
            obj_class = self._extract_class_from_name(target)
            
            candidate_objects = []
            
            if obj_class in self.ROOMS or target_lower in self.ROOMS:
                room_name = target_lower if target_lower in self.ROOMS else obj_class
                candidate_objects = self._find_objects_in_room(room_name)
            else:
                all_objects = self.environment.get("objects", [])
                candidate_objects = [obj.get("name", "") for obj in all_objects if obj.get("name")]
            
            if not candidate_objects:
                self.stats["llm_failures"] += 1
                return None
            
            # 2. command에 적합한 객체만 필터링
            filtered_candidates = self.filter_by_action(command, 
                [{"name": name} for name in candidate_objects])
            
            if not filtered_candidates:
                self.stats["llm_failures"] += 1
                return None
            
            # 3. LLM 프롬프트 생성
            prompt = self._create_semantic_resolution_prompt(
                command=command,
                target=target,
                original_input=original_input,
                candidates=filtered_candidates,
                target_type=obj_class
            )
            
            # 4. LLM 호출
            client = openai.OpenAI(api_key=self.openai_api_key)
            
            response = client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an intelligent robot assistant that understands user intent and resolves ambiguous commands. Respond ONLY with a valid JSON object."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.llm_temperature,
                max_tokens=self.llm_max_tokens,
                response_format={"type": "json_object"}
            )
            
            # 5. 응답 파싱
            result_text = response.choices[0].message.content.strip()
            result = json.loads(result_text)
            
            resolved = result.get("resolved_target")
            reason = result.get("reason", "No reason provided")
            confidence = result.get("confidence", 0.5)
            
            # 6. 검증
            if resolved not in filtered_candidates:
                self.stats["llm_failures"] += 1
                return None
            
            self.stats["llm_successes"] += 1
            
            return resolved
            
        except json.JSONDecodeError as e:
            self.stats["llm_failures"] += 1
            return None
        except Exception as e:
            self.stats["llm_failures"] += 1
            return None
    
    def _create_semantic_resolution_prompt(
        self,
        command: str,
        target: str,
        original_input: str,
        candidates: List[str],
        target_type: str
    ) -> str:
        """의미적 불일치 해결을 위한 LLM 프롬프트 생성"""
        
        # Robot 위치 정보
        robot_room = self._get_robot_room()
        
        # 후보 객체 상세 정보 수집
        candidates_info = []
        for obj_name in candidates:
            obj_class = self._extract_class_from_name(obj_name)
            obj_room = self._get_object_room(obj_name)
            
            candidates_info.append({
                "name": obj_name,
                "class": obj_class,
                "room": obj_room
            })
        
        # Command 설명
        command_desc = {
            "switchoff": "turn off/deactivate a switch",
            "switchon": "turn on/activate a switch",
            "pick": "pick up a small object",
            "grab": "grab a small object",
            "take": "take a small object",
            "place": "place an object on a surface",
            "put": "put an object on a surface"
        }
        
        prompt = f"""You are helping a robot understand user intent when there is a semantic mismatch.

**User's Original Input:** "{original_input}"

**Parsed Command:** {command} (means: {command_desc.get(command, command)})
**Parsed Target:** {target} (type: {target_type})

**Problem:** 
The command "{command}" cannot be directly applied to "{target}" (a {target_type}).
- "{command}" requires a specific object type, not a {target_type}.

**Available Objects that CAN be used with "{command}":**
"""
        
        for i, info in enumerate(candidates_info, 1):
            prompt += f"\n{i}. {info['name']}"
            prompt += f"\n   - Type: {info['class']}"
            prompt += f"\n   - Location: {info['room']}"
        
        prompt += f"""

**Robot's Current Location:** {robot_room or 'unknown'}

**Your Task:**
Based on the user's intent in "{original_input}", which object should the robot actually {command}?

Consider:
1. The user likely meant to {command} an object IN/AT the {target}, not the {target} itself
2. Choose the most logical object that matches the user's intent
3. Prefer objects in the same location as the {target}

Respond with ONLY a JSON object in this format:
{{
    "resolved_target": "exact_object_name_from_candidates",
    "reason": "brief explanation (1-2 sentences)",
    "confidence": 0.0-1.0
}}

Example response:
{{
    "resolved_target": "switch_02",
    "reason": "User wants to turn off the lights in the library, so targeting the switch in that room",
    "confidence": 0.95
}}

Respond ONLY with the JSON object, no other text."""
        
        return prompt
    
    def _find_objects_in_room(self, room_name: str) -> List[str]:
        """
        특정 방에 있는 모든 객체 찾기
        
        Args:
            room_name: 방 이름 (예: "library", "lab")
            
        Returns:
            List[str]: 해당 방에 있는 객체 인스턴스 이름 리스트
        """
        all_objects = self.environment.get("objects", [])
        room_objects = []
        
        for obj in all_objects:
            obj_name = obj.get("name", "")
            obj_room = obj.get("current_room") or obj.get("room")
            
            if obj_room and obj_room.lower() == room_name.lower():
                room_objects.append(obj_name)
        
        return room_objects
    
    # ========================================================================
    # LLM Integration: Hybrid Selection Method
    # ========================================================================
    
    def llm_select_best(
        self, 
        candidates: List[str], 
        command: str, 
        original_target: str = None,
        original_input: str = None
    ) -> Optional[Dict[str, Any]]:
        """
        LLM을 사용하여 후보 중 최적의 객체 선택 (Hybrid 방식)
        
        Args:
            candidates: 후보 객체 리스트 (규칙 기반으로 이미 필터링됨)
            command: 명령어 (pick, place, move 등)
            original_target: 원본 target 표현 (예: "anything", "it")
            original_input: 사용자의 원본 입력 텍스트
            
        Returns:
            Dict: {"selected": str, "reason": str, "confidence": float} 또는 None
        """
        if not self.llm_enabled:
            return None
        
        if not candidates or len(candidates) == 0:
            return None
        
        # 단일 후보면 LLM 불필요
        if len(candidates) == 1:
            return {
                "selected": candidates[0],
                "reason": "Only one candidate available",
                "confidence": 1.0,
                "method": "single_candidate"
            }
        
        self.stats["llm_calls"] += 1
        
        try:
            # 1. 컨텍스트 정보 수집
            robot_pos = self._get_robot_position()
            robot_room = self._get_robot_room()
            
            # 2. 후보 정보 구조화
            candidates_info = []
            for candidate in candidates:
                obj_pos = self._get_object_position(candidate)
                obj_room = self._get_object_room(candidate)
                obj_class = self._extract_class_from_name(candidate)
                
                distance = None
                if robot_pos and obj_pos:
                    distance = self._calculate_distance(robot_pos, obj_pos)
                
                candidates_info.append({
                    "name": candidate,
                    "class": obj_class,
                    "room": obj_room,
                    "distance": round(distance, 2) if distance else None,
                    "same_room": (obj_room == robot_room) if (obj_room and robot_room) else None
                })
            
            # 3. 히스토리 정보 (최근 3개)
            recent_history = []
            for entry in self.interaction_history[:3]:
                recent_history.append({
                    "object": entry["object"],
                    "command": entry["command"]
                })
            
            # 4. LLM 프롬프트 생성
            prompt = self._create_llm_selection_prompt(
                command=command,
                original_target=original_target,
                original_input=original_input,
                candidates=candidates_info,
                robot_room=robot_room,
                recent_history=recent_history
            )
            
            # 5. LLM 호출
            client = openai.OpenAI(api_key=self.openai_api_key)
            
            response = client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an intelligent robot assistant that helps select the most appropriate object for a given task. Respond ONLY with a valid JSON object."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.llm_temperature,
                max_tokens=self.llm_max_tokens,
                response_format={"type": "json_object"}
            )
            
            # 6. 응답 파싱
            result_text = response.choices[0].message.content.strip()
            result = json.loads(result_text)
            
            selected = result.get("selected")
            reason = result.get("reason", "No reason provided")
            confidence = result.get("confidence", 0.5)
            
            # 7. 검증
            if selected not in candidates:
                self.stats["llm_failures"] += 1
                return None
            
            
            self.stats["llm_successes"] += 1
            
            return {
                "selected": selected,
                "reason": reason,
                "confidence": confidence,
                "method": "llm_hybrid"
            }
            
        except (json.JSONDecodeError, Exception):
            self.stats["llm_failures"] += 1
            return None
    
    def _create_llm_selection_prompt(
        self,
        command: str,
        original_target: str,
        original_input: str,
        candidates: List[Dict],
        robot_room: str,
        recent_history: List[Dict]
    ) -> str:
        """LLM 선택을 위한 프롬프트 생성"""
        
        prompt = f"""You are helping a robot select the most appropriate object for a task.

**Task Information:**
- Command: {command}
- Original request: "{original_input or original_target or 'N/A'}"
- Robot's current location: {robot_room or 'unknown'}

**Candidate Objects (already filtered by rules):**
"""
        
        for i, candidate in enumerate(candidates, 1):
            prompt += f"\n{i}. {candidate['name']}"
            prompt += f"\n   - Type: {candidate['class']}"
            prompt += f"\n   - Room: {candidate['room']}"
            if candidate['distance'] is not None:
                prompt += f"\n   - Distance: {candidate['distance']}m"
            if candidate['same_room'] is not None:
                prompt += f"\n   - Same room as robot: {'Yes' if candidate['same_room'] else 'No'}"
        
        if recent_history:
            prompt += f"\n\n**Recent Interaction History:**"
            for entry in recent_history:
                prompt += f"\n- {entry['command']} {entry['object']}"
        
        prompt += f"""

**Selection Criteria:**
1. Objects in the same room as the robot are preferred
2. Closer objects are preferred
3. Consider the command type (pick: small items, place: surfaces)
4. Consider recent history for context

**Your Task:**
Select the MOST APPROPRIATE object from the candidates and respond with ONLY a JSON object in this format:
{{
    "selected": "exact_object_name_from_candidates",
    "reason": "brief explanation (1-2 sentences)",
    "confidence": 0.0-1.0
}}

Example response:
{{
    "selected": "book_01",
    "reason": "Closest pickable item in the same room as the robot",
    "confidence": 0.9
}}

Respond ONLY with the JSON object, no other text."""
        
        return prompt
    
    def is_pronoun(self, target: str) -> bool:
        if target is None:
            return False
        
        # 문자열 소문자 변환하여 체크
        target_lower = str(target).lower().strip()
        
        # PRONOUNS에 포함되어 있는지 확인
        return target_lower in self.PRONOUNS
    
    def detect_pronoun_in_input(self, input_text: str) -> Optional[str]:
        if not input_text:
            return None
        
        input_lower = input_text.lower()
        
        # 대명사 찾기 (단어 경계 고려)
        import re
        for pronoun in self.PRONOUNS:
            # 단어 경계로 구분된 대명사 찾기
            pattern = r'\b' + pronoun + r'\b'
            if re.search(pattern, input_lower):
                return pronoun
        
        return None
    
    def resolve_pronoun(self, pronoun: str, command: str, original_input: str = "") -> Optional[str]:
        """대명사를 히스토리 기반으로 실제 객체로 해결"""
        
        # 히스토리가 비어있으면 해결 불가
        if not self.interaction_history:
            return None
        
        # 명령 타입에 따라 적절한 객체 찾기
        resolved_object = None
        
        if command in ["pick", "grab", "take"]:
            resolved_object = self.get_latest_pickable_object()
        elif command in ["place", "put", "drop"]:
            resolved_object = self.get_latest_pickable_object()
        elif command in ["move", "go", "walk"]:
            resolved_object = self.get_latest_object()
        else:
            resolved_object = self.get_latest_object()
        
        # 통계 업데이트
        if resolved_object:
            self.stats["pronoun_resolutions"] += 1
        
        return resolved_object
    
    def try_resolve_pronoun_in_task(self, task: Dict) -> bool:
        """task에서 대명사를 감지하고 해결 시도"""
        grounded = task.get("grounded", {})
        target = grounded.get("target")
        command = grounded.get("command")
        original_input = task.get("input", "")
        
        # target이 대명사인지 확인
        if not self.is_pronoun(target):
            # target이 대명사가 아니지만 원본 입력에 대명사가 있는지 확인
            detected_pronoun = self.detect_pronoun_in_input(original_input)
            if not detected_pronoun:
                return False
            
            # ParLex가 대명사를 다른 값으로 변환했을 수 있으므로 일단 진행
        else:
            detected_pronoun = target
        
        # 대명사 해결
        resolved_object = self.resolve_pronoun(detected_pronoun, command, original_input)
        
        if resolved_object:
            # task 업데이트
            grounded["target"] = resolved_object
            
            # 메타데이터 추가
            if "metadata" not in task:
                task["metadata"] = {}
            
            task["metadata"]["pronoun_resolved"] = True
            task["metadata"]["original_pronoun"] = detected_pronoun
            task["metadata"]["resolved_to"] = resolved_object
            
            return True
        return False
    
    def resolve(self, parlex_result: Dict) -> Dict:
        """불특정 표현/대명사를 구체적 객체로 해결"""
        
        if not parlex_result.get("tasks"):
            return parlex_result
        
        converted_tasks = []
        for task in parlex_result.get("tasks", []):
            original_input = task.get("input", "")
            
            # trash 명령 감지 (입력 텍스트 기반)
            if self._detect_trash_command(original_input):
                # trash 명령을 pick + place bin으로 변환
                new_tasks = self._convert_to_trash_tasks(task)
                converted_tasks.extend(new_tasks)
            # clean 명령 감지 (입력 텍스트 기반)
            elif self._detect_clean_command(original_input):
                # clean 명령을 여러 pick + place bin으로 변환
                new_tasks = self._convert_to_clean_tasks(task)
                converted_tasks.extend(new_tasks)
            else:
                # 일반 task는 그대로 추가
                converted_tasks.append(task)
        
        # 변환된 tasks로 교체
        parlex_result["tasks"] = converted_tasks
        
        # 각 task 순회
        for i, task in enumerate(parlex_result.get("tasks", []), 1):
            grounded = task.get("grounded", {})
            target = grounded.get("target")
            command = grounded.get("command")

            # 대명사 감지 및 해결 시도
            if self.try_resolve_pronoun_in_task(task):
                # 대명사가 해결되었으면 target 업데이트
                target = task.get("grounded", {}).get("target")
            
            # 원본 입력에서 불특정 표현 확인 (LLM fallback이 이미 변환한 경우 대비)
            original_input = task.get("input", "").lower()
            has_ambiguous_in_original = any(
                ambiguous_term in original_input 
                for ambiguous_term in ['anything', 'something', 'whatever', 'anywhere']
            )
            
            # LLM fallback이 사용되었고 원본에 불특정 표현이 있는 경우
            is_llm_fallback = task.get("metadata", {}).get("method") == "llm_fallback"
            
            # 불특정 표현 감지 (직접 또는 LLM fallback 사용 시)
            if self.is_ambiguous(target) or (is_llm_fallback and has_ambiguous_in_original):
                self.stats["total_resolutions"] += 1
                
                # 1. 환경의 모든 객체 가져오기
                all_objects = self.environment.get("objects", [])
                
                # 2. 명령어에 따라 필터링
                candidates = self.filter_by_action(command, all_objects)
                
                # 2.5. spatial_info 기반 관계 필터링 (있으면)
                spatial_info = grounded.get("spatial_info")
                spatial_filtered = False
                if spatial_info and candidates:
                    parsed_spatial = self._parse_spatial_info(spatial_info)
                    if parsed_spatial:
                        candidates = self.filter_by_spatial_relation(
                            candidates,
                            parsed_spatial["relation"],
                            parsed_spatial["object"]
                        )
                        spatial_filtered = True  # 필터링 적용됨
                
                # 3. 우선순위 + 방 + 거리 기반 선택
                if candidates:
                    # 우선순위 + 방 우선 선택 (같은 방 객체 중 가장 가까운 것)
                    resolved_target = self.select_with_priority_and_distance(candidates, command)
                    
                    if resolved_target:
                        
                        # 4. ParLex 결과 업데이트
                        grounded["target"] = resolved_target
                        
                    # spatial filtering이 적용된 경우 spatial_info 제거
                    if spatial_filtered:
                        grounded["spatial_info"] = None
                        
                        # 5. 메타데이터 추가
                        if "metadata" not in task:
                            task["metadata"] = {}
                        
                        task["metadata"]["ambiguity_resolved"] = True
                        task["metadata"]["original_target"] = target
                        task["metadata"]["candidates_count"] = len(candidates)
                        
                        # move 명령(room 선택)일 때는 거리 정보 스킵
                        if command in ["move", "go", "walk"]:
                            task["metadata"]["resolution_method"] = "room_selection"
                            task["metadata"]["selected_room"] = resolved_target
                        else:
                            # 일반 객체 선택 시 상세 정보 수집
                            robot_pos = self._get_robot_position()
                            robot_room = self._get_robot_room()
                            obj_pos = self._get_object_position(resolved_target)
                            obj_room = self._get_object_room(resolved_target)
                            distance = self._calculate_distance(robot_pos, obj_pos) if (robot_pos and obj_pos) else None
                            
                            task["metadata"]["resolution_method"] = "room_priority_distance"
                            task["metadata"]["robot_room"] = robot_room
                            task["metadata"]["selected_object_room"] = obj_room
                            task["metadata"]["same_room"] = (obj_room == robot_room) if (obj_room and robot_room) else None
                            task["metadata"]["distance"] = f"{distance:.2f}m" if distance else None
                        
                        self.stats["successful_resolutions"] += 1
                    else:
                        self.stats["failed_resolutions"] += 1
                else:
                    self.stats["failed_resolutions"] += 1
            
            elif self.detect_semantic_mismatch(command, target):
                self.stats["total_resolutions"] += 1
                
                # LLM으로 의미적 불일치 해결
                original_input = task.get("input", "")
                resolved_target = self.resolve_semantic_mismatch_with_llm(
                    command=command,
                    target=target,
                    original_input=original_input
                )
                
                if resolved_target:
                    # ParLex 결과 업데이트
                    grounded["target"] = resolved_target
                    
                    # 메타데이터 추가
                    if "metadata" not in task:
                        task["metadata"] = {}
                    
                    task["metadata"]["semantic_mismatch_resolved"] = True
                    task["metadata"]["original_target"] = target
                    task["metadata"]["resolved_target"] = resolved_target
                    task["metadata"]["resolution_method"] = "semantic_mismatch_llm"
                    
                    # 상세 정보 수집
                    robot_pos = self._get_robot_position()
                    robot_room = self._get_robot_room()
                    obj_pos = self._get_object_position(resolved_target)
                    obj_room = self._get_object_room(resolved_target)
                    distance = self._calculate_distance(robot_pos, obj_pos) if (robot_pos and obj_pos) else None
                    
                    task["metadata"]["robot_room"] = robot_room
                    task["metadata"]["selected_object_room"] = obj_room
                    task["metadata"]["same_room"] = (obj_room == robot_room) if (obj_room and robot_room) else None
                    task["metadata"]["distance"] = f"{distance:.2f}m" if distance else None
                    
                    self.stats["successful_resolutions"] += 1
                    self.stats["semantic_mismatch_resolutions"] += 1
                else:
                    self.stats["failed_resolutions"] += 1
            
            else:
                # Class 이름 → 인스턴스 선택
                if target and command and "_" not in target:
                    
                    # 원본 입력에서 room 필터 추출 ("in [room]", "[room] [target]" 패턴)
                    original_input = task.get("input", "")
                    room_filter = self._extract_room_from_input(original_input, target)
                    
                    # 해당 class의 모든 인스턴스 찾기
                    all_objects = self.environment.get("objects", [])
                    class_instances = [
                        obj.get("name", "") 
                        for obj in all_objects 
                        if self._extract_class_from_name(obj.get("name", "")) == target.lower()
                    ]
                    
                    # 🔑 Room 필터 적용 (in [room] 패턴)
                    if room_filter:
                        filtered_by_room = []
                        for inst in class_instances:
                            inst_room = self._get_object_room(inst)
                            if inst_room == room_filter:
                                filtered_by_room.append(inst)
                        
                        # room 필터링된 결과가 있으면 사용
                        if filtered_by_room:
                            class_instances = filtered_by_room
                    
                    if class_instances:
                        selected = None
                        spatial_info = grounded.get("spatial_info")
                        
                        # Case 1: spatial_info가 있으면 spatial filtering 적용
                        if spatial_info:
                            parsed_spatial = self._parse_spatial_info(spatial_info)
                            
                            if parsed_spatial:
                                # spatial relation으로 필터링
                                filtered_candidates = self.filter_by_spatial_relation(
                                    class_instances,
                                    parsed_spatial["relation"],
                                    parsed_spatial["object"]
                                )
                                
                                if filtered_candidates:
                                    # 필터링된 후보 중 가장 가까운 것 선택
                                    selected = self.select_with_room_priority(filtered_candidates)
                                    
                                    # spatial_info 제거 + effective_room 설정
                                    if selected:
                                        grounded["spatial_info"] = None
                                        
                                        # 🔑 CRITICAL: Reference object의 room을 effective_room으로 설정
                                        # "book on counter"에서 book의 실제 위치는 counter가 있는 room
                                        ref_obj = parsed_spatial["object"]
                                        ref_room = self._get_object_room_for_class(ref_obj)
                                        if ref_room:
                                            grounded["effective_room"] = ref_room
                                            task["grounded"]["effective_room"] = ref_room  # Ensure it's set in task as well
                        
                        # Case 2: spatial_info 없으면 room priority + distance로 선택
                        if not selected:
                            selected = self.select_with_room_priority(class_instances)
                        
                        # 선택 결과 적용
                        if selected:
                            grounded["target"] = selected
                            target = selected
                            
                            # 메타데이터 추가
                            if "metadata" not in task:
                                task["metadata"] = {}
                            task["metadata"]["class_resolved"] = True
                            task["metadata"]["original_class"] = target.split("_")[0]
                            task["metadata"]["candidates_count"] = len(class_instances)
                            if spatial_info:
                                task["metadata"]["spatial_filtered"] = True
                
                # Phase 4: 복합 객체명 분리 (예: "classroom desk" → desk + room_filter)
                compound_result = self._resolve_compound_name(target)
                room_filter = compound_result.get("room_filter")
                resolved_object = compound_result.get("object")
                
                # 복합 객체명이 분리된 경우 target 업데이트
                if room_filter and resolved_object != target:
                    grounded["target"] = resolved_object
                    target = resolved_object
                    
                    # metadata에 room_filter 저장 (EnhancedSpatialGrounder에서 활용 가능)
                    if "metadata" not in task:
                        task["metadata"] = {}
                    task["metadata"]["room_filter"] = room_filter
                    task["metadata"]["compound_name_resolved"] = True
                    
                    # place 명령일 때 room_filter를 직접 적용하여 재선택
                    # (EnhancedSpatialGrounder가 복합 객체명을 처리하지 못할 수 있으므로)
                    if command in ["place", "put"]:
                        
                        # 1. 해당 타입의 모든 객체 가져오기
                        all_objects = self.environment.get("objects", [])
                        obj_class = self._extract_class_from_name(resolved_object)
                        
                        # 2. 클래스 일치하는 객체 찾기
                        matching_objects = [
                            obj.get("name") for obj in all_objects
                            if self._extract_class_from_name(obj.get("name", "")) == obj_class
                        ]
                        
                        # 3. room_filter 적용
                        if matching_objects:
                            filtered_by_room = [
                                obj_name for obj_name in matching_objects
                                if self._get_object_room(obj_name) == room_filter
                            ]
                            
                            if filtered_by_room:
                                # 4. 가장 가까운 것 선택
                                best_match = self.select_nearest(filtered_by_room)
                                if best_match and best_match != target:
                                    grounded["target"] = best_match
                                    target = best_match
                                    task["metadata"]["room_filtered_result"] = best_match
                
                # clean 명령은 _convert_to_clean_tasks에서 이미 처리됨 (여기서는 skip)
                
                # pick 명령일 때 집을 수 없는 객체(room, board 등)인지 확인
                if command in ["pick", "grab", "take"] and target:
                    obj_class = self._extract_class_from_name(target)
                    
                    if not self._is_pickable(obj_class):
                        
                        self.stats["total_resolutions"] += 1
                        
                        # 1. 환경의 모든 객체 가져오기
                        all_objects = self.environment.get("objects", [])
                        
                        # 2. 명령어에 따라 필터링
                        candidates = self.filter_by_action(command, all_objects)
                        
                        # 2.5. Phase 4: room_filter 적용 (복합 객체명에서 추출된 경우)
                        if room_filter and candidates:
                            filtered_by_room = [
                                obj_name for obj_name in candidates
                                if self._get_object_room(obj_name) == room_filter
                            ]
                            if filtered_by_room:
                                candidates = filtered_by_room
                        
                        # 2.6. Phase 5: spatial_info 기반 관계 필터링 (있으면)
                        spatial_info = grounded.get("spatial_info")
                        if spatial_info and candidates:
                            parsed_spatial = self._parse_spatial_info(spatial_info)
                            if parsed_spatial:
                                candidates = self.filter_by_spatial_relation(
                                    candidates,
                                    parsed_spatial["relation"],
                                    parsed_spatial["object"]
                                )
                        
                        # 3. 우선순위 + 방 + 거리 기반 선택
                        if candidates:
                            resolved_target = self.select_with_priority_and_distance(candidates, command)
                            
                            if resolved_target:
                                
                                # ParLex 결과 업데이트
                                grounded["target"] = resolved_target
                                
                                # 메타데이터 추가
                                if "metadata" not in task:
                                    task["metadata"] = {}
                                
                                task["metadata"]["ambiguity_resolved"] = True
                                task["metadata"]["original_target"] = target
                                task["metadata"]["resolution_method"] = "invalid_target_override"
                                task["metadata"]["candidates_count"] = len(candidates)
                                
                                # pick 명령이므로 항상 일반 객체 (room 아님)
                                # 상세 정보 수집
                                robot_pos = self._get_robot_position()
                                robot_room = self._get_robot_room()
                                obj_pos = self._get_object_position(resolved_target)
                                obj_room = self._get_object_room(resolved_target)
                                distance = self._calculate_distance(robot_pos, obj_pos) if (robot_pos and obj_pos) else None
                                
                                task["metadata"]["robot_room"] = robot_room
                                task["metadata"]["selected_object_room"] = obj_room
                                task["metadata"]["same_room"] = (obj_room == robot_room) if (obj_room and robot_room) else None
                                task["metadata"]["distance"] = f"{distance:.2f}m" if distance else None
                                
                                self.stats["successful_resolutions"] += 1
                            else:
                                self.stats["failed_resolutions"] += 1
                        else:
                            self.stats["failed_resolutions"] += 1
            
            # 최종 target이 유효한 경우 히스토리에 추가
            final_target = task.get("grounded", {}).get("target")
            if final_target and final_target not in [None, "null", "unknown"]:
                # 대명사가 아닌 경우에만 히스토리에 추가
                # (대명사는 히스토리를 참조하는 것이지, 히스토리에 추가될 대상이 아님)
                if not self.is_pronoun(final_target):
                    # place 명령은 히스토리에 추가하지 않음
                    # (place의 target은 놓을 "위치"이지, 상호작용한 "객체"가 아님)
                    # 예: "place bin_01"에서 bin_01은 용기일 뿐, pick한 객체가 중요함
                    if command not in ["place", "put", "drop", "set"]:
                        self.add_to_history(final_target, command, context={
                            "task_index": i,
                            "original_input": task.get("input", "")
                        })
        
        return parlex_result
    
    def _extract_class_from_name(self, instance_name: str) -> str:
        """인스턴스 이름에서 클래스 추출 (예: book_01 → book)"""
        if not instance_name:
            return "unknown"
        
        # 언더스코어로 분리
        if '_' in instance_name:
            parts = instance_name.split('_')
            # 마지막 부분이 숫자인지 확인
            if len(parts) >= 2 and parts[-1].isdigit():
                return '_'.join(parts[:-1])
        
        return instance_name.lower()
    
    def _resolve_compound_name(self, target: str) -> Dict[str, str]:
        """복합 객체명 분리 (예: "classroom desk" → desk + room_filter)"""
        if not target:
            return {"object": target}
        
        target_lower = target.lower().strip()
        
        # 방 이름으로 시작하는지 확인
        for room in self.ROOMS:
            # "classroom desk", "lab book" 패턴
            if target_lower.startswith(room + " "):
                obj = target_lower[len(room)+1:].strip()
                if obj:  # 객체명이 있는 경우만
                    return {"room_filter": room, "object": obj}
            
            # "desk classroom" (역순) 패턴도 체크
            if target_lower.endswith(" " + room):
                obj = target_lower[:-(len(room)+1)].strip()
                if obj:
                    return {"room_filter": room, "object": obj}
        
        # 복합 객체명이 아닌 경우
        return {"object": target}
    
    def _extract_room_from_input(self, input_text: str, target: str) -> Optional[str]:
        """
        입력 텍스트에서 room 필터 추출 ("in [room]", "[room] [target]" 패턴)
        """
        if not input_text:
            return None
        
        input_lower = input_text.lower().strip()
        
        # 패턴 1: "in [room]" (가장 명확)
        for room in self.ROOMS:
            if f" in {room}" in input_lower or f"in {room}" == input_lower:
                return room
        
        # 패턴 2: "[room] [target]" (예: "library light", "lab switch")
        if target:
            target_lower = target.lower()
            for room in self.ROOMS:
                # "library light" 패턴
                if input_lower.startswith(f"{room} {target_lower}"):
                    return room
                # "light library" 패턴
                if input_lower.endswith(f"{target_lower} {room}"):
                    return room
        
        # 패턴 3: "on [room]" (덜 일반적이지만 처리)
        for room in self.ROOMS:
            if f" on {room}" in input_lower:
                return room
        
        return None

    
    def _detect_trash_command(self, original_input: str) -> bool:
        input_lower = original_input.lower()
        return any(trash_cmd in input_lower for trash_cmd in self.TRASH_COMMANDS)
    
    def _detect_clean_command(self, original_input: str) -> bool:
        input_lower = original_input.lower()
        return any(clean_cmd in input_lower for clean_cmd in self.CLEAN_COMMANDS)
    
    def _select_best_bin(self) -> Optional[str]:
        # 1. 환경에서 모든 bin 찾기
        all_objects = self.environment.get("objects", [])
        bin_candidates = []
        
        for obj in all_objects:
            obj_name = obj.get("name", "")
            obj_class = self._extract_class_from_name(obj_name)
            
            # bin 또는 trashcan 타입 찾기
            if obj_class in ['bin', 'trashcan']:
                bin_candidates.append(obj_name)
        
        if not bin_candidates:
            return None
        
        
        # 2. 로봇 위치 가져오기
        robot_room = self._get_robot_room()
        
        # 3. 현재 방의 bin 우선
        same_room_bins = [
            bin_name for bin_name in bin_candidates
            if self._get_object_room(bin_name) == robot_room
        ]
        
        if same_room_bins:
            selected = self.select_nearest(same_room_bins)
            return selected
        
        # 4. 전체 중 가장 가까운 bin
        selected = self.select_nearest(bin_candidates)
        return selected
    
    def _query_llm_for_clean_selection(self, area_name: str, area_type: str, candidate_objects: List[str]) -> List[str]:
        if not candidate_objects:
            return []
        
        # 객체들의 상세 정보 수집
        objects_info = []
        for obj_name in candidate_objects:
            obj_class = self._extract_class_from_name(obj_name)
            obj_room = self._get_object_room(obj_name)
            obj_pos = self._get_object_position(obj_name)
            
            info = {
                "name": obj_name,
                "class": obj_class,
                "room": obj_room
            }
            if obj_pos:
                info["position"] = f"({obj_pos[0]:.1f}, {obj_pos[1]:.1f}, {obj_pos[2]:.1f})"
            
            objects_info.append(info)
        
        # LLM 프롬프트 생성
        prompt = f"""You are helping a robot clean a {area_type} in a laboratory environment.

Area to clean: {area_name} (type: {area_type})

Objects currently in/on this area:
"""
        for i, obj_info in enumerate(objects_info, 1):
            prompt += f"{i}. {obj_info['name']} (class: {obj_info['class']}, room: {obj_info.get('room', 'unknown')})\n"
        
        prompt += f"""
Task: Select which objects should be thrown into the trash bin when cleaning this {area_type}.

Guidelines:
- SELECT: trash, garbage, waste, disposable items, unnecessary clutter
- KEEP: valuable items (laptop, book, important documents), furniture, permanent fixtures
- When uncertain, prefer to KEEP the item (conservative approach)

Response format: Return ONLY a JSON array of object names to trash, like: ["object1", "object2"]
If nothing should be trashed, return: []

Your selection:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful robot assistant that makes intelligent decisions about cleaning and organizing."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            llm_response = response.choices[0].message.content.strip()
            
            # JSON 파싱
            import re
            json_match = re.search(r'\[.*?\]', llm_response, re.DOTALL)
            if json_match:
                import json
                selected = json.loads(json_match.group(0))
                
                # 유효성 검사: 반환된 객체들이 실제 후보에 있는지 확인
                valid_selected = [obj for obj in selected if obj in candidate_objects]
                return valid_selected
            else:
                return candidate_objects
        except Exception:
            return candidate_objects
    
    def _convert_to_clean_tasks(self, task: Dict) -> List[Dict]:
        original_input = task.get("input", "")
        grounded = task.get("grounded", {})
        target = grounded.get("target")
        
        if not target:
            return [task]
        
        obj_class = self._extract_class_from_name(target)
        area_type = obj_class
        area_name = target
        
        # Case 1: 표면 객체 (desk, table, counter, shelf)
        if obj_class in ["desk", "table", "counter", "shelf"]:
            # 표면 위의 모든 pickable 객체 찾기
            objects_on_surface = self._find_objects_on_surface(obj_class)
            
            if not objects_on_surface:
                return [task]
            
            # pickable 필터링
            pickable_objects = [
                obj for obj in objects_on_surface
                if self._extract_class_from_name(obj) in self.PICKABLE_OBJECTS
            ]
            
            if not pickable_objects:
                return [task]
            
            # LLM에게 선택 요청
            selected_objects = self._query_llm_for_clean_selection(
                area_name=area_name,
                area_type=area_type,
                candidate_objects=pickable_objects
            )
            
        # Case 2: 방(room) 전체
        elif obj_class in self.ROOMS:
            # 해당 방의 모든 pickable 객체 찾기
            all_objects = self.environment.get("objects", [])
            room_objects = [
                obj.get("name", "") 
                for obj in all_objects 
                if obj.get("name") and self._get_object_room(obj.get("name", "")) == target
            ]
            
            # pickable 필터링
            pickable_objects = [
                obj for obj in room_objects
                if self._extract_class_from_name(obj) in self.PICKABLE_OBJECTS
            ]
            
            if not pickable_objects:
                return [task]
            
            # LLM에게 선택 요청
            selected_objects = self._query_llm_for_clean_selection(
                area_name=area_name,
                area_type="room",
                candidate_objects=pickable_objects
            )
            
        # Case 3: 기타 - 단일 객체로 간주하여 trash 처리
        else:
            return self._convert_to_trash_tasks(task)
        
        if not selected_objects:
            return [task]
        
        # 최적 bin 선택
        best_bin = self._select_best_bin()
        if not best_bin:
            return [task]
        
        # 로봇과 같은 방의 객체 우선 + 거리순 정렬
        robot_room = self._get_robot_room()
        robot_pos = self._get_robot_position()
        
        same_room = []
        other_room = []
        
        for obj_name in selected_objects:
            obj_room = self._get_object_room(obj_name)
            if obj_room == robot_room:
                same_room.append(obj_name)
            else:
                other_room.append(obj_name)
        
        # 거리순 정렬
        if robot_pos:
            same_room.sort(key=lambda obj: self._calculate_distance(
                robot_pos, self._get_object_position(obj)) if self._get_object_position(obj) else float('inf'))
            other_room.sort(key=lambda obj: self._calculate_distance(
                robot_pos, self._get_object_position(obj)) if self._get_object_position(obj) else float('inf'))
        
        sorted_objects = same_room + other_room
        
        # 각 객체에 대해 pick + place bin task 생성
        tasks = []
        for obj_name in sorted_objects:
            # pick task
            pick_task = {
                "input": f"pick {obj_name}",
                "grounded": {
                    "command": "pick",
                    "target": obj_name
                },
                "metadata": {
                    "method": "clean_command_llm_selection",
                    "original_input": original_input,
                    "conversion_step": "pick",
                    "area_type": area_type,
                    "area_name": area_name,
                    "llm_selected": True
                }
            }
            tasks.append(pick_task)
            
            # place bin task
            place_task = {
                "input": f"place {best_bin}",
                "grounded": {
                    "command": "place",
                    "target": best_bin
                },
                "metadata": {
                    "method": "clean_command_llm_selection",
                    "original_input": original_input,
                    "conversion_step": "place",
                    "selected_bin": best_bin,
                    "area_type": area_type,
                    "llm_selected": True
                }
            }
            tasks.append(place_task)
        
        return tasks
    
    def _convert_to_trash_tasks(self, task: Dict) -> List[Dict]:
        original_input = task.get("input", "")
        grounded = task.get("grounded", {})
        target = grounded.get("target")
        command = grounded.get("command", "trash")
        
        # target이 대명사인 경우 먼저 해결
        if self.is_pronoun(target) or self.detect_pronoun_in_input(original_input):
            # 대명사 해결 (pick 명령으로 간주하여 pickable 객체 찾기)
            resolved_target = self.resolve_pronoun(target if target else "it", "pick", original_input)
            
            if resolved_target:
                target = resolved_target
            else:
                return [task]
        
        if not target or target in [None, "null", "unknown"]:
            return [task]
        
        best_bin = self._select_best_bin()
        if not best_bin:
            return [task]
        
        # 2. pick task 생성
        pick_task = {
            "input": f"pick {target}",
            "grounded": {
                "command": "pick",
                "target": target
            },
            "metadata": {
                "method": "trash_command_conversion",
                "original_input": original_input,
                "conversion_step": "pick"
            }
        }
        
        # 3. place bin task 생성
        place_task = {
            "input": f"place {best_bin}",
            "grounded": {
                "command": "place",
                "target": best_bin
            },
            "metadata": {
                "method": "trash_command_conversion",
                "original_input": original_input,
                "conversion_step": "place",
                "selected_bin": best_bin
            }
        }
        
        
        return [pick_task, place_task]
    
    def _is_pickable(self, obj_class: str) -> bool:
        # 하드코딩된 화이트리스트 사용 (최고 성능)
        return obj_class in self.PICKABLE_OBJECTS
    
    def _is_placeable(self, obj_class: str) -> bool:
        # 하드코딩된 리스트 사용 (최고 성능)
        return obj_class in self.PLACEABLE_OBJECTS
    
    def _get_placeable_priority(self, obj_class: str) -> int:
        if obj_class in self.PLACEABLE_PRIORITY['primary']:
            return 0  # 가장 높은 우선순위
        elif obj_class in self.PLACEABLE_PRIORITY['secondary']:
            return 1
        elif obj_class in self.PLACEABLE_PRIORITY['tertiary']:
            return 2
        elif obj_class in self.PLACEABLE_PRIORITY['fallback']:
            return 3
        else:
            return 999  # 알 수 없는 객체
    
    def _get_robot_position(self) -> Optional[Dict[str, float]]:
        agent_data = self.environment.get("agent", {})
        position = agent_data.get("position")
        
        if position and all(k in position for k in ['x', 'y', 'z']):
            return {
                'x': float(position['x']),
                'y': float(position['y']),
                'z': float(position['z'])
            }
        return None
    
    def _get_robot_room(self) -> Optional[str]:
        agent_data = self.environment.get("agent", {})
        current_room = agent_data.get("current_room")
        
        if current_room:
            return str(current_room)
        return None
    
    def _get_object_position(self, obj_name: str) -> Optional[Dict[str, float]]:
        all_objects = self.environment.get("objects", [])
        
        for obj in all_objects:
            if obj.get("name") == obj_name:
                position = obj.get("position")
                if position and all(k in position for k in ['x', 'y', 'z']):
                    return {
                        'x': float(position['x']),
                        'y': float(position['y']),
                        'z': float(position['z'])
                    }
        return None
    
    def _get_object_room_for_class(self, obj_class: str) -> Optional[str]:
        all_objects = self.environment.get("objects", [])
        
        for obj in all_objects:
            obj_name = obj.get("name", "")
            if self._extract_class_from_name(obj_name) == obj_class.lower():
                # 첫 번째 매칭되는 인스턴스의 room 반환
                return obj.get("current_room") or obj.get("room")
        
        return None
    
    def _get_object_room(self, obj_name: str) -> Optional[str]:
        all_objects = self.environment.get("objects", [])
        
        for obj in all_objects:
            if obj.get("name") == obj_name:
                current_room = obj.get("current_room")
                if current_room:
                    return str(current_room)
        return None
    
    def _calculate_distance(self, pos1: Dict[str, float], pos2: Dict[str, float]) -> float:
        try:
            dx = pos1['x'] - pos2['x']
            dy = pos1['y'] - pos2['y']
            dz = pos1['z'] - pos2['z']
            
            distance = math.sqrt(dx**2 + dy**2 + dz**2)
            return distance
            
        except (KeyError, TypeError):
            return float('inf')
    
    def select_nearest(self, candidates: List[str]) -> Optional[str]:
        if not candidates:
            return None
        
        # 로봇 위치 가져오기
        robot_pos = self._get_robot_position()
        if not robot_pos:
            return candidates[0]
        
        # 각 후보와의 거리 계산
        distances = []
        for obj_name in candidates:
            obj_pos = self._get_object_position(obj_name)
            if obj_pos:
                distance = self._calculate_distance(robot_pos, obj_pos)
                distances.append({
                    'name': obj_name,
                    'distance': distance,
                    'position': obj_pos
                })
        
        # 거리 없으면 첫 번째 반환
        if not distances:
            return candidates[0]
        
        # 거리 순 정렬
        distances.sort(key=lambda x: x['distance'])
        
        # 가장 가까운 객체 반환
        nearest = distances[0]
        
        return nearest['name']
        
    def select_with_priority_and_distance(self, candidates: List[str], command: str = None) -> Optional[str]:
        if not candidates:
            return None
        
        # move 명령일 때 room 선택 → 첫 번째 반환 (거리 계산 불가)
        if command in ["move", "go", "walk"]:
            # room은 objects 배열에 없으므로 거리 계산 불가
            # 첫 번째 room 반환 (순서 무관)
            selected = candidates[0]
            return selected
        
        # place 명령일 때 우선순위 고려
        if command in ["place", "put"]:
            # 우선순위별로 후보 그룹화
            priority_groups = {0: [], 1: [], 2: [], 3: [], 999: []}
            
            for obj_name in candidates:
                obj_class = self._extract_class_from_name(obj_name)
                priority = self._get_placeable_priority(obj_class)
                priority_groups[priority].append(obj_name)
            
            # 가장 높은 우선순위 그룹 선택
            for priority in sorted(priority_groups.keys()):
                if priority_groups[priority]:
                    candidates = priority_groups[priority]
                    break
        
        # 로봇 위치 및 방 정보
        robot_pos = self._get_robot_position()
        robot_room = self._get_robot_room()
        
        if not robot_pos or not robot_room:
            return self.select_nearest(candidates)
        
        
        # 같은 방 / 다른 방 객체 분리
        same_room_objs = []
        other_room_objs = []
        
        for obj_name in candidates:
            obj_room = self._get_object_room(obj_name)
            if obj_room == robot_room:
                same_room_objs.append(obj_name)
            else:
                other_room_objs.append(obj_name)
        
        
        # ========================================================================
        # Hybrid LLM Integration: 후보가 여러 개일 때 LLM 사용
        # ========================================================================
        
        # 1순위: 같은 방 객체가 있으면 그중 선택
        if same_room_objs:
            
            # LLM으로 최종 선택 (같은 방 후보가 여러 개일 때)
            if len(same_room_objs) > 1 and self.llm_enabled:
                llm_result = self.llm_select_best(
                    candidates=same_room_objs,
                    command=command,
                    original_target="ambiguous",
                    original_input=None
                )
                
                if llm_result and llm_result.get("selected"):
                    selected = llm_result["selected"]
                    obj_pos = self._get_object_position(selected)
                    distance = self._calculate_distance(robot_pos, obj_pos) if obj_pos else float('inf')
                    return selected
            
            # LLM 없거나 실패 시 거리 기반 선택
            selected = self.select_nearest(same_room_objs)
            obj_pos = self._get_object_position(selected)
            distance = self._calculate_distance(robot_pos, obj_pos) if obj_pos else float('inf')
            return selected
        
        # 2순위: 같은 방에 없으면 전체 중 선택
        
        # LLM으로 최종 선택 (전체 후보가 여러 개일 때)
        if len(candidates) > 1 and self.llm_enabled:
            llm_result = self.llm_select_best(
                candidates=candidates,
                command=command,
                original_target="ambiguous",
                original_input=None
            )
            
            if llm_result and llm_result.get("selected"):
                selected = llm_result["selected"]
                obj_room = self._get_object_room(selected)
                obj_pos = self._get_object_position(selected)
                distance = self._calculate_distance(robot_pos, obj_pos) if obj_pos else float('inf')
                return selected
        
        # LLM 없거나 실패 시 거리 기반 선택
        selected = self.select_nearest(candidates)
        obj_room = self._get_object_room(selected)
        obj_pos = self._get_object_position(selected)
        distance = self._calculate_distance(robot_pos, obj_pos) if obj_pos else float('inf')
        return selected
    
    def select_with_room_priority(self, candidates: List[str]) -> Optional[str]:
        """같은 방 우선, 그 중 가장 가까운 객체 반환"""
        if not candidates:
            return None
        
        # 로봇 위치 및 방 정보
        robot_pos = self._get_robot_position()
        robot_room = self._get_robot_room()
        
        if not robot_pos or not robot_room:
            return self.select_nearest(candidates)
        
        
        # 같은 방 / 다른 방 객체 분리
        same_room_objs = []
        other_room_objs = []
        
        for obj_name in candidates:
            obj_room = self._get_object_room(obj_name)
            if obj_room == robot_room:
                same_room_objs.append(obj_name)
            else:
                other_room_objs.append(obj_name)
        
        
        # 1순위: 같은 방 객체가 있으면 그중 가장 가까운 것
        if same_room_objs:
            selected = self.select_nearest(same_room_objs)
            obj_pos = self._get_object_position(selected)
            distance = self._calculate_distance(robot_pos, obj_pos) if obj_pos else float('inf')
            return selected
        
        # 2순위: 같은 방에 없으면 전체 중 가장 가까운 것
        selected = self.select_nearest(candidates)
        obj_room = self._get_object_room(selected)
        obj_pos = self._get_object_position(selected)
        distance = self._calculate_distance(robot_pos, obj_pos) if obj_pos else float('inf')
        return selected
    
    # ========================================================================
    
    def filter_by_action(self, command: str, all_objects: List[Dict]) -> List[str]:
        filtered = []
        
        
        # pick, grab, take 명령어: 집을 수 있는 객체만
        if command in ["pick", "grab", "take"]:
            for obj in all_objects:
                obj_name = obj.get("name", "")
                obj_class = self._extract_class_from_name(obj_name)
                
                if self._is_pickable(obj_class):
                    filtered.append(obj_name)
            
        
        # place, put 명령어: 물건을 놓을 수 있는 표면/컨테이너만
        elif command in ["place", "put"]:
            for obj in all_objects:
                obj_name = obj.get("name", "")
                obj_class = self._extract_class_from_name(obj_name)
                
                if self._is_placeable(obj_class):
                    filtered.append(obj_name)
            
        
        # move, go, walk 명령어: 방(room) 객체만
        elif command in ["move", "go", "walk"]:
            # 하드코딩된 방 리스트 사용 (최고 성능)
            filtered = self.ROOMS.copy()
        
        # switchon, switchoff, turn_on, turn_off 명령어: switch만
        elif command in ["switchon", "switchoff", "turn_on", "turn_off"]:
            for obj in all_objects:
                obj_name = obj.get("name", "")
                obj_class = self._extract_class_from_name(obj_name)
                
                # switch만 허용
                if obj_class == "switch":
                    filtered.append(obj_name)
        
        # open, close 명령어: door만 허용
        elif command in ["open", "close"]:
            for obj in all_objects:
                obj_name = obj.get("name", "")
                obj_class = self._extract_class_from_name(obj_name)
                
                # door만 허용
                if "door" in obj_class.lower():
                    filtered.append(obj_name)
        
        # clean 명령어: 모든 객체 (특수 처리는 resolve에서)
        elif command == "clean":
            filtered = [obj.get("name", "") for obj in all_objects if obj.get("name")]
        
        # 기타 명령어: 모든 객체
        else:
            filtered = [obj.get("name", "") for obj in all_objects if obj.get("name")]
        
        return filtered
    
    def _find_objects_on_surface(self, surface_name: str) -> List[str]:
        # relation_information에서 "on" 관계 찾기
        relations = self.environment.get("relation_information", [])
        
        # surface_name에 해당하는 모든 인스턴스 찾기 (desk → desk_01, desk_02 등)
        all_objects = self.environment.get("objects", [])
        surface_instances = []
        
        for obj in all_objects:
            obj_name = obj.get("name", "")
            obj_class = self._extract_class_from_name(obj_name)
            if obj_class == surface_name.lower():
                surface_instances.append(obj_name)
        
        if not surface_instances:
            return []
        
        # 각 surface 인스턴스 위의 객체 찾기
        objects_on_surface = []
        
        for relation in relations:
            if not isinstance(relation, dict):
                continue
            
            # Unity 형식: predicate, target / 기존 형식: relation, object 모두 지원
            relation_type = relation.get("predicate", relation.get("relation", "")).lower()
            target_obj = relation.get("target", relation.get("object", ""))
            subject_obj = relation.get("subject", "")
            
            # "on" 관계이고, target이 surface 인스턴스 중 하나인 경우
            if relation_type == "on" and target_obj in surface_instances:
                if subject_obj not in objects_on_surface:
                    objects_on_surface.append(subject_obj)
        
        return objects_on_surface
    
    def get_statistics(self) -> Dict[str, Any]:
        """통계 정보 반환"""
        return self.stats.copy()

def create_ambiguity_resolver(okb_path: str = None) -> AmbiguityResolver:
    """AmbiguityResolver 팩토리 함수"""
    return AmbiguityResolver(okb_path)

if __name__ == "__main__":
    # 간단한 테스트
    
    resolver = create_ambiguity_resolver()
    
    # 불특정 표현 감지 테스트
    test_cases = [
        ("anything", True),
        ("something", True),
        ("null", True),
        (None, True),
        ("book", False),
        ("pencil", False),
        ("laptop", False)
    ]
    
    for target, expected in test_cases:
        result = resolver.is_ambiguous(target)
        status = "OK" if result == expected else "FAIL"
    
    # ParLex 결과 처리 테스트
    mock_parlex_result = {
        "original_input": "pick anything",
        "tasks": [{
            "task_id": 1,
            "input": "pick anything",
            "grounded": {
                "command": "pick",
                "target": "anything",
                "target_is_plural": False,
                "spatial_info": None
            }
        }]
    }
    
    result = resolver.resolve(mock_parlex_result)
    
    # 필터링 로직 테스트
    all_objects = resolver.environment.get("objects", [])
    
    # pick 명령어 테스트
    pickable_objects = resolver.filter_by_action("pick", all_objects)
    for obj_name in pickable_objects[:5]:
        obj_class = resolver._extract_class_from_name(obj_name)
        obj_metadata = resolver.objects_metadata.get("objects", {}).get(obj_class, {})
        spatial_props = obj_metadata.get("spatial_properties", {})
    
    # move 명령어 테스트  
    movable_rooms = resolver.filter_by_action("move", all_objects)
    for room_name in movable_rooms:
        pass  # Display room
