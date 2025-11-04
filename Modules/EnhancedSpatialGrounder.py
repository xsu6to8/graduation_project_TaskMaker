"""
Enhanced Spatial Grounder

안정적인 공간 정보 처리 및 객체 참조 해결
다단계 Fallback 및 거리 기반 객체 선택 지원
"""

import os
import json
import time
import re
from typing import List, Optional, Dict, Tuple, Set
from collections import defaultdict

class EnhancedSpatialGrounder:
    """Enhanced Spatial Grounder"""
    
    def __init__(self, env_file_path=None):
        if env_file_path is None:
            base_path = os.path.dirname(os.path.abspath(__file__))
            self.env_file_path = os.path.join(base_path, "..", "OKB", "lab_env.json")
        else:
            self.env_file_path = env_file_path

        self.env_state = None
        self.name_to_class = {}
        self.class_to_names = defaultdict(list)
        self.name_to_room = {}
        self.room_to_names = defaultdict(list)
        
        self.enable_fuzzy_matching = True
        self.enable_exhaustive_search = True
        self.enable_multiple_fallbacks = True
        
        self.simulated_robot_room = None
        self.task_sequence = []
    
    def _simulate_robot_movement(self, target_room: str):
        """로봇 이동 시뮬레이션"""
        if target_room:
            self.simulated_robot_room = target_room
            
            try:
                target_room_position = self._get_room_center_position(target_room)
                if target_room_position:
                    self._update_robot_position_in_file(target_room_position, target_room)
            except Exception:
                pass
    
    def _get_room_center_position(self, room_name: str) -> Optional[Dict]:
        """방의 중심 좌표 계산"""
        try:
            if not self.env_state or 'rooms' not in self.env_state:
                return None
            
            for room in self.env_state['rooms']:
                if room.get('name') == room_name:
                    bounds = room.get('bounds', {})
                    if bounds:
                        center_x = (bounds.get('x_min', 0) + bounds.get('x_max', 0)) / 2
                        center_y = (bounds.get('y_min', 0) + bounds.get('y_max', 0)) / 2
                        center_z = (bounds.get('z_min', 0) + bounds.get('z_max', 0)) / 2
                        return {"x": center_x, "y": center_y, "z": center_z}
            return None
        except Exception:
            return None
    
    def _update_robot_position_in_file(self, position: Dict, room_name: str):
        """lab_env.json 파일의 로봇 위치 직접 업데이트"""
        try:
            import os
            import json
            
            base_path = os.path.dirname(os.path.abspath(__file__))
            env_file_path = os.path.join(base_path, "..", "OKB", "lab_env.json")
            
            with open(env_file_path, 'r', encoding='utf-8') as f:
                env_data = json.load(f)
            
            if 'agent' in env_data:
                env_data['agent']['position'] = position
                env_data['agent']['current_room'] = room_name
                
                with open(env_file_path, 'w', encoding='utf-8') as f:
                    json.dump(env_data, f, indent=2, ensure_ascii=False)
        except Exception:
            pass
    
    def _get_effective_robot_room(self) -> Optional[str]:
        """실제 로봇 위치 또는 시뮬레이션된 위치 반환"""
        if self.simulated_robot_room:
            return self.simulated_robot_room
        return self._get_robot_current_room()
    
    def _detect_multitask_context(self, parsed_command: Dict) -> bool:
        """멀티테스크 컨텍스트 감지"""
        # 입력에서 "and" 키워드가 있는지 확인
        input_text = parsed_command.get("input", "").lower()
        return " and " in input_text
    
    def load_enhanced_environment(self):
        """Enhanced 환경 데이터 로드"""
        try:
            self.name_to_class.clear()
            self.class_to_names.clear()
            self.name_to_room.clear()
            self.room_to_names.clear()
            
            with open(self.env_file_path, 'r', encoding='utf-8') as f:
                self.env_state = json.load(f)
            
            objects = self.env_state.get("objects", [])
            
            if isinstance(objects, list):
                for obj_data in objects:
                    name = obj_data.get("name", "")
                    if not name:
                        continue
                    
                    class_name = self._extract_class_from_name(name)
                    room = obj_data.get("current_room", "unknown")
                    
                    self.name_to_class[name] = class_name
                    self.class_to_names[class_name].append(name)
                    self.name_to_room[name] = room
                    self.room_to_names[room].append(name)
            
            rooms = self.env_state.get("rooms", [])
            for room_data in rooms:
                room_name = room_data.get("name", "")
                if room_name:
                    self.name_to_class[room_name] = "space"
                    self.class_to_names["space"].append(room_name)
                    self.name_to_room[room_name] = room_name
                    self.room_to_names[room_name].append(room_name)
        except Exception:
            self.env_state = {"rooms": [], "objects": []}
    
    def _ensure_environment_loaded(self):
        """환경 데이터 로드 확인"""
        self.load_enhanced_environment()
    
    def _extract_class_from_name(self, name: str) -> str:
        """Name에서 class 추출 (book_01 → book)"""
        if not name:
            return "unknown"
        
        if '_' in name:
            parts = name.split('_')
            if len(parts) >= 2 and parts[-1].isdigit():
                return '_'.join(parts[:-1])
        
        return name.lower()
    
    def find_objects_by_class_enhanced(self, class_name: str) -> List[str]:
        """클래스별 객체 이름 리스트"""
        self._ensure_environment_loaded()
        class_name = class_name.lower()
        
        room_names = ["lab", "classroom", "library", "hallway"]
        if class_name in room_names:
            if class_name in self.name_to_class:
                return [class_name]
        
        exact_matches = self.class_to_names.get(class_name, [])
        if exact_matches:
            return exact_matches.copy()
        
        if self.enable_fuzzy_matching:
            fuzzy_matches = []
            for class_key, names in self.class_to_names.items():
                if class_name not in room_names and (class_name in class_key or class_key in class_name):
                    fuzzy_matches.extend(names)
            
            if fuzzy_matches:
                return fuzzy_matches
        
        if self.enable_exhaustive_search:
            exhaustive_matches = []
            for name in self.name_to_class.keys():
                if class_name in name.lower():
                    exhaustive_matches.append(name)
            
            if exhaustive_matches:
                return exhaustive_matches
        
        return []
    
    def filter_objects_by_room_enhanced(self, candidates: List[str], target_room: str) -> List[str]:
        """Room 기반 객체 필터링"""
        self._ensure_environment_loaded()
        target_room = target_room.lower()
        room_filtered = []
        
        for name in candidates:
            obj_room = self.name_to_room.get(name, "").lower()
            if obj_room == target_room:
                room_filtered.append(name)
        
        if not room_filtered and self.enable_fuzzy_matching:
            for name in candidates:
                obj_room = self.name_to_room.get(name, "").lower()
                if target_room in obj_room or obj_room in target_room:
                    room_filtered.append(name)
        
        return room_filtered
    
    def interpret_spatial_info_enhanced(self, spatial_info: str) -> Optional[Dict]:
        """공간 정보 해석"""
        if not spatial_info:
            return None
        
        spatial_info = spatial_info.lower().strip()
        
        preposition_map = {
            "from": "from_room", "out of": "from_room", "away from": "from_room",
            "in": "in_room", "inside": "in_room", "within": "in_room", "into": "in_room",
            "at": "at_location", "by": "at_location", "near": "near_location",
            "on": "on_top_of", "on top of": "on_top_of", "onto": "on_top_of", "upon": "on_top_of",
            "under": "under", "below": "under", "beneath": "under",
            "above": "above", "over": "above",
            "next to": "next_to", "beside": "next_to", "adjacent to": "next_to",
            "to": "to_location", "towards": "to_location"
        }
        
        sorted_preps = sorted(preposition_map.items(), key=lambda x: len(x[0]), reverse=True)
        
        for prep, relation in sorted_preps:
            if prep in spatial_info:
                parts = spatial_info.split(prep, 1)
                if len(parts) > 1:
                    target = parts[1].strip()
                    result = {"relation": relation, "target": target}
                    return result
        
        return None
    
    def resolve_object_reference_enhanced(self, object_class: str, spatial_relation: Optional[Dict] = None, robot_position: Optional[Dict] = None, simulated_robot_room: Optional[str] = None) -> Optional[str]:
        """객체 참조 해결"""
        
        if "_" in object_class and object_class in self.name_to_class:
            return object_class
        
        if not robot_position:
            robot_position = self._get_robot_position()
        
        if simulated_robot_room:
            robot_current_room = simulated_robot_room
            
            if self.task_sequence:
                last_task = self.task_sequence[-1]
                last_target = last_task.get("target")
                if last_target:
                    last_target_position = self.get_object_position_enhanced(last_target)
                    if last_target_position:
                        robot_position = last_target_position
        else:
            robot_current_room = self._get_effective_robot_room()
        
        candidates = self.find_objects_by_class_enhanced(object_class)
        
        if not candidates and self.enable_multiple_fallbacks:
            if object_class.endswith('s'):
                singular = object_class[:-1]
                candidates = self.find_objects_by_class_enhanced(singular)
            else:
                plural = object_class + 's'
                candidates = self.find_objects_by_class_enhanced(plural)
        
        if not candidates and self.enable_multiple_fallbacks:
            partial_matches = []
            for name in self.name_to_class.keys():
                if object_class in name.lower() or name.lower() in object_class:
                    partial_matches.append(name)
            
            candidates = partial_matches
        
        if not candidates:
            return None
        
        if spatial_relation and "relation" in spatial_relation and "target" in spatial_relation:
            relation = spatial_relation["relation"]
            target_location = spatial_relation["target"]
            
            if relation in ["from_room", "in_room", "at_location", "near_location", "to_location"]:
                room_filtered = self.filter_objects_by_room_enhanced(candidates, target_location)
                if room_filtered:
                    candidates = room_filtered
            elif relation in ["on_top_of", "under", "next_to"]:
                pass
        
        if not candidates:
            return None
        
        if robot_current_room and len(candidates) > 1:
            same_room_candidates = self._filter_objects_by_room_enhanced(candidates, robot_current_room)
            if same_room_candidates:
                if robot_position:
                    closest_name = self._select_closest_object(same_room_candidates, robot_position)
                    if closest_name:
                        return closest_name
                selected = same_room_candidates[0]
                return selected
        
        if robot_position and len(candidates) > 1:
            closest_name = self._select_closest_object(candidates, robot_position)
            if closest_name:
                return closest_name
        
        selected = candidates[0]
        return selected
    
    def _get_robot_current_room(self) -> Optional[str]:
        """로봇의 현재 방 정보 반환"""
        try:
            if not self.env_state or 'agent' not in self.env_state:
                return None
            
            agent = self.env_state['agent']
            current_room = agent.get('current_room')
            if current_room:
                return current_room
            
            return None
        except Exception as e:
            return None

    def _get_robot_position(self) -> Optional[Dict]:
        """로봇의 현재 좌표 정보 반환"""
        try:
            if not self.env_state or 'agent' not in self.env_state:
                return None
            
            agent = self.env_state['agent']
            position = agent.get('position')
            if position:
                return position
            
            return None
        except Exception as e:
            return None

    def _filter_objects_by_room_enhanced(self, object_names: List[str], room_name: str) -> List[str]:
        """객체 이름 리스트에서 특정 방에 있는 객체들만 필터링"""
        try:
            room_objects = []
            for name in object_names:
                if name in self.name_to_room:
                    if self.name_to_room[name] == room_name:
                        room_objects.append(name)
            
            return room_objects
        except Exception as e:
            return object_names
    
    def _select_closest_object(self, candidates: List[str], robot_position: Dict) -> Optional[str]:
        """거리 기반 객체 선택"""
        closest_name = None
        min_distance = float('inf')
        
        for name in candidates:
            pos = self.get_object_position_enhanced(name)
            if pos:
                distance = ((pos["x"] - robot_position["x"])**2 + 
                          (pos["y"] - robot_position["y"])**2 + 
                          (pos["z"] - robot_position["z"])**2)**0.5
                
                if distance < min_distance:
                    min_distance = distance
                    closest_name = name
        
        return closest_name
    
    def get_object_position_enhanced(self, name: str) -> Optional[Dict]:
        """객체 위치 획득"""
        self._ensure_environment_loaded()
        if not self.env_state:
            return None
        
        objects = self.env_state.get("objects", [])
        
        if isinstance(objects, list):
            for obj_data in objects:
                if obj_data.get("name") == name:
                    return obj_data.get("position", {})
        
        rooms = self.env_state.get("rooms", [])
        for room in rooms:
            if room.get("name") == name:
                bounds = room.get("bounds", {})
                return {
                    "x": (bounds.get("x_min", 0) + bounds.get("x_max", 0)) / 2,
                    "y": (bounds.get("y_min", 0) + bounds.get("y_max", 0)) / 2,
                    "z": (bounds.get("z_min", 0) + bounds.get("z_max", 0)) / 2
                }
        
        return None
    
    def map_command_to_environment_enhanced(self, parsed_command: Dict, simulated_robot_room: Optional[str] = None) -> Optional[Dict]:
        """명령-환경 매핑"""
        self._ensure_environment_loaded()
        if not parsed_command or "grounded" not in parsed_command:
            return None
        
        start_time = time.time()
        
        grounded = parsed_command["grounded"]
        command = grounded.get("command")
        target = grounded.get("target")
        spatial_info = grounded.get("spatial_info")
        
        is_multitask = self._detect_multitask_context(parsed_command)
        
        result = {
            "original_input": parsed_command.get("input", ""),
            "command": command,
            "environment": {},
            "enhancement_stats": {}
        }
        
        robot_position = None
        if "robot_state" in self.env_state:
            robot_position = self.env_state["robot_state"].get("position")
        
        if target:
            spatial_relation = self.interpret_spatial_info_enhanced(spatial_info) if spatial_info else None
            concrete_target = self.resolve_object_reference_enhanced(
                target, spatial_relation, robot_position, simulated_robot_room
            )
            
            if concrete_target:
                target_class = self.name_to_class.get(concrete_target, "unknown")
                target_room = self.name_to_room.get(concrete_target, "unknown")
                
                result["environment"]["target"] = {
                    "abstract": target,
                    "concrete": concrete_target
                }
                result["environment"]["target_data"] = {
                    "position": self.get_object_position_enhanced(concrete_target),
                    "class": target_class,
                    "room": target_room
                }
                
                self.task_sequence.append({
                    "command": command,
                    "target": concrete_target,
                    "target_room": target_room,
                    "simulated_robot_room": self.simulated_robot_room
                })
        
        result["final_grounded"] = {
            "command": command,
            "target": result["environment"].get("target", {}).get("concrete")
        }
        
        processing_time = time.time() - start_time
        result["enhancement_stats"] = {
            "processing_time_ms": processing_time * 1000,
            "total_names": len(self.name_to_class),
            "total_rooms": len(self.room_to_names),
            "fuzzy_matching_enabled": self.enable_fuzzy_matching,
            "exhaustive_search_enabled": self.enable_exhaustive_search,
            "multiple_fallbacks_enabled": self.enable_multiple_fallbacks,
            "multitask_detected": is_multitask,
            "simulated_robot_room": self.simulated_robot_room
        }
        
        return result

def process_single_task_enhanced(task_data, simulated_robot_room: Optional[str] = None):
    """Enhanced 단일 작업 처리"""
    if not isinstance(task_data, dict):
        return None
        
    grounder = EnhancedSpatialGrounder()
    result = grounder.map_command_to_environment_enhanced(task_data, simulated_robot_room)
    
    if result:
        grounded = task_data.get("grounded", {})
        if isinstance(grounded, dict):
            result["plural_info"] = {
                "target_is_plural": grounded.get("target_is_plural", False)
            }
        else:
            result["plural_info"] = {
                "target_is_plural": False
            }
        return result
    return None
