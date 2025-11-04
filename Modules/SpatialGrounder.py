"""
Robot SpatialGrounder

공간 관계 해석과 객체 그라운딩을 담당
"""

import os
import json
from typing import List, Optional

class SpatialRelation:
    relation_type: str
    target_object: str  

class RobotSpatialObjectGrounder:
    def __init__(self, env_file_path=None):
        if env_file_path is None:
            base_path = os.path.dirname(os.path.abspath(__file__))
            self.env_file_path = os.path.join(base_path, "..", "OKB", "lab_env.json")
        else:
            self.env_file_path = env_file_path

        self.env_state = None
        self.object_type_map = {}
        self.alias_map = {}
        
        self.proximity_threshold = 2.0
        self.vertical_threshold = 0.3
        self.alignment_threshold = 1.0
        
    def load_environment(self):
        """환경 데이터 로드"""
        self.object_type_map.clear()
        self.alias_map.clear()
        
        with open(self.env_file_path, 'r', encoding='utf-8') as f:
            self.env_state = json.load(f)
        self._build_reference_maps()
    
    def _ensure_environment_loaded(self):
        """환경 데이터 로드 확인"""
        self.load_environment()
    
    def _build_reference_maps(self):
        self.object_type_map.clear()
        self.alias_map.clear()
        
        objects = self.env_state.get("objects", [])
        
        if isinstance(objects, list):
            for obj_data in objects:
                obj_id = str(obj_data.get("id", ""))
                obj_class = obj_data.get("name", "").split("_")[0]
                
                if obj_class not in self.object_type_map:
                    self.object_type_map[obj_class] = []
                self.object_type_map[obj_class].append(obj_id)
                
                if obj_class not in self.alias_map:
                    self.alias_map[obj_class] = obj_id
        else:
            for obj_id, obj_data in objects.items():
                obj_class = obj_data.get("class", "unknown")
                
                if obj_class not in self.object_type_map:
                    self.object_type_map[obj_class] = []
                self.object_type_map[obj_class].append(obj_id)
                
                if obj_class not in self.alias_map:
                    self.alias_map[obj_class] = obj_id
        
        rooms = self.env_state.get("rooms", []) or self.env_state.get("environment", {}).get("rooms", [])
        for room_data in rooms:
            room_name = room_data["name"]
            if "space" not in self.object_type_map:
                self.object_type_map["space"] = []
            self.object_type_map["space"].append(room_name)
            
            if room_name not in self.alias_map:
                self.alias_map[room_name] = room_name
    
    def find_objects_by_type(self, object_type: str) -> List[str]:
        self._ensure_environment_loaded()
        if object_type in self.object_type_map:
            return self.object_type_map[object_type].copy()
        
        if object_type in self.alias_map:
            single_obj = self.alias_map[object_type]
            if single_obj in [room["name"] for room in self.env_state.get("environment", {}).get("rooms", [])]:
                return [single_obj]
            obj_class = self.env_state["objects"][single_obj]["class"]
            return self.object_type_map.get(obj_class, []).copy()
        
        matches = []
        for obj_class, obj_list in self.object_type_map.items():
            if object_type in obj_class or obj_class in object_type:
                matches.extend(obj_list)
        return matches
    
    def interpret_spatial_info(self, spatial_info: str) -> Optional[dict]:
        self._ensure_environment_loaded()
        if not spatial_info:
            return None
        
        spatial_info = spatial_info.lower().strip()
        
        preposition_map = {
            "on": "on_top_of","on top of": "on_top_of", "onto": "on_top_of", 
            "above": "above","over": "above",            
            "under": "under","below": "under","beneath": "under",
            "in": "inside", "inside": "inside","within": "inside",
            "near": "next_to", "next to": "next_to", "beside": "next_to", "adjacent to": "next_to","close to": "next_to","by": "next_to",
            "in front of": "in_front_of","front of": "in_front_of","before": "in_front_of",
            "behind": "behind","back of": "behind","after": "behind",
            "left of": "left_of","to the left of": "left_of",
            "right of": "right_of","to the right of": "right_of"
        }
        
        sorted_preps = sorted(preposition_map.items(), key=lambda x: len(x[0]), reverse=True)
        
        for prep, relation in sorted_preps:
            if prep in spatial_info:
                parts = spatial_info.split(prep, 1)  
                if len(parts) > 1:
                    target = parts[1].strip()
                    return {"relation": relation, "target": target}
        
        return None
    
    def _calculate_distance(self, obj1_id: str, obj2_id: str) -> float:
        pos1 = self.get_object_position(obj1_id)
        pos2 = self.get_object_position(obj2_id)
        
        if not pos1 or not pos2:
            return float('inf')
        
        return ((pos1.get("x", 0) - pos2.get("x", 0))**2 + 
                (pos1.get("y", 0) - pos2.get("y", 0))**2 + 
                (pos1.get("z", 0) - pos2.get("z", 0))**2)**0.5

    def _check_alignment(self, obj1_id: str, obj2_id: str, axis: str) -> bool:
        pos1 = self.get_object_position(obj1_id)
        pos2 = self.get_object_position(obj2_id)
        
        if not pos1 or not pos2:
            return False
        
        if axis == "x":
            return abs(pos1.get("x", 0) - pos2.get("x", 0)) < self.alignment_threshold
        elif axis == "y":
            return abs(pos1.get("y", 0) - pos2.get("y", 0)) < self.alignment_threshold
        elif axis == "z":
            return abs(pos1.get("z", 0) - pos2.get("z", 0)) < self.alignment_threshold
        return False

    def get_object_room(self, obj_id: str) -> Optional[str]:
        """객체가 포함된 방 이름 반환"""
        obj_data = self.env_state["objects"].get(obj_id, {})
        if "current_room" in obj_data:
            return obj_data["current_room"]
        
        contains_relations = self.env_state.get("spatial_relations", {}).get("contains", [])
        for relation in contains_relations:
            if relation.get("target") == obj_id:
                return relation.get("source")
        
        obj_pos = self.get_object_position(obj_id)
        if obj_pos:
            for room in self.env_state.get("environment", {}).get("rooms", []):
                bounds = room["bounds"]
                if (bounds["x_min"] <= obj_pos["x"] <= bounds["x_max"] and
                    bounds["y_min"] <= obj_pos["y"] <= bounds["y_max"] and
                    bounds["z_min"] <= obj_pos["z"] <= bounds["z_max"]):
                    return room["name"]
        
        return None

    def check_spatial_relation(self, obj1_id: str, obj2_id: str, relation: str) -> bool:
        obj1_data = self.env_state["objects"].get(obj1_id, {})
        obj2_data = self.env_state["objects"].get(obj2_id, {})
        
        if not obj1_data or not obj2_data:
            rooms = [room["name"] for room in self.env_state.get("environment", {}).get("rooms", [])]
            if obj2_id in rooms and relation == "inside":
                obj1_room = self.get_object_room(obj1_id)
                return obj1_room == obj2_id
            return False
        
        relations_list = self.env_state.get("spatial_relations", {}).get(relation, [])
        for rel in relations_list:
            if rel.get("source") == obj1_id and rel.get("target") == obj2_id:
                return True
        
        pos1 = self.get_object_position(obj1_id)
        pos2 = self.get_object_position(obj2_id)
        
        if not pos1 or not pos2:
            return False
        
        if relation == "on_top_of":
            on_surface = obj1_data.get("on_surface_of")
            if on_surface == obj2_id:
                return True
            height_diff = pos1.get("y", 0) - pos2.get("y", 0)
            x_align = self._check_alignment(obj1_id, obj2_id, "x")
            z_align = self._check_alignment(obj1_id, obj2_id, "z")
            return height_diff > self.vertical_threshold and x_align and z_align
        
        elif relation == "under":
            height_diff = pos2.get("y", 0) - pos1.get("y", 0)
            x_align = self._check_alignment(obj1_id, obj2_id, "x")
            z_align = self._check_alignment(obj1_id, obj2_id, "z")
            return height_diff > self.vertical_threshold and x_align and z_align
        
        elif relation == "above":
            height_diff = pos1.get("y", 0) - pos2.get("y", 0)
            x_align = self._check_alignment(obj1_id, obj2_id, "x")
            z_align = self._check_alignment(obj1_id, obj2_id, "z")
            return height_diff > 0 and x_align and z_align
        
        elif relation == "inside":
            in_container = obj1_data.get("in_container_of")
            if in_container == obj2_id:
                return True
            
            obj1_room = self.get_object_room(obj1_id)
            if obj1_room == obj2_id:
                return True
                
            obj2_bounds = obj2_data.get("bounding_box", {})
            if obj2_bounds:
                x_inside = obj2_bounds.get("x_min", 0) <= pos1.get("x", 0) <= obj2_bounds.get("x_max", 0)
                y_inside = obj2_bounds.get("y_min", 0) <= pos1.get("y", 0) <= obj2_bounds.get("y_max", 0)
                z_inside = obj2_bounds.get("z_min", 0) <= pos1.get("z", 0) <= obj2_bounds.get("z_max", 0)
                return x_inside and y_inside and z_inside
        
        elif relation == "next_to":
            distance = self._calculate_distance(obj1_id, obj2_id)
            return distance < self.proximity_threshold
        
        elif relation == "in_front_of":
            y_diff = pos1.get("y", 0) - pos2.get("y", 0)
            x_align = self._check_alignment(obj1_id, obj2_id, "x")
            z_align = self._check_alignment(obj1_id, obj2_id, "z")
            return y_diff > 0.2 and (x_align or z_align)
        
        elif relation == "behind":
            y_diff = pos2.get("y", 0) - pos1.get("y", 0)
            x_align = self._check_alignment(obj1_id, obj2_id, "x")
            z_align = self._check_alignment(obj1_id, obj2_id, "z")
            return y_diff > 0.2 and (x_align or z_align)
        
        elif relation == "left_of":
            x_diff = pos2.get("x", 0) - pos1.get("x", 0)
            y_align = self._check_alignment(obj1_id, obj2_id, "y")
            z_align = self._check_alignment(obj1_id, obj2_id, "z")
            return x_diff > 0.2 and (y_align or z_align)
        
        elif relation == "right_of":
            x_diff = pos1.get("x", 0) - pos2.get("x", 0)
            y_align = self._check_alignment(obj1_id, obj2_id, "y")
            z_align = self._check_alignment(obj1_id, obj2_id, "z")
            return x_diff > 0.2 and (y_align or z_align)
        
        return False
    
    def get_object_position(self, obj_id: str) -> Optional[dict]:
        self._ensure_environment_loaded()
        obj_data = self.env_state["objects"].get(obj_id, {})
        if obj_data:
            return obj_data.get("position")
        
        for room in self.env_state.get("environment", {}).get("rooms", []):
            if room["name"] == obj_id:
                bounds = room["bounds"]
                return {
                    "x": (bounds["x_min"] + bounds["x_max"]) / 2,
                    "y": (bounds["y_min"] + bounds["y_max"]) / 2,
                    "z": bounds["z_min"]
                }
        return None
    
    def resolve_object_reference(self, object_type, spatial_info=None, context=None, robot_position=None):
        spatial_relation = self.interpret_spatial_info(spatial_info) if spatial_info else None
        candidates = self.find_objects_by_type(object_type)
        
        if not candidates:
            return None
        
        if spatial_relation and "relation" in spatial_relation and "target" in spatial_relation:
            relation = spatial_relation["relation"]
            target_type = spatial_relation["target"]
            target_objects = self.find_objects_by_type(target_type)
            
            if target_objects:
                filtered = []
                for obj_id in candidates:
                    for target_obj_id in target_objects:
                        if self.check_spatial_relation(obj_id, target_obj_id, relation):
                            filtered.append(obj_id)
                            break
                
                if filtered:
                    candidates = filtered
                
                if relation == "next_to" and not filtered:
                    for obj_id in self.find_objects_by_type(object_type):
                        for target_obj_id in target_objects:
                            if (self.check_spatial_relation(obj_id, target_obj_id, "left_of") or
                                self.check_spatial_relation(obj_id, target_obj_id, "right_of")):
                                filtered.append(obj_id)
                                break
                    if filtered:
                        candidates = filtered
        
        if candidates:
            if robot_position:
                closest_obj = None
                min_dist = float('inf')
                for obj_id in candidates:
                    pos = self.get_object_position(obj_id)
                    if pos:
                        dist = ((pos["x"] - robot_position["x"])**2 + 
                               (pos["y"] - robot_position["y"])**2 + 
                               (pos["z"] - robot_position["z"])**2)**0.5
                        if dist < min_dist:
                            min_dist = dist
                            closest_obj = obj_id
                if closest_obj:
                    return closest_obj
            
            return candidates[0]
        
        return None
    
    def map_command_to_environment(self, parsed_command):
        self._ensure_environment_loaded()
        if not parsed_command or "grounded" not in parsed_command:
            return None
        
        grounded = parsed_command["grounded"]
        command = grounded.get("command")
        target = grounded.get("target")
        spatial_info = grounded.get("spatial_info")
        
        result = {
            "original_input": parsed_command.get("input", ""),
            "command": command,
            "environment": {}
        }
        
        robot_position = None
        if "robot_state" in self.env_state:
            robot_position = self.env_state["robot_state"].get("position")
        
        if target:
            concrete_target = self.resolve_object_reference(target, spatial_info=spatial_info, robot_position=robot_position)
            
            if concrete_target:
                target_data = self.env_state["objects"].get(concrete_target, {})
                if not target_data:
                    for room in self.env_state.get("environment", {}).get("rooms", []):
                        if room["name"] == concrete_target:
                            target_data = {"class": "space", "current_room": concrete_target}
                            break
                
                result["environment"]["target"] = {"abstract": target, "concrete": concrete_target}
                result["environment"]["target_data"] = {
                    "position": self.get_object_position(concrete_target),
                    "class": target_data.get("class", "unknown")
                }
        
        result["final_grounded"] = {
            "command": command,
            "target": result["environment"].get("target", {}).get("concrete")
        }
        
        return result

class SpatialGrounder:
    def __init__(self):
        self.objects = []
        self.object_aliases = {}
        self.spatial_patterns = {}
        self.grounder = RobotSpatialObjectGrounder()
        self.load_knowledge()
        
    def load_knowledge(self):
        base_path = os.path.dirname(os.path.abspath(__file__))
        okb_path = os.path.join(base_path, "..", "OKB")
        objects_file = os.path.join(okb_path, "lab_objects.json")
        
        with open(objects_file, 'r', encoding='utf-8') as f:
            objects_data = json.load(f)
            self.objects = list(objects_data.get("objects", {}).keys())
            
            for obj_name, obj_info in objects_data.get("objects", {}).items():
                self.object_aliases[obj_name.lower()] = obj_name
                for alias in obj_info.get("aliases", []):
                    self.object_aliases[alias.lower()] = obj_name
    
    def extract_spatial_relations(self, text: str) -> List[SpatialRelation]:
        text_lower = text.lower()
        words = text_lower.split()
        relations = []
        
        i = 0
        while i < len(words):
            found_relation = False

            complex_preps = [
                ("in front of", "in_front_of"),
                ("to the left of", "left_of"),
                ("to the right of", "right_of"),
                ("on top of", "on"),
                ("next to", "next_to"),
                ("close to", "near"),
                ("back of", "behind")
            ]

            for prep_phrase, relation_type in complex_preps:
                phrase_words = prep_phrase.split()
                if i + len(phrase_words) <= len(words):
                    if words[i:i+len(phrase_words)] == phrase_words:
                        target_pos = i + len(phrase_words)
                        target_obj = self._find_object_at_position(words, target_pos)
                        if target_obj:
                            relations.append(SpatialRelation(relation_type=relation_type, target_object=target_obj))
                        i += len(phrase_words) + 1 
                        found_relation = True
                        break
            
            if found_relation:
                continue
            
            for relation_type, pattern_info in self.spatial_patterns.items():
                for keyword in pattern_info["keywords"]:
                    if keyword == words[i]: 
                        obj_pos = i + 1
                        target_obj = self._find_object_at_position(words, obj_pos)
                        if target_obj:
                            relations.append(SpatialRelation(relation_type=relation_type, target_object=target_obj))
                        found_relation = True
                        break
                
                if found_relation:
                    i += 2 
                    break
            
            if not found_relation:
                i += 1
        
        return relations

    def _find_object_at_position(self, words: List[str], pos: int) -> Optional[str]:
        if pos >= len(words):
            return None
        
        word = words[pos]
        if word in self.object_aliases:
            return self.object_aliases[word]
        
        for length in range(min(4, len(words) - pos), 1, -1):
            compound = " ".join(words[pos:pos + length])
            if compound in self.object_aliases:
                return self.object_aliases[compound]
        
        return None
    
    def process_parlex_result(self, parsed_command):
        return self.grounder.map_command_to_environment(parsed_command)

def load_parlex_result(file_path=None):
    if file_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        robot_dir = os.path.dirname(current_dir)
        file_path = os.path.join(robot_dir, "result_TaskMaker_robot", "ParLex_robot_result.json")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def process_single_task(task_data):
    if not isinstance(task_data, dict):
        return None
        
    grounder = RobotSpatialObjectGrounder()
    result = grounder.map_command_to_environment(task_data)
    
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

def process_parlex_file(file_path=None):
    parlex_data = load_parlex_result(file_path)
    if not parlex_data:
        return None
    
    tasks = parlex_data.get("tasks", [])
    if not tasks:
        return None
    
    processed_results = []
    for task in tasks:
        result = process_single_task(task)
        if result:
            processed_results.append(result)
    
    if processed_results:
        final_result = {
            "original_input": parlex_data.get("original_input"),
            "task_count": len(processed_results),
            "is_multi_task": len(processed_results) > 1,
            "processed_tasks": processed_results,
        }
        save_sg_result(final_result)
        return final_result
    return None

def save_sg_result(result, file_path=None):
    if file_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        robot_dir = os.path.dirname(current_dir)
        file_path = os.path.join(robot_dir, "result_TaskMaker_robot", "SpatialGround_robot_Result.json")
    
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

def main():
    result = process_parlex_file()
    if result:
        pass
    else:
        pass

if __name__ == "__main__":
    main()