"""
Custom Task Planner

위치 추적 기반 실행 계획 생성
Room 간 자동 navigation 및 상태 추적 지원
"""

import os
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from functools import lru_cache

@dataclass
class PlanStep:
    """개별 계획 단계"""
    step_id: int
    action: str
    target: Optional[str] = None
    destination: Optional[str] = None
    conditions: List[str] = None
    description: str = ""

@dataclass 
class ExecutionPlan:
    """전체 실행 계획"""
    plan_name: str
    description: str
    steps: List[PlanStep]
    estimated_time: int
    prerequisites: List[str] = None
    success_conditions: List[str] = None

@dataclass
class PositionTrackingNode:
    """위치 추적 노드"""
    task_index: int
    action: str
    target: str
    target_room: str
    instance_name: str

class CustomTaskPlanner:
    """순차적 위치 추적 기반 태스크 플래너"""
    
    def __init__(self, okb_path: str = None):
        if okb_path is None:
            base_path = os.path.dirname(os.path.abspath(__file__))
            self.okb_path = os.path.join(base_path, "..", "OKB")
        else:
            self.okb_path = okb_path
            
        self.primitive_tasks = self._load_primitive_tasks()
        self.composite_tasks = self._load_composite_tasks()
        self.objects = self._load_objects()
        self.environment = self._load_environment()
        self.planning_templates = self._load_planning_templates()
        self.room_navigation_routes = self._load_room_navigation_routes()
        self.stats = {
            "plans_generated": 0,
            "simple_plans_used": 0,
            "bubble_sort_tracking_used": 0,
            "room_navigation_insertions": 0
        }
        

    @lru_cache(maxsize=32)
    def _load_primitive_tasks(self) -> Dict[str, Any]:
        """Primitive tasks 로드"""
        try:
            with open(os.path.join(self.okb_path, "primitive_task.json"), 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("primitive_tasks", {})
        except Exception:
            return {}
            
    @lru_cache(maxsize=32)
    def _load_composite_tasks(self) -> Dict[str, Any]:
        """Composite tasks 로드"""
        try:
            with open(os.path.join(self.okb_path, "composite_task.json"), 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("composite_tasks", {})
        except Exception:
            return {}

    @lru_cache(maxsize=32) 
    def _load_objects(self) -> Dict[str, Any]:
        """Objects 정보 로드"""
        try:
            with open(os.path.join(self.okb_path, "lab_objects.json"), 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("objects", {})
        except Exception:
            return {}

    def _load_environment(self) -> Dict[str, Any]:
        """환경 정보 로드"""
        try:
            env_path = os.path.join(self.okb_path, "lab_env.json")
            with open(env_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                return data
        except (FileNotFoundError, json.JSONDecodeError):
            return {"rooms": [], "objects": [], "agent": {"current_room": "lab"}}

    def _load_planning_templates(self) -> Dict[str, Any]:
        """Planning templates 로드"""
        templates = {
            "navigation": {
                "pattern": ["move", "go", "walk", "travel"],
                "template": "move -> target location"
            },
            "manipulation": {
                "pattern": ["pick", "grab", "take", "place", "put"],
                "template": "move -> target -> action"
            },
            "device_control": {
                "pattern": ["turn_on", "turn_off", "switchon", "switchoff"],  
                "template": "move -> target -> action"
            },
            "surveillance": {
                "pattern": ["watch", "monitor", "check"],
                "template": "move -> target -> action"
            }
        }
        return templates

    def _load_room_navigation_routes(self) -> Dict[Tuple[str, str], List[Tuple[str, str]]]:
        """Room 간 이동 경로"""
        routes = {
            ("lab", "classroom"): [
                ("move", "labdoor_01"), ("open", "labdoor_01"), 
                ("move", "classroomdoor_01"), ("open", "classroomdoor_01"),
            ],
            ("lab", "library"): [
                ("move", "labdoor_01"), ("open", "labdoor_01"), 
                ("move", "librarydoor_01"), ("open", "librarydoor_01"),
            ],
            ("lab", "hallway"): [
                ("move", "labdoor_01"), ("open", "labdoor_01"),
            ],
            ("classroom", "lab"): [
                ("move", "classroomdoor_01"), ("open", "classroomdoor_01"),
                ("move", "labdoor_01"), ("open", "labdoor_01"),
            ],
            ("classroom", "library"): [
                ("move", "classroomdoor_01"), ("open", "classroomdoor_01"),
                ("move", "librarydoor_01"), ("open", "librarydoor_01"),
            ],
            ("classroom", "hallway"): [
                ("move", "classroomdoor_01"), ("open", "classroomdoor_01"),
            ],
            ("library", "lab"): [
                ("move", "librarydoor_01"), ("open", "librarydoor_01"),
                ("move", "labdoor_01"), ("open", "labdoor_01"),
            ],
            ("library", "classroom"): [
                ("move", "librarydoor_01"), ("open", "librarydoor_01"),
                ("move", "classroomdoor_01"), ("open", "classroomdoor_01"),
            ],
            ("library", "hallway"): [
                ("move", "librarydoor_01"), ("open", "librarydoor_01"),
            ],
            ("hallway", "lab"): [
                ("move", "labdoor_01"), ("open", "labdoor_01"),
            ],
            ("hallway", "classroom"): [
                ("move", "classroomdoor_01"), ("open", "classroomdoor_01"),
            ],
            ("hallway", "library"): [
                ("move", "librarydoor_01"), ("open", "librarydoor_01"),
            ]
        }
        return routes

    def _get_robot_current_room(self) -> str:
        """로봇의 현재 room 정보 반환"""
        try:
            self.environment = self._load_environment()
            
            if not self.environment:
                return "lab"
            
            if 'agent' in self.environment:
                agent = self.environment['agent']
                current_room = agent.get('current_room', 'lab')
                return current_room
            
            if 'objects' in self.environment:
                objects = self.environment['objects']
                if isinstance(objects, list):
                    for obj in objects:
                        if obj.get('name', '').lower() == 'robot' or 'robot' in obj.get('name', '').lower():
                            current_room = obj.get('current_room', 'lab')
                            return current_room
            
            return "lab"
        except Exception:
            return "lab"
    
    def _get_original_robot_room(self) -> str:
        """원본 로봇 위치 반환"""
        try:
            env_path = os.path.join(self.okb_path, "lab_env.json")
            if os.path.exists(env_path):
                with open(env_path, 'r', encoding='utf-8') as file:
                    env_data = json.load(file)
                
                if 'agent' in env_data:
                    agent = env_data['agent']
                    current_room = agent.get('current_room', 'lab')
                    return current_room
            
            return "lab"
        except Exception:
            return "lab"

    def _get_robot_position(self) -> Optional[Dict]:
        """로봇의 현재 좌표 정보 반환"""
        try:
            if not self.environment:
                return None
            
            if 'agent' in self.environment:
                agent = self.environment['agent']
                position = agent.get('position', {})
                if position:
                    return position
            
            if 'objects' in self.environment:
                objects = self.environment['objects']
                if isinstance(objects, list):
                    for obj in objects:
                        if obj.get('name', '').lower() == 'robot' or 'robot' in obj.get('name', '').lower():
                            position = obj.get('position', {})
                            if position:
                                return position
            
            return None
        except Exception:
            return None

    def _get_object_room_from_instance(self, instance_name: str) -> Optional[str]:
        """Instance name으로부터 room 정보 반환"""
        try:
            if not self.environment or 'objects' not in self.environment:
                return None
            
            objects = self.environment['objects']
            if isinstance(objects, list):
                for obj in objects:
                    if obj.get('name', '').lower() == instance_name.lower():
                        current_room = obj.get('current_room') or obj.get('room')
                        if current_room:
                            return current_room
                        break
            
            if instance_name.lower() in ["lab", "classroom", "library", "hallway"]:
                return instance_name.lower()
            
            relations = self.environment.get('relation_information', [])
            if relations:
                for rel in relations:
                    if not isinstance(rel, dict):
                        continue
                    
                    subject = rel.get('subject', '')
                    predicate = rel.get('predicate', '')
                    target = rel.get('target', '')
                    
                    if subject.lower() == instance_name.lower() and predicate in ['on', 'in']:
                        for obj in objects:
                            if obj.get('name', '').lower() == target.lower():
                                target_room = obj.get('current_room') or obj.get('room')
                                if target_room:
                                    return target_room
                
            return None
        except Exception:
            return None

    def _extract_grounded_tasks(self, parlex_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """ParLex 결과에서 grounded task들 추출"""
        tasks = []
        
        if not parlex_result.get("tasks"):
            return tasks
        
        for task in parlex_result["tasks"]:
            if task.get("grounded"):
                grounded = task["grounded"]
                command = grounded.get("command", "")
                target = grounded.get("target")
                
                if command and target:
                    task_info = {
                        "command": command,
                        "target": target,
                        "original_input": task.get("input", "")
                    }
                    
                    effective_room = grounded.get("effective_room")
                    if effective_room:
                        task_info["effective_room"] = effective_room
                    
                    tasks.append(task_info)
        
        return tasks

    def _generate_room_navigation_steps(self, from_room: str, to_room: str, start_step_id: int = 1) -> List[PlanStep]:
        """Room 간 이동 단계 생성"""
        if from_room == to_room:
            return []
        
        route_key = (from_room, to_room)
        if route_key not in self.room_navigation_routes:
            return []
        
        route = self.room_navigation_routes[route_key]
        steps = []
        
        for i, (action, target) in enumerate(route):
            step = PlanStep(
                step_id=start_step_id + i,
                action=action,
                target=target,
                description=f"Room navigation: {action}({target})"
            )
            steps.append(step)
        
        return steps

    def _apply_bubble_sort_position_tracking(self, grounded_tasks: List[Dict[str, Any]]) -> List[PlanStep]:
        """
        위치 추적 기반 실행 단계 생성
        버블 소트에서 착안
        """
        if not grounded_tasks:
            return []
        
        self.stats["bubble_sort_tracking_used"] = self.stats.get("bubble_sort_tracking_used", 0) + 1
        
        steps = []
        current_step_id = 1
        original_robot_room = self._get_original_robot_room()
        simulated_robot_room = original_robot_room
        
        for i, task in enumerate(grounded_tasks):
            command = task["command"]
            target = task["target"]
            
            if i == 0:
                current_robot_room_for_task = original_robot_room
            else:
                current_robot_room_for_task = simulated_robot_room
            
            effective_room = task.get("effective_room")
            if effective_room:
                target_room = effective_room
            else:
                target_room = self._get_object_room_from_instance(target)
            
            if not target_room:
                continue
            
            if current_robot_room_for_task != target_room:
                room_nav_steps = self._generate_room_navigation_steps(
                    current_robot_room_for_task, target_room, current_step_id
                )
                
                steps.extend(room_nav_steps)
                current_step_id += len(room_nav_steps)
                simulated_robot_room = target_room
                self.stats["room_navigation_insertions"] += 1
            
            if command != "move":
                move_step = PlanStep(
                    step_id=current_step_id,
                    action="move",
                    target=target,
                    description=f"{command} 수행을 위해 {target}으로 이동"
                )
                steps.append(move_step)
                current_step_id += 1
            
            action_step = PlanStep(
                step_id=current_step_id,
                action=command,
                target=target,
                description=f"{command} {target}"
            )
            steps.append(action_step)
            current_step_id += 1
            
            if command in ["pick", "place", "grab", "put"]:
                simulated_robot_room = target_room
        
        return steps

    def generate_execution_plan(self, parlex_result: Dict[str, Any]) -> ExecutionPlan:
        """ParLex 결과로부터 실행 계획 생성"""
        
        if not parlex_result.get("tasks"):
            raise ValueError("ParLex 결과에 태스크가 없습니다")
        
        self.stats["plans_generated"] += 1
        
        grounded_tasks = self._extract_grounded_tasks(parlex_result)
        
        if not grounded_tasks:
            return self._generate_fallback_plan_from_parlex(parlex_result)
        
        plan_steps = self._apply_bubble_sort_position_tracking(grounded_tasks)
        original_input = " and ".join([task["original_input"] for task in grounded_tasks])
        
        execution_plan = ExecutionPlan(
            plan_name="bubble_sort_position_tracking_plan",
            description=f"버블정렬 방식 위치 추적 계획: {original_input}",
            steps=plan_steps,
            estimated_time=self._estimate_execution_time(plan_steps),
            prerequisites=[],
            success_conditions=[]
        )
        
        return execution_plan

    def generate_execution_plan_enhanced(self, enhanced_parlex_result: Dict[str, Any], spatial_result: Dict[str, Any]) -> ExecutionPlan:
        """Enhanced 실행 계획 생성"""
        
        if not enhanced_parlex_result.get("tasks"):
            raise ValueError("Enhanced ParLex 결과에 태스크가 없습니다")
        
        self.stats["plans_generated"] += 1
        
        enhanced_tasks = self._extract_spatial_grounded_tasks(spatial_result, enhanced_parlex_result)
        
        if not enhanced_tasks:
            return self._generate_fallback_plan_from_parlex(enhanced_parlex_result)
        
        plan_steps = self._apply_bubble_sort_position_tracking(enhanced_tasks)
        original_input = enhanced_parlex_result.get("original_input", "")
        
        execution_plan = ExecutionPlan(
            plan_name="enhanced_bubble_sort_position_tracking_plan",
            description=f"Enhanced 버블정렬 방식 위치 추적 계획: {original_input}",
            steps=plan_steps,
            estimated_time=self._estimate_execution_time(plan_steps),
            prerequisites=[],
            success_conditions=[]
        )
        
        return execution_plan
    
    def _extract_spatial_grounded_tasks(self, spatial_result: Dict[str, Any], parlex_result: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Spatial Grounding 결과에서 실제 객체 정보 추출"""
        tasks = []
        
        if not spatial_result or not spatial_result.get("processed_tasks"):
            return tasks
        
        processed_tasks = spatial_result["processed_tasks"]
        
        for i, processed_task in enumerate(processed_tasks):
            env = processed_task.get("environment", {})
            final_grounded = processed_task.get("final_grounded", {})
            
            command = final_grounded.get("command", "")
            target = final_grounded.get("target")
            
            if command and target:
                task = {
                    "command": command,
                    "target": target,
                    "target_data": env.get("target_data", {}),
                    "original_input": processed_task.get("original_input", "")
                }
                
                if parlex_result and parlex_result.get("tasks"):
                    if i < len(parlex_result["tasks"]):
                        parlex_task = parlex_result["tasks"][i]
                        parlex_grounded = parlex_task.get("grounded", {})
                        effective_room = parlex_grounded.get("effective_room")
                        
                        if effective_room:
                            task["effective_room"] = effective_room
                
                tasks.append(task)
        
        return tasks
    
    def _generate_simple_execution_steps(self, tasks: List[Dict[str, Any]]) -> List[PlanStep]:
        """단순한 실행 단계 생성"""
        steps = []
        current_step_id = 1
        current_robot_room = self._get_robot_current_room()
        
        for i, task in enumerate(tasks):
            command = task["command"]
            target = task["target"]
            
            move_step = PlanStep(
                step_id=current_step_id,
                action="move",
                target=target,
                description=f"{command} 수행을 위해 {target}으로 이동"
            )
            steps.append(move_step)
            current_step_id += 1
            
            action_step = PlanStep(
                step_id=current_step_id,
                action=command,
                target=target,
                description=f"{command} {target}"
            )
            steps.append(action_step)
            current_step_id += 1
        
        return steps
    
    def _handle_simple_move_task_enhanced(self, task: Dict[str, Any]) -> ExecutionPlan:
        """Enhanced 단순 move 명령 처리"""
        grounded = task.get("grounded", {})
        target = grounded.get("target")
        
        step = PlanStep(
            step_id=1,
            action="move",
            target=target,
            description=f"{target}로 이동 (Enhanced)"
        )
        
        return ExecutionPlan(
            plan_name="enhanced_simple_move",
            description=f"Enhanced 단순 이동: {task.get('input', '')}",
            steps=[step],
            estimated_time=30,
            prerequisites=[],
            success_conditions=[]
        )

    def _handle_simple_move_task(self, task: Dict[str, Any]) -> ExecutionPlan:
        """단순 move 명령 처리"""
        grounded = task.get("grounded", {})
        target = grounded.get("target")
        
        step = PlanStep(
            step_id=1,
            action="move",
            target=target,
            description=f"{target}로 이동"
        )
        
        return ExecutionPlan(
            plan_name="simple_move",
            description=f"단순 이동: {task.get('input', '')}",
            steps=[step],
            estimated_time=30,
            prerequisites=[],
            success_conditions=[]
        )

    def _generate_fallback_plan_from_parlex(self, parlex_result: Dict[str, Any]) -> ExecutionPlan:
        """ParLex 기반 fallback 계획 생성"""
        first_task = parlex_result["tasks"][0]
        grounded = first_task.get("grounded", {})
        
        command = grounded.get("command", "")
        target = grounded.get("target", "")
        original_input = first_task.get("input", "")
        
        steps = []
        
        if target:
            steps.append(PlanStep(1, "move", target, None, [], f"{target}로 이동"))
            if command:
                steps.append(PlanStep(2, command, target, None, [], f"{command} 명령 수행"))
        
        if not steps:
            steps.append(PlanStep(1, "wait", None, None, [], "대기"))
        
        return ExecutionPlan(
            plan_name="fallback_plan",
            description=f"Fallback 계획: {original_input}",
            steps=steps,
            estimated_time=60,
            prerequisites=[],
            success_conditions=[]
        )

    def _estimate_execution_time(self, steps: List[PlanStep]) -> int:
        """실행 시간 추정"""
        time_estimates = {
            "move": 30,
            "pick": 15,
            "place": 15,
            "open": 5,
            "switchon": 5,
            "switchoff": 5,
            "watch": 45,
        }
        
        total_time = 0
        for step in steps:
            total_time += time_estimates.get(step.action, 20)
            
        return total_time

    def plan_to_primitive_sequence(self, plan: ExecutionPlan) -> List[str]:
        """ExecutionPlan을 primitive sequence로 변환"""
        sequence = []
        
        for step in plan.steps:
            if step.destination:
                sequence.append(f"{step.action}('{step.target}', '{step.destination}')")
            else:
                sequence.append(f"{step.action}('{step.target}')")
                
        return sequence

    def get_planning_statistics(self) -> Dict[str, Any]:
        """Planning 통계 반환"""
        total_plans = self.stats["plans_generated"]
        
        if total_plans == 0:
            return self.stats
            
        return {
            **self.stats,
            "bubble_sort_usage_rate": self.stats["bubble_sort_tracking_used"] / max(total_plans, 1),
            "room_navigation_avg": self.stats["room_navigation_insertions"] / max(self.stats["bubble_sort_tracking_used"], 1)
        }

def create_custom_planner(okb_path: str = None) -> CustomTaskPlanner:
    """Custom Task Planner 팩토리 함수"""
    return CustomTaskPlanner(okb_path)

if __name__ == "__main__":
    planner = create_custom_planner()
