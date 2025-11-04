"""
TaskMaker Integration: ParLex + Planner + SpatialGrounder

자연어 파싱 → 공간 그라운딩 → 계획 → 텍스트 시퀀스 생성
"""

import os
import json
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from dotenv import load_dotenv

load_dotenv()

from ParLex import RobotParLex
from AmbiguityResolver import AmbiguityResolver
from EnhancedSpatialGrounder import EnhancedSpatialGrounder, process_single_task_enhanced
from SpatialGrounder import process_single_task
from CustomTaskPlanner import CustomTaskPlanner

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

@dataclass
class TaskMakerResult:
    """TaskMaker 파이프라인 최종 결과"""
    original_input: str
    parlex_result: Dict[str, Any]
    plan_result: Dict[str, Any]
    spatial_result: Dict[str, Any]
    plan_sequence: List[str]
    final_status: str
    processing_time: float
    pipeline_metadata: Dict[str, Any]

class TaskMakerPipeline:
    """자연어 → 계획 시퀀스(텍스트) 파이프라인"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path
        
        self.parlex = RobotParLex(config_path)
        self.ambiguity_resolver = AmbiguityResolver()
        self.spatial_grounder = EnhancedSpatialGrounder()
        self.custom_planner = CustomTaskPlanner()
        
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        self.stats = {
            "total_processed": 0,
            "successful_parlex": 0,
            "successful_planning": 0,
            "successful_grounding": 0,
            "successful_execution": 0
        }
    
    def invalidate_all_caches(self):
        """모듈별 캐시 무효화"""
        if hasattr(self, 'spatial_grounder'):
            if hasattr(self.spatial_grounder, '_invalidate_caches'):
                self.spatial_grounder._invalidate_caches()
            elif hasattr(self.spatial_grounder, 'load_enhanced_environment'):
                self.spatial_grounder.load_enhanced_environment()
        
        if hasattr(self, 'custom_planner'):
            if hasattr(self.custom_planner, '_load_primitive_tasks') and hasattr(self.custom_planner._load_primitive_tasks, 'cache_clear'):
                self.custom_planner._load_primitive_tasks.cache_clear()
            if hasattr(self.custom_planner, '_load_composite_tasks') and hasattr(self.custom_planner._load_composite_tasks, 'cache_clear'):
                self.custom_planner._load_composite_tasks.cache_clear()
            if hasattr(self.custom_planner, '_load_objects') and hasattr(self.custom_planner._load_objects, 'cache_clear'):
                self.custom_planner._load_objects.cache_clear()
        
        if hasattr(self, 'parlex') and hasattr(self.parlex, '_load_knowledge') and hasattr(self.parlex._load_knowledge, 'cache_clear'):
            self.parlex._load_knowledge.cache_clear()
        
    def process_natural_language_command(self, user_input: str) -> TaskMakerResult:
        """NL 입력 → 파싱 → 그라운딩 → 계획 → 시퀀스 생성"""
        start_time = time.time()
        
        try:
            parlex_result = self.parlex.process_input(user_input)
            self.stats["successful_parlex"] += 1
            
            parlex_result = self.ambiguity_resolver.resolve(parlex_result)
            
            spatial_result = self._ground_plan_to_environment_enhanced(parlex_result)
            if spatial_result.get("processed_tasks"):
                self.stats["successful_grounding"] += 1
            
            plan_result = self._generate_plan_from_spatial_results(parlex_result, spatial_result)
            if plan_result.get("generated_plan"):
                self.stats["successful_planning"] += 1
            
            plan_sequence = self._generate_plan_sequence(plan_result, spatial_result)
            if plan_sequence:
                self.stats["successful_execution"] += 1
            
            final_status = self._determine_final_status(parlex_result, plan_result, spatial_result, plan_sequence)
            
            result = TaskMakerResult(
                original_input=user_input,
                parlex_result=parlex_result,
                plan_result=plan_result,
                spatial_result=spatial_result,
                plan_sequence=plan_sequence,
                final_status=final_status,
                processing_time=time.time() - start_time,
                pipeline_metadata={
                    "parlex_method": parlex_result.get("tasks", [{}])[0].get("metadata", {}).get("method", "unknown"),
                    "plan_steps_count": len(plan_result.get("parsed_steps", [])),
                    "grounded_objects_count": len(spatial_result.get("processed_tasks", [])),
                    "plan_sequence_length": len(plan_sequence)
                }
            )
            
            self.stats["total_processed"] += 1
            return result
        except Exception as e:
            error_result = TaskMakerResult(
                original_input=user_input,
                parlex_result={},
                plan_result={},
                spatial_result={},
                plan_sequence=[],
                final_status="ERROR",
                processing_time=time.time() - start_time,
                pipeline_metadata={"error": str(e)}
            )
            
            self.stats["total_processed"] += 1
            return error_result
            
    def _generate_plan_from_parlex(self, parlex_result: Dict[str, Any]) -> Dict[str, Any]:
        """Custom Task Planner를 사용해 계획 생성"""
        
        if not parlex_result.get("tasks"):
            return {"error": "No tasks from ParLex to plan"}
        
        try:
            execution_plan = self.custom_planner.generate_execution_plan(parlex_result)
            
            parsed_steps = []
            for step in execution_plan.steps:
                parsed_steps.append({
                    "step_number": step.step_id,
                    "action": step.action,
                    "target": step.target or "",
                    "destination": step.destination or "",
                    "description": step.description,
                    "conditions": step.conditions or [],
                    "original_line": f"{step.action}({step.target or ''}" + 
                                   (f", {step.destination}" if step.destination else "") + ")"
                })
            
            primitive_sequence = self.custom_planner.plan_to_primitive_sequence(execution_plan)
            
            return {
                "generated_plan": f"# {execution_plan.description}\n" + 
                                "\n".join([f"# {step.description}" for step in execution_plan.steps]) + 
                                "\n" + "\n".join(primitive_sequence),
                "parsed_steps": parsed_steps,
                "task_description": execution_plan.description,
                "function_name": execution_plan.plan_name,
                "parlex_grounded": parlex_result["tasks"][0].get("grounded", {}),
                "original_command": parlex_result["tasks"][0].get("input", ""),
                "success": True,
                "planning_method": "custom_planner",
                "plan_name": execution_plan.plan_name,
                "estimated_time": execution_plan.estimated_time,
                "prerequisites": execution_plan.prerequisites,
                "success_conditions": execution_plan.success_conditions
            }
        except Exception as e:
            return {"error": f"Custom planner failed: {str(e)}"}
    
    def _ground_plan_to_environment_enhanced(self, parlex_result: Dict[str, Any]) -> Dict[str, Any]:
        """계획 전 단계에서 향상된 공간 그라운딩"""
        
        if not parlex_result.get("tasks"):
            return {"error": "No ParLex tasks to ground"}
            
        processed_tasks = []
        for task in parlex_result["tasks"]:
            try:
                spatial_result = process_single_task_enhanced(task)
                if spatial_result:
                    processed_tasks.append(spatial_result)
            except Exception:
                spatial_result = process_single_task(task)
                if spatial_result:
                    processed_tasks.append(spatial_result)
        
        return {
            "processed_tasks": processed_tasks,
            "environment_state": getattr(self.spatial_grounder, 'env_state', {}),
            "grounding_method": "enhanced_spatial_grounder"
        }
    
    def _generate_plan_from_spatial_results(self, parlex_result: Dict[str, Any], spatial_result: Dict[str, Any]) -> Dict[str, Any]:
        """향상된 공간 그라운딩 결과 기반 계획 생성"""
        
        if not parlex_result.get("tasks"):
            return {"error": "No tasks from ParLex to plan"}
        
        if not spatial_result.get("processed_tasks"):
            return self._generate_plan_from_parlex(parlex_result)
        
        try:
            enhanced_parlex = self._merge_parlex_with_spatial(parlex_result, spatial_result)
            execution_plan = self.custom_planner.generate_execution_plan_enhanced(enhanced_parlex, spatial_result)
            
            parsed_steps = []
            for step in execution_plan.steps:
                parsed_steps.append({
                    "step_number": step.step_id,
                    "action": step.action,
                    "target": step.target or "",
                    "destination": step.destination or "",
                    "description": step.description,
                    "conditions": step.conditions or [],
                    "original_line": f"{step.action}({step.target or ''}" + 
                                   (f", {step.destination}" if step.destination else "") + ")"
                })
            
            primitive_sequence = self.custom_planner.plan_to_primitive_sequence(execution_plan)
            
            return {
                "generated_plan": f"# {execution_plan.description}\n" + 
                                "\n".join([f"# {step.description}" for step in execution_plan.steps]) + 
                                "\n" + "\n".join(primitive_sequence),
                "parsed_steps": parsed_steps,
                "task_description": execution_plan.description,
                "function_name": execution_plan.plan_name,
                "parlex_grounded": enhanced_parlex["tasks"][0].get("grounded", {}),
                "original_command": enhanced_parlex["tasks"][0].get("input", ""),
                "success": True,
                "planning_method": "enhanced_custom_planner",
                "plan_name": execution_plan.plan_name,
                "estimated_time": execution_plan.estimated_time,
                "prerequisites": execution_plan.prerequisites,
                "success_conditions": execution_plan.success_conditions,
                "spatial_enhanced": True
            }
        except Exception:
            return self._generate_plan_from_parlex(parlex_result)
    
    def _merge_parlex_with_spatial(self, parlex_result: Dict[str, Any], spatial_result: Dict[str, Any]) -> Dict[str, Any]:
        """ParLex 결과와 공간 그라운딩 결과 병합"""
        
        enhanced_parlex = parlex_result.copy()
        enhanced_tasks = []
        
        processed_tasks = spatial_result.get("processed_tasks", [])
        
        for i, task in enumerate(parlex_result["tasks"]):
            enhanced_task = task.copy()
            
            if i < len(processed_tasks):
                spatial_task = processed_tasks[i]
                final_grounded = spatial_task.get("final_grounded", {})
                environment = spatial_task.get("environment", {})
                
                enhanced_grounded = enhanced_task.get("grounded", {}).copy()
                enhanced_grounded.update({
                    "spatial_target": final_grounded.get("target"),
                    "spatial_destination": final_grounded.get("destination"),
                    "target_data": environment.get("target_data", {}),
                    "destination_data": environment.get("destination_data", {}),
                    "enhanced_grounding": True
                })
                
                enhanced_task["grounded"] = enhanced_grounded
                enhanced_task["spatial_enhanced"] = True
            
            enhanced_tasks.append(enhanced_task)
        
        enhanced_parlex["tasks"] = enhanced_tasks
        return enhanced_parlex
        
    def _ground_plan_to_environment(self, parlex_result: Dict[str, Any], plan_result: Dict[str, Any]) -> Dict[str, Any]:
        """기본 공간 그라운딩"""
        
        if not parlex_result.get("tasks"):
            return {"error": "No ParLex tasks to ground"}
            
        processed_tasks = []
        for task in parlex_result["tasks"]:
            try:
                spatial_result = self.spatial_grounder.map_command_to_environment(task)
                if spatial_result:
                    processed_tasks.append(spatial_result)
            except Exception:
                pass
        
        return {
            "processed_tasks": processed_tasks,
            "environment_state": getattr(self.spatial_grounder, 'env_state', {}),
            "grounding_method": "spatial_grounder"
        }
    
    def _generate_plan_sequence(self, plan_result: Dict[str, Any], spatial_result: Dict[str, Any]) -> List[str]:
        """Unity용 텍스트 명령 시퀀스 생성"""
        
        plan_sequence = []
        
        parsed_steps = plan_result.get("parsed_steps", [])
        if not parsed_steps:
            raw_plan = plan_result.get("generated_plan", "")
            if raw_plan:
                plan_sequence = self._extract_commands_from_raw_plan(raw_plan)
            return plan_sequence
        
        for step in parsed_steps:
            command = self._convert_step_to_text_command(step, spatial_result)
            if command:
                plan_sequence.append(command)
        
        return plan_sequence
    
    def _convert_step_to_text_command(self, step: Dict[str, Any], spatial_result: Dict[str, Any]) -> str:
        """파싱된 스텝을 실제 인스턴스 이름으로 텍스트 명령으로 변환"""
        
        action = step.get("action", "")
        target = step.get("target", "")
        destination = step.get("destination", "")
        
        real_target = self._get_real_instance_name(target, spatial_result) if target else ""
        real_destination = self._get_real_instance_name(destination, spatial_result) if destination else ""
        
        if action == "move" and real_target:
            return f"move({real_target})"
        elif action == "walk" and real_target:
            return f"walk_to({real_target})"
        elif action == "find" and real_target:
            return f"find({real_target})"
        elif action == "grab" and real_target:
            return f"grab({real_target})"
        elif action == "pick" and real_target:
            return f"pick({real_target})"
        elif action == "switchon" and real_target:
            return f"switchon({real_target})"
        elif action == "switchoff" and real_target:
            return f"switchoff({real_target})"
        elif action == "open" and real_target:
            return f"open({real_target})"
        elif action == "close" and real_target:
            return f"close({real_target})"
        elif action == "place" and real_target and real_destination:
            return f"place({real_target}, {real_destination})"
        elif action == "putback" and real_target and real_destination:
            return f"place({real_target}, {real_destination})"
        elif action == "putin" and real_target and real_destination:
            return f"put_in({real_target}, {real_destination})"
        elif action == "lookat" and real_target:
            return f"look_at({real_target})"
        else:
            if real_destination:
                return f"{action}({real_target}, {real_destination})"
            elif real_target:
                return f"{action}({real_target})"
            else:
                return f"{action}()"
    
    def _get_real_instance_name(self, abstract_name: str, spatial_result: Dict[str, Any]) -> str:
        """공간 그라운딩 결과를 사용해 추상명을 실제 인스턴스명으로 변환"""
        
        if not abstract_name:
            return ""
            
        processed_tasks = spatial_result.get("processed_tasks", [])
        for task in processed_tasks:
            env = task.get("environment", {})
            
            for mapping_type in ["target", "destination"]:
                mapping_info = env.get(mapping_type, {})
                if isinstance(mapping_info, dict) and mapping_info.get("abstract", "").lower() == abstract_name.lower():
                    concrete_id = mapping_info.get("concrete")
                    if concrete_id:
                        if isinstance(concrete_id, str) and ('_' in concrete_id or concrete_id.endswith('01')):
                            return concrete_id
                        else:
                            real_name = self._id_to_instance_name(concrete_id)
                            if real_name and not real_name.startswith("unknown_"):
                                return real_name
        
        if isinstance(abstract_name, str) and ('_' in abstract_name and abstract_name.split('_')[-1].isdigit()):
            return abstract_name
        
        return abstract_name
    
    def _id_to_instance_name(self, obj_id: int) -> str:
        """환경에서 ID를 인스턴스명으로 변환"""
        
        env_state = getattr(self.spatial_grounder, 'env_state', {})
        
        objects = env_state.get("objects", [])
        if isinstance(objects, list):
            for obj_data in objects:
                if obj_data.get("id") == obj_id:
                    return obj_data.get("name", f"object_{obj_id}")
        else:
            for obj_id_str, obj_data in objects.items():
                if int(obj_id_str) == obj_id:
                    return obj_data.get("name", f"object_{obj_id}")
        
        rooms = env_state.get("rooms", [])
        for room in rooms:
            if room.get("id") == obj_id:
                return room.get("name", f"room_{obj_id}")
        
        return f"unknown_{obj_id}"
    
    def _extract_commands_from_raw_plan(self, raw_plan: str) -> List[str]:
        """생성된 계획 텍스트에서 명령 추출"""
        
        commands = []
        lines = raw_plan.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('def'):
                continue
                
            import re
            patterns = [
                r"walk\(['\"]([^'\"]+)['\"]\)",
                r"find\(['\"]([^'\"]+)['\"]\)",
                r"grab\(['\"]([^'\"]+)['\"]\)",
                r"switchon\(['\"]([^'\"]+)['\"]\)",
                r"switchoff\(['\"]([^'\"]+)['\"]\)",
                r"open\(['\"]([^'\"]+)['\"]\)",
                r"close\(['\"]([^'\"]+)['\"]\)",
                r"lookat\(['\"]([^'\"]+)['\"]\)"
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, line)
                for match in matches:
                    action = pattern.split('\\(')[0].replace('\\', '')
                    commands.append(f"{action}({match})")
        
        return commands
    
    def _create_planning_context(self, grounded_info: Dict[str, Any]) -> Dict[str, Any]:
        """그라운딩 정보를 기반으로 계획 컨텍스트 구성"""
        
        env_state = getattr(self.spatial_grounder, 'env_state', {})
        robot_state = env_state.get("robot_state", {})
        robot_position = robot_state.get("position", {"x": 0, "y": 0, "z": 0})
        
        available_objects = list(env_state.get("objects", {}).keys())[:10]
        
        return {
            "robot_position": robot_position,
            "available_objects": available_objects,
            "target": grounded_info.get("target", ""),
            "destination": grounded_info.get("destination", ""),
            "command": grounded_info.get("command", "")
        }
        
    def _create_task_description(self, grounded_info: Dict[str, Any]) -> str:
        """플래너를 위한 자연어 작업 설명 생성"""
        
        command = grounded_info.get("command", "")
        target = grounded_info.get("target", "")
        destination = grounded_info.get("destination", "")
        
        if command == "move_to":
            return f"Move to {destination}"
        elif command == "watch":
            return f"Watch the {target} area"
        elif command == "watch":
            return f"Watch and monitor {target}"
        elif command == "turn_on":
            return f"Turn on the {target}"
        elif command == "turn_off":
            return f"Turn off the {target}"
        elif command == "give":
            return f"Give {target} to {destination}"
        elif command == "place_on":
            return f"Place {target} on {destination}"
        else:
            return f"Execute {command} command"
            
    def _parse_generated_plan(self, generated_plan: str) -> List[Dict[str, Any]]:
        """생성된 계획 텍스트를 구조화된 스텝으로 파싱"""
        
        steps = []
        
        clean_plan = generated_plan
        if "```python" in clean_plan:
            start_idx = clean_plan.find("```python") + 9
            end_idx = clean_plan.find("```", start_idx)
            if end_idx != -1:
                clean_plan = clean_plan[start_idx:end_idx].strip()
            else:
                clean_plan = clean_plan[start_idx:].strip()
        elif "```" in clean_plan:
            start_idx = clean_plan.find("```") + 3
            end_idx = clean_plan.find("```", start_idx)
            if end_idx != -1:
                clean_plan = clean_plan[start_idx:end_idx].strip()
            else:
                clean_plan = clean_plan[start_idx:].strip()
        
        lines = clean_plan.split('\n')
        
        for line in lines:
            line = line.strip()
            
            if line and not line.startswith('#') and not line.startswith('def'):
                import re
                action_match = re.match(r'(\w+)\((.*)\)', line)
                if action_match:
                    action_name = action_match.group(1)
                    params_str = action_match.group(2)
                    params = [p.strip().strip("'\"") for p in params_str.split(',') if p.strip()]
                    
                    target = params[0] if len(params) > 0 else ""
                    destination = params[1] if len(params) > 1 else ""
                    
                    steps.append({
                        "step_number": len(steps) + 1,
                        "action": action_name,
                        "target": target,
                        "destination": destination,
                        "original_line": line,
                        "parameters": params
                    })
        
        return steps
    
    def _extract_action_from_line(self, line: str) -> Optional[Dict[str, Any]]:
        """코드 한 줄에서 액션 정보 추출"""
        import re
        
        patterns = [
            (r"walk\(['\"]([^'\"]+)['\"]", "walk"),
            (r"find\(['\"]([^'\"]+)['\"]", "find"),
            (r"grab\(['\"]([^'\"]+)['\"]", "grab"),
            (r"switchon\(['\"]([^'\"]+)['\"]", "switchon"),
            (r"switchoff\(['\"]([^'\"]+)['\"]", "switchoff"),
            (r"open\(['\"]([^'\"]+)['\"]", "open"),
            (r"close\(['\"]([^'\"]+)['\"]", "close"),
            (r"putback\(['\"]([^'\"]+)['\"],\s*['\"]([^'\"]+)['\"]", "putback"),
            (r"putin\(['\"]([^'\"]+)['\"],\s*['\"]([^'\"]+)['\"]", "putin")
        ]
        
        for pattern, action in patterns:
            match = re.search(pattern, line)
            if match:
                if action in ["putback", "putin"]:
                    return {
                        "action": action,
                        "target": match.group(1),
                        "destination": match.group(2)
                    }
                else:
                    return {
                        "action": action,
                        "target": match.group(1)
                    }
        
        return None
    
    def _call_openai_api(self, prompt: str, max_tokens: int = 600, stop: List[str] = None, model: str = "gpt-4o-mini") -> tuple:
        """OpenAI API 호출로 계획 생성"""
        
        if not self.openai_api_key:
            return False, "OpenAI API key not set"
            
        try:
            client = openai.OpenAI(api_key=self.openai_api_key)
            
            primitive_tasks = self._load_primitive_tasks()
            primitive_actions = list(primitive_tasks.keys())
            
            system_message = f"""You are a robot task planner that generates Python code using ONLY primitive tasks.

STRICT RULES:
1. Use ONLY these primitive tasks: {primitive_actions}
2. Each line must be a single primitive task call
3. Do not create complex or compound actions
4. Break down complex tasks into simple primitive steps
5. Only use objects from the provided object list

Generate clean, executable Python code following these constraints."""

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                stop=stop,
                temperature=0.1
            )
            return True, response.choices[0].message.content.strip()
        except Exception as e:
            return False, f"OpenAI API error: {str(e)}"
    
    def set_openai_api_key(self, api_key: str):
        """OpenAI API 키 설정"""
        self.openai_api_key = api_key


def create_taskmaker_pipeline(config_path: str = None) -> TaskMakerPipeline:
    """TaskMaker 파이프라인 생성"""
    return TaskMakerPipeline(config_path)

def main():
    """간단 데모 실행"""
    
    pipeline = create_taskmaker_pipeline()
    
    test_commands = [
        "watch the hallway",
        "move to room1 and watch the tv",
        "turn on the lights in lobby",
        "go to room2 then watch there"
    ]
    
    for command in test_commands:
        result = pipeline.process_natural_language_command(command)
        filepath = pipeline.save_result(result)
    
    stats = pipeline.get_pipeline_statistics()
    for stage, rate in stats['success_rates'].items():
        pass

if __name__ == "__main__":
    main()
