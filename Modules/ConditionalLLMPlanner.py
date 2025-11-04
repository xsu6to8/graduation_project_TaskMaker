"""
Conditional LLM Planner

새로운 명령어 감지 시에만 활성화되는 LLM 기반 플래너
"""

import os
import json
import re
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    from langchain_groq import ChatGroq
    from langchain_core.messages import HumanMessage
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

@dataclass
class ConditionalLLMPlanStep:
    """LLM 플랜 스텝"""
    step_id: int
    action: str
    target: Optional[str] = None
    destination: Optional[str] = None
    description: str = ""

@dataclass
class ConditionalLLMExecutionPlan:
    """LLM 실행 플랜"""
    plan_name: str
    description: str
    steps: List[ConditionalLLMPlanStep]
    estimated_time: int
    prerequisites: List[str] = None
    success_conditions: List[str] = None
    trigger_reason: str = ""

class ConditionalLLMPlanner:
    """새로운 command 감지 시에만 동작하는 조건부 LLM 플래너"""
    
    def __init__(self, okb_path: str = None):
        if okb_path is None:
            base_path = os.path.dirname(os.path.abspath(__file__))
            self.okb_path = os.path.join(base_path, "..", "OKB")
        else:
            self.okb_path = okb_path
            
        # OKB 데이터 로드 (알려진 명령어 확인용)
        self.primitive_tasks = self._load_primitive_tasks()
        self.composite_tasks = self._load_composite_tasks()
        self.known_commands = self._extract_known_commands()
        
        # LLM 초기화
        self.llm_client = None
        if LLM_AVAILABLE:
            try:
                self.llm_client = ChatGroq(
                    api_key=os.getenv("GROQ_API_KEY"),
                    model="llama-3.3-70b-versatile",
                    temperature=0.2,
                    max_tokens=800
                )
            except Exception:
                self.llm_client = None
        
        # 통계
        self.stats = {
            "activation_count": 0,
            "new_commands_detected": 0,
            "successful_plans": 0,
            "failed_plans": 0
        }

    def _load_primitive_tasks(self) -> Dict[str, Any]:
        """Primitive tasks 로드"""
        try:
            with open(os.path.join(self.okb_path, "primitive_task.json"), 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("primitive_tasks", {})
        except Exception:
            return {}

    def _load_composite_tasks(self) -> Dict[str, Any]:
        """Composite tasks 로드"""
        try:
            with open(os.path.join(self.okb_path, "composite_task.json"), 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("composite_tasks", {})
        except Exception:
            return {}

    def _extract_known_commands(self) -> set:
        """OKB에서 알려진 모든 명령어 추출"""
        known_commands = set()
        
        # Primitive tasks에서 명령어 추출
        for cmd_name in self.primitive_tasks.keys():
            known_commands.add(cmd_name.lower())
            # 언더스코어를 공백으로 바꾼 버전도 추가
            known_commands.add(cmd_name.replace("_", " ").lower())
        
        # Composite tasks에서 명령어 추출
        for cmd_name in self.composite_tasks.keys():
            known_commands.add(cmd_name.lower())
            known_commands.add(cmd_name.replace("_", " ").lower())
        
        # 일반적인 동의어도 추가
        synonyms = {
            "move": ["go", "walk", "travel", "navigate"],
            "pick": ["grab", "take", "get", "pickup"],
            "place": ["put", "set", "drop", "place_on"],
            "switch_on": ["turn_on", "activate", "switchon", "turnon"],
            "switch_off": ["turn_off", "deactivate", "switchoff", "turnoff"],
            "open": ["open_door", "unlock"],
            "close": ["close_door", "shut"],
            "watch": ["monitor", "guard", "survey"],
        }
        
        for base_cmd, synonym_list in synonyms.items():
            if base_cmd in known_commands:
                known_commands.update([syn.lower() for syn in synonym_list])
        
        return known_commands

    def should_activate(self, parlex_result: Dict[str, Any]) -> tuple[bool, str]:
        """LLM 플래너 활성화 여부 판단"""
        if not parlex_result.get("tasks"):
            return False, "No tasks to analyze"
        
        activation_reasons = []
        
        for task in parlex_result["tasks"]:
            grounded = task.get("grounded", {})
            command = grounded.get("command")
            
            if not command:
                activation_reasons.append("Unknown/null command detected")
                continue
            
            command_lower = command.lower()
            
            # 1. 완전히 새로운 명령어인지 확인
            if command_lower not in self.known_commands:
                activation_reasons.append(f"New command detected: '{command}'")
                self.stats["new_commands_detected"] += 1
            
            # 2. 복합적인 명령어인지 확인 (여러 동작 조합)
            if self._is_complex_new_command(command_lower):
                activation_reasons.append(f"Complex new command pattern: '{command}'")
            
            # 3. OKB에 없는 객체나 위치가 포함된 경우
            if self._contains_unknown_objects(grounded):
                activation_reasons.append(f"Unknown objects in command: '{command}'")
        
        if activation_reasons:
            reason = "; ".join(activation_reasons)
            self.stats["activation_count"] += 1
            return True, reason
        
        return False, "All commands are known"

    def _is_complex_new_command(self, command: str) -> bool:
        """복합적이거나 새로운 패턴의 명령어인지 판단"""
        complex_patterns = [
            "bring.*to",  # "bring X to Y"
            "deliver.*from.*to",  # "deliver X from A to B"
            "collect.*and.*place",  # "collect X and place Y"
            "find.*then.*move",  # "find X then move Y"
            "prepare.*for",  # "prepare X for Y"
            "organize.*in",  # "organize X in Y"
            "clean.*area",  # "clean the area"
            "setup.*equipment",  # "setup equipment"
            "inspect.*and.*report",  # "inspect X and report"
        ]
        
        for pattern in complex_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                return True
        
        return False

    def _contains_unknown_objects(self, grounded: Dict[str, Any]) -> bool:
        """알려지지 않은 객체나 위치가 포함되어 있는지 확인"""
        target = grounded.get("target")
        destination = grounded.get("destination")
        
        if target and (not target or re.search(r'\d{3,}|serial|model', target.lower())):
            return True
            
        if destination and (not destination or re.search(r'\d{3,}|serial|model', destination.lower())):
            return True
            
        return False

    def generate_plan_for_new_command(self, parlex_result: Dict[str, Any], reason: str) -> Optional[ConditionalLLMExecutionPlan]:
        """새로운 명령어에 대한 실행 계획 생성"""
        
        if not self.llm_client:
            self.stats["failed_plans"] += 1
            return None
        
        try:
            # 첫 번째 task 분석
            first_task = parlex_result["tasks"][0]
            grounded = first_task.get("grounded", {})
            original_input = first_task.get("input", "")
            
            # LLM 프롬프트 구성
            prompt = self._build_new_command_prompt(grounded, original_input, reason)
            
            # LLM 호출
            response = self.llm_client.invoke([HumanMessage(content=prompt)])
            response_text = response.content.strip()
            
            # 응답 파싱
            execution_plan = self._parse_llm_response(response_text, original_input, reason)
            
            if execution_plan:
                self.stats["successful_plans"] += 1
                return execution_plan
            else:
                self.stats["failed_plans"] += 1
                return None
                
        except Exception:
            self.stats["failed_plans"] += 1
            return None

    def _build_new_command_prompt(self, grounded: Dict[str, Any], original_input: str, reason: str) -> str:
        """새로운 명령어를 위한 LLM 프롬프트 구성"""
        
        # 사용 가능한 primitive actions 목록
        available_actions = list(self.primitive_tasks.keys())
        
        command = grounded.get("command", "unknown")
        target = grounded.get("target", "")
        destination = grounded.get("destination", "")
        
        prompt = f"""
You are a robot task planner. Create an execution plan for a new command by composing existing primitive actions.

New command details:
- Original input: "{original_input}"
- Parsed command: "{command}"
- Target object: "{target}"
- Destination: "{destination}"
- Activation reason: {reason}

Available primitive actions:
{', '.join(available_actions)}

General robot behavior patterns:
1. Decompose complex tasks into multiple steps
2. If movement is required, use move first
3. Move to the location before manipulating an object
4. Each step should use only one primitive action

Decomposition examples:
- "bring laptop to classroom" → move(laptop), pick(laptop), move(classroom), place(laptop, desk)
- "clean the area" → move(area), pick(trash), move(trash_bin), place(trash, trash_bin)
- "setup equipment" → move(equipment), pick(equipment), move(target_location), place(equipment, target_location)

Response format (JSON):
{{
    "reasoning": "Explain the task decomposition logic",
    "steps": [
        {{"action": "move", "target": "object_name", "description": "description"}},
        {{"action": "pick", "target": "object_name", "description": "description"}},
        {{"action": "place", "target": "object_name", "destination": "location_name", "description": "description"}}
    ],
    "estimated_time": time_in_seconds,
    "prerequisites": ["prerequisite1", "prerequisite2"],
    "success_conditions": ["success_condition1", "success_condition2"]
}}

Generate an execution plan for the new command "{command}".
"""
        
        return prompt

    def _parse_llm_response(self, response_text: str, original_input: str, reason: str) -> Optional[ConditionalLLMExecutionPlan]:
        """LLM 응답을 실행 계획으로 파싱"""
        
        try:
            # JSON 추출
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if not json_match:
                return None
            
            plan_data = json.loads(json_match.group())
            
            if not plan_data.get("steps"):
                return None
            
            # ConditionalLLMPlanStep 객체들로 변환
            steps = []
            for i, step_data in enumerate(plan_data["steps"], 1):
                step = ConditionalLLMPlanStep(
                    step_id=i,
                    action=step_data.get("action", "wait"),
                    target=step_data.get("target"),
                    destination=step_data.get("destination"),
                    description=step_data.get("description", f"Step {i}")
                )
                steps.append(step)
            
            # 전체 실행 계획 생성
            execution_plan = ConditionalLLMExecutionPlan(
                plan_name=f"conditional_llm_plan_{int(time.time())}",
                description=f"새로운 명령어 플랜: {original_input}",
                steps=steps,
                estimated_time=plan_data.get("estimated_time", len(steps) * 30),
                prerequisites=plan_data.get("prerequisites", []),
                success_conditions=plan_data.get("success_conditions", []),
                trigger_reason=reason
            )
            
            return execution_plan
        except (json.JSONDecodeError, Exception):
            return None

    def plan_to_primitive_sequence(self, plan: ConditionalLLMExecutionPlan) -> List[str]:
        """실행 계획을 primitive sequence로 변환"""
        sequence = []
        
        for step in plan.steps:
            if step.destination:
                sequence.append(f"{step.action}('{step.target}', '{step.destination}')")
            else:
                sequence.append(f"{step.action}('{step.target}')")
                
        return sequence

    def get_statistics(self) -> Dict[str, Any]:
        """통계 정보 반환"""
        return {
            **self.stats,
            "known_commands_count": len(self.known_commands),
            "success_rate": self.stats["successful_plans"] / max(self.stats["activation_count"], 1)
        }

def create_conditional_llm_planner(okb_path: str = None) -> ConditionalLLMPlanner:
    """조건부 LLM 플래너 팩토리 함수"""
    return ConditionalLLMPlanner(okb_path)

if __name__ == "__main__":
    planner = create_conditional_llm_planner()
    
    test_parlex_result = {
        "tasks": [{
            "input": "bring coffee to meeting room",
            "grounded": {
                "command": "bring_coffee_to_meeting_room",
                "target": "coffee",
                "destination": "meeting_room"
            }
        }]
    }
    
    should_activate, reason = planner.should_activate(test_parlex_result)
    
    if should_activate:
        plan = planner.generate_plan_for_new_command(test_parlex_result, reason)
        if plan:
            sequence = planner.plan_to_primitive_sequence(plan)
    
