"""
Robot ParLex

자연어 명령 파싱 및 OKB 기반 grounding
LLM fallback 지원
"""

import os
import json
import re
import time
from functools import lru_cache
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

from langchain_groq import ChatGroq

from langchain_core.messages import HumanMessage
from langchain_teddynote import logging
from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache

load_dotenv()
logging.langsmith("ParLex_robot_Grqoq_HYU")
set_llm_cache(SQLiteCache(database_path="robot_cache.db"))

chat = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile",
    temperature=0.2,
    max_tokens=500
)

class RobotParLex:
    PLURAL_INDICATORS = {
        'multiple', 'several', 'many', 'few', 'some', 'all', 'both', 'various',
        'two', 'three', 'four', '2', '3', '4', 'these', 'those', 'each', 'every'
    }
    PLURAL_SUFFIXES = ['s', 'es', 'ies', 'ves']
    
    COMMAND_SYNONYMS = {
        "move_to": "move", "travel": "move", "walk": "move", "go": "move",
        "pick_up": "pick", "grab": "pick", "take": "pick", "get": "pick",
        "place_on": "place", "put": "place", "set": "place"
    }
    
    def __init__(self, config_path: str = None):
        self._commands = []
        self._objects = []
        self._object_aliases = {}
        self._command_patterns = {}
        self._load_knowledge(config_path)
        self._build_mappings()
        
        self.commands = tuple(self._commands)
        self.objects = tuple(self._objects)
        
        self.planning_enabled = True
        self.plan_cache = {}
    
    @lru_cache(maxsize=128) 
    def _load_knowledge(self, config_path: str = None):
        if config_path is None:
            base_path = os.path.dirname(os.path.abspath(__file__))
            okb_path = os.path.join(base_path, "..", "OKB")
        else:
            okb_path = config_path
        
        with open(os.path.join(okb_path, "primitive_task.json"), 'r', encoding='utf-8') as f:
            self._commands = list(json.load(f).get("primitive_tasks", {}).keys())
        with open(os.path.join(okb_path, "lab_objects.json"), 'r', encoding='utf-8') as f:
            objects_data = json.load(f)
            self._objects = list(objects_data.get("objects", {}).keys())
            
            for obj_name, obj_info in objects_data.get("objects", {}).items():
                self._object_aliases[obj_name.lower()] = obj_name
                for alias in obj_info.get("aliases", []):
                    self._object_aliases[alias.lower()] = obj_name
            
            for room_name, room_info in objects_data.get("rooms", {}).items():
                self._object_aliases[room_name.lower()] = room_name
                for alias in room_info.get("aliases", []):
                    self._object_aliases[alias.lower()] = room_name
                self._objects.append(room_name)
    
    def _build_mappings(self):
        for cmd in self._commands:
            self._command_patterns[cmd] = cmd
            self._command_patterns[cmd.replace("_", " ")] = cmd
        self._command_patterns.update(self.COMMAND_SYNONYMS)
    
    def detect_plurality(self, text: str, target_word: str = None) -> bool:
        if not text or not target_word:
            return False
        
        text_lower = text.lower()
        target_lower = target_word.lower()
        
        if any(indicator in text_lower for indicator in self.PLURAL_INDICATORS):
            return True
        
        for suffix in self.PLURAL_SUFFIXES:
            if target_lower.endswith(suffix) and len(target_lower) > len(suffix):
                if suffix == 's':
                    if target_lower.endswith(('ies', 'ves', 'ses', 'zes', 'xes')):
                        return True
                    elif not target_lower.endswith(('ss', 'us', 'is')):
                        return True
                else:
                    return True
        
        words = text_lower.split()
        if target_lower in words:
            target_index = words.index(target_lower)
            if target_index > 0 and words[target_index - 1] in self.PLURAL_INDICATORS:
                return True
        
        return False
    
    def clean_input(self, text: str) -> List[str]:
        cleaned = re.sub(r'[.!?;,]', '', text.lower())
        words = cleaned.split()
        stop_words = {'the', 'a', 'an', 'please', 'can', 'you', 'would', 'could'} 
        return [word for word in words if word not in stop_words]
    
    def _find_command(self, words: List[str]) -> Optional[str]:
        if not words:
            return None
        
        if len(words) > 1:
            two_word = f"{words[0]} {words[1]}"
            if two_word in self._command_patterns:
                return self._command_patterns[two_word]
        
        if words[0] in self._command_patterns:
            return self._command_patterns[words[0]]
        
        return None
    
    def _find_objects(self, words: List[str]) -> List[str]:
        found_objects = []
        for word in words:
            if word.lower() in self._object_aliases:
                obj = self._object_aliases[word.lower()]
                if obj not in found_objects: 
                    found_objects.append(obj)
        return found_objects
    
    def _find_spatial_info(self, words: List[str]) -> Optional[str]:
        spatial_preps = ['on', 'in', 'at', 'near', 'under', 'above', 'beside', 'behind', 'by', 'from', 'to']
        spatial_clues = []
        
        i = 0
        while i < len(words):
            word = words[i]
            
            if word == "next" and i + 1 < len(words) and words[i + 1] == "to":
                if i + 2 < len(words):
                    target_word = words[i + 2].lower()
                    obj = self._object_aliases.get(target_word)
                    if obj:
                        spatial_clues.append(f"next to {obj}")
                    elif target_word in ['lab', 'library', 'classroom', 'hallway', 'office', 'room']:
                        spatial_clues.append(f"next to {target_word}")
                i += 3
                continue
            
            if word in spatial_preps and i + 1 < len(words):
                target_word = words[i + 1].lower()
                
                obj = self._object_aliases.get(target_word)
                if obj:
                    spatial_clues.append(f"{word} {obj}")
                elif target_word in ['lab', 'library', 'classroom', 'hallway', 'office', 'room']:
                    spatial_clues.append(f"{word} {target_word}")
                
                i += 2
                continue
            
            i += 1
        
        return ", ".join(spatial_clues) if spatial_clues else None
    
    def _parse_single_task(self, text: str) -> Dict:
        words = self.clean_input(text)
        
        result = {
            "command": None, 
            "target": None, 
            "target_is_plural": False,
            "spatial_info": None
        }
        
        result["command"] = self._find_command(words)
        result["spatial_info"] = self._find_spatial_info(words)
        objects = self._find_objects(words)
        
        if objects:
            result["target"] = objects[0]
            result["target_is_plural"] = self.detect_plurality(text, objects[0])
        
        return result

    def _split_multi_task(self, text: str) -> List[str]:
        if any(word in text.lower() for word in ["and", "then"]):
            parts = re.split(r'\s+and\s+|\s+then\s+', text.lower())
            resolved_parts = []
            last_object = None
            
            for part in parts:
                words = part.split()
                new_words = []
                
                for word in words:
                    if word == "it" and last_object:
                        new_words.append(last_object)
                    else:
                        new_words.append(word)
                        if word.lower() in self._object_aliases:
                            last_object = word
                
                resolved_parts.append(" ".join(new_words))
            
            return resolved_parts
        
        return [text]
    
    def _llm_fallback(self, user_input: str) -> Dict:
        prompt = f"""
You are a robot command parlex for ROBOT tasks.

USER INPUT: "{user_input}"
COMMANDS: {', '.join(self._commands)}
OBJECTS: {', '.join(self._objects)}

RULES:
- All commands need only target (no destination)
- Detect plurality (multiple, several, etc.)

OUTPUT JSON:
{{
  "tasks": [{{
    "command": "command_name",
    "target": "object/null",
    "target_is_plural": true/false,
    "spatial_info": null,
    "reasoning": "explanation"
  }}],
  "multi_task": false
}}

Parse: "{user_input}"
"""
        try:
            response = chat.invoke([HumanMessage(content=prompt)])
            response_text = response.content.strip()
            
            # JSON extraction
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            elif "{" in response_text:
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                json_text = response_text[json_start:json_end]
            else:
                json_text = response_text
            
            return json.loads(json_text)
        except Exception:
            return {
                "tasks": [{
                    "command": None, "target": None, "target_is_plural": False,
                    "spatial_info": None, "reasoning": "Emergency fallback"
                }],
                "multi_task": False
            }
    
    def _is_task_success(self, task_result: Dict) -> bool:
        command = task_result.get("command")
        target = task_result.get("target")
        
        if not command:
            return False
        
        return target is not None
    
    def process_input(self, user_input: str, include_planning: bool = True) -> Dict:
        start_time = time.time()
        
        tasks = self._split_multi_task(user_input)
        okb_results = []
        okb_success = True
        
        for i, task in enumerate(tasks):
            task_result = self._parse_single_task(task.strip())
            
            if self._is_task_success(task_result):
                okb_results.append({
                    "task_id": i + 1,
                    "input": task.strip(),
                    "grounded": task_result,
                    "metadata": {
                        "status": "success",
                        "method": "okb",
                        "processing_time": f"{time.time() - start_time:.2f}초"
                    }
                })
            else:
                okb_success = False
                break
        
        if okb_success and okb_results:
            base_result = {
                "original_input": user_input,
                "task_count": len(okb_results),
                "is_multi_task": len(okb_results) > 1,
                "tasks": okb_results
            }
        else:
            llm_result = self._llm_fallback(user_input)
            structured_tasks = []
            
            for i, task in enumerate(llm_result["tasks"]):
                task_input = user_input if i == 0 else f"Task {i + 1}"
                
                structured_tasks.append({
                    "task_id": i + 1,
                    "input": task_input,
                    "grounded": {
                        "command": task["command"],
                        "target": task["target"],
                        "target_is_plural": task.get("target_is_plural", False),
                        "spatial_info": task["spatial_info"]
                    },
                    "metadata": {
                        "status": "success" if task["command"] else "partial",
                        "method": "llm_fallback",
                        "reasoning": task["reasoning"],
                        "processing_time": f"{time.time() - start_time:.2f}초"
                    }
                })
            
            base_result = {
                "original_input": user_input,
                "task_count": len(structured_tasks),
                "is_multi_task": llm_result["multi_task"],
                "tasks": structured_tasks
            }
        
        if include_planning and self.planning_enabled and base_result.get("tasks"):
            planning_start = time.time()
            plan_result = self.generate_execution_plan(base_result)
            
            base_result["execution_plan"] = plan_result
            base_result["planning_time"] = f"{time.time() - planning_start:.2f}초"
            base_result["total_processing_time"] = f"{time.time() - start_time:.2f}초"
        
        return base_result
    
    def save_result(self, result: Dict):
        base_path = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(os.path.dirname(base_path), "result_TaskMaker_robot")
        os.makedirs(output_dir, exist_ok=True)
        
        with open(os.path.join(output_dir, "ParLex_robot_result.json"), 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    def generate_execution_plan(self, parsed_result: Dict) -> Dict:
        """Generate ProgPrompt-style execution plan from parsed command"""
        
        if not parsed_result or not parsed_result.get("tasks"):
            return {"error": "No valid parsed result for planning"}
            
        first_task = parsed_result["tasks"][0]
        grounded = first_task.get("grounded", {})
        
        plan_prompt = self._build_planning_prompt(grounded)
        
        try:
            response = chat.invoke([HumanMessage(content=plan_prompt)])
            plan_text = response.content.strip()
            
            parsed_plan = self._parse_plan_steps(plan_text)
            
            return {
                "task_input": first_task.get("input", ""),
                "grounded_command": grounded,
                "generated_plan": plan_text,
                "parsed_steps": parsed_plan,
                "planning_method": "progprompt_integrated"
            }
        except Exception as e:
            return {
                "error": str(e),
                "fallback": "Could not generate execution plan"
            }
            
    def _load_primitive_tasks(self) -> Dict[str, Any]:
        """Load primitive tasks from OKB"""
        try:
            base_path = os.path.dirname(os.path.abspath(__file__))
            okb_path = os.path.join(base_path, "..", "OKB")
            with open(os.path.join(okb_path, "primitive_task.json"), 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("primitive_tasks", {})
        except Exception:
            return {}
    
    def _build_planning_prompt(self, grounded: Dict) -> str:
        """Build ProgPrompt-style planning prompt using primitive tasks"""
        
        primitive_tasks = self._load_primitive_tasks()
        primitive_actions = list(primitive_tasks.keys())
        
        action_imports = f"""# PRIMITIVE TASKS ONLY - Use only: {primitive_actions}
# Available primitive tasks: {', '.join(primitive_actions)}"""
        
        objects_list = f"available_objects = {list(self._objects)}"
        
        examples = self._get_planning_examples()
        
        command = grounded.get("command", "")
        target = grounded.get("target", "")
        
        task_description = self._create_task_description(command, target)
        function_name = task_description.lower().replace(' ', '_').replace(',', '')
        
        prompt = f"""{action_imports}

{objects_list}

# Example execution plans:
{examples}

# Generate execution plan for: {task_description}
def {function_name}():
    # Write step-by-step execution plan
    # Use comments for high-level goals  
    # Include assertions for precondition checking
    # Add recovery actions when needed
"""
        
        return prompt
    
    def _get_planning_examples(self) -> str:
        """Get example plans using primitive tasks only"""
        
        primitive_tasks = self._load_primitive_tasks()
        primitive_actions = list(primitive_tasks.keys())
        
        examples = []
        
        if 'move' in primitive_actions:
            examples.append(f"""
def move_to_location():
    # Use primitive task: move
    move('target_location')
""")
        
        if 'pick' in primitive_actions:
            examples.append(f"""
def pick_object():
    # Use primitive task: pick
    pick('target_object')
""")
        
        if 'place' in primitive_actions:
            examples.append(f"""
def place_object():
    # Use primitive task: place
    place('target_object', 'destination_location')
""")
        
        return "\n".join(examples)
    
    def _create_task_description(self, command: str, target: str) -> str:
        """Create natural language task description"""
        
        if command == "move":
            return f"move to {target}" if target else "move to location"
        elif command == "watch":
            return f"watch {target}" if target else "watch area"
        elif command == "turn_on":
            return f"turn on {target}" if target else "turn on device"
        elif command == "turn_off":
            return f"turn off {target}" if target else "turn off device"
        elif command == "pick":
            return f"pick {target}" if target else "pick object"
        elif command == "place":
            return f"place {target}" if target else "place object"
        else:
            return f"execute {command} {target}" if target else f"execute {command}"
    
    def _parse_plan_steps(self, plan_text: str) -> List[Dict]:
        """Parse generated plan into structured steps"""
        
        lines = plan_text.split('\n')
        steps = []
        step_counter = 1
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('def '):
                continue
                
            line = re.sub(r'^[\s\t]+', '', line)
            
            if line.startswith('#'):
                steps.append({
                    "step_id": step_counter,
                    "type": "comment", 
                    "content": line,
                    "action": None,
                    "parameters": []
                })
            elif line.startswith('assert'):
                steps.append({
                    "step_id": step_counter,
                    "type": "assertion",
                    "content": line,
                    "action": "assert",
                    "parameters": [line]
                })
            else:
                action_match = re.match(r'(\w+)\((.*)\)', line)
                if action_match:
                    action_name = action_match.group(1)
                    params_str = action_match.group(2)
                    params = [p.strip().strip("'\"") for p in params_str.split(',') if p.strip()]
                    
                    steps.append({
                        "step_id": step_counter,
                        "type": "action",
                        "content": line,
                        "action": action_name,
                        "parameters": params
                    })
            
            step_counter += 1
            
        return steps
def create_parser(config_path: str = None) -> RobotParLex:
    return RobotParLex(config_path)

def main():
    parlex = create_parser()

    while True:
        user_input = input("\n(종료: quit, planning off: !noplan)\n> ").strip()
        
        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == '!noplan':
            parlex.planning_enabled = not parlex.planning_enabled
            continue
        elif user_input.lower() == '!cache':
            parlex._load_knowledge.cache_clear()
            continue
        elif not user_input:
            continue
        
        result = parlex.process_input(user_input, include_planning=parlex.planning_enabled)
        parlex.save_result(result)
        
        for task in result["tasks"]:
            grounded = task["grounded"]
            if grounded['target']:
                plural = " (plural)" if grounded.get('target_is_plural') else " (singular)"
        
        if "execution_plan" in result:
            plan = result["execution_plan"]
            if "generated_plan" in plan:
                pass
            
            if "parsed_steps" in plan:
                for i, step in enumerate(plan["parsed_steps"], 1):
                    pass
                    
            if "planning_time" in result:
                pass

if __name__ == "__main__":
    main()