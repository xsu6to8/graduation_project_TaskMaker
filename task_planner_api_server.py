"""
TaskMaker API 서버 - Unity와 통신하는 HTTP REST API
팀프로젝트용 네트워크 서버
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
import json
import traceback
from datetime import datetime

# TaskMaker 모듈 임포트
sys.path.append(os.path.join(os.path.dirname(__file__), 'Modules'))
from TaskMaker import TaskMakerPipeline

# Flask 앱 초기화
app = Flask(__name__)
CORS(app)  # Unity에서 접근할 수 있도록 CORS 허용

# TaskMaker 파이프라인 글로벌 초기화
pipeline = TaskMakerPipeline()

@app.route('/health', methods=['GET'])
def health_check():
    """서버 상태 확인 엔드포인트"""
    return jsonify({
        "status": "healthy",
        "message": "TaskMaker API Server is running",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/plan', methods=['POST'])
def generate_plan():
    try:
        # 요청 데이터 파싱
        if not request.is_json:
            return jsonify({
                "success": False,
                "error": "Content-Type must be application/json"
            }), 400
            
        data = request.get_json()
        user_input = data.get('user_input', '').strip()
        session_id = data.get('session_id', 'default')
        options = data.get('options', {})
        
        # 입력 검증
        if not user_input:
            return jsonify({
                "success": False,
                "error": "user_input is required and cannot be empty"
            }), 400
        
        print(f"\n[REQUEST] 세션: {session_id}")
        print(f"[INPUT] {user_input}")
        
        # TaskMaker 파이프라인 실행
        result = pipeline.process_natural_language_command(user_input)
        
        # 응답 데이터 구성
        response_data = {
            "success": True,
            "plan_sequence": result.plan_sequence or [],
            "metadata": {
                "processing_time": result.processing_time,
                "status": result.final_status,
                "steps_count": len(result.plan_sequence) if result.plan_sequence else 0,
                "original_input": result.original_input,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        # 디버그 정보 추가 (요청시)
        if options.get('include_debug', False):
            response_data["debug_info"] = {
                "parlex_result": result.parlex_result,
                "plan_result": result.plan_result,
                "spatial_result": result.spatial_result,
                "pipeline_metadata": result.pipeline_metadata
            }
            
        # 생성된 plan을 콘솔에 출력
        print(f"[STATUS] {result.final_status}")
        if result.plan_sequence:
            print(f"[PLAN] {len(result.plan_sequence)}개 스텝 생성:")
            for i, step in enumerate(result.plan_sequence, 1):
                print(f"  {i}. {step}")
        else:
            print("[PLAN] 생성된 plan 없음")
        print(f"[TIME] {result.processing_time:.3f}초\n")
            
        
        return jsonify(response_data)
        
    except Exception as e:
        # 에러 로깅
        error_msg = str(e)
        traceback.print_exc()
        
        print(f" 플래닝 실패: {error_msg}")
        
        return jsonify({
            "success": False,
            "error": error_msg,
            "error_type": type(e).__name__,
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/batch_plan', methods=['POST'])
def generate_batch_plans():
    try:
        data = request.get_json()
        batch_inputs = data.get('batch_inputs', [])
        session_id = data.get('session_id', 'batch')
        
        if not batch_inputs or not isinstance(batch_inputs, list):
            return jsonify({
                "success": False,
                "error": "batch_inputs must be a non-empty list"
            }), 400
        
        print(f"\n[BATCH REQUEST] 세션: {session_id}, {len(batch_inputs)}개 입력")
        
        results = []
        successful = 0
        
        for i, user_input in enumerate(batch_inputs):
            try:
                print(f"[BATCH {i+1}/{len(batch_inputs)}] {user_input}")
                result = pipeline.process_natural_language_command(user_input)
                
                # 생성된 plan을 콘솔에 출력
                if result.plan_sequence:
                    print(f"  ✓ {len(result.plan_sequence)}개 스텝 생성")
                else:
                    print(f"  ✗ Plan 생성 실패")
                
                results.append({
                    "index": i,
                    "input": user_input,
                    "plan_sequence": result.plan_sequence or [],
                    "status": result.final_status,
                    "processing_time": result.processing_time
                })
                
                if result.final_status == "SUCCESS":
                    successful += 1
                    
            except Exception as e:
                results.append({
                    "index": i,
                    "input": user_input,
                    "plan_sequence": [],
                    "status": "ERROR",
                    "error": str(e)
                })
        
        print(f"[BATCH COMPLETE] 성공: {successful}/{len(batch_inputs)}\n")
        
        return jsonify({
            "success": True,
            "results": results,
            "summary": {
                "total_requests": len(batch_inputs),
                "successful": successful,
                "failed": len(batch_inputs) - successful,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat()
            }
        })
        
    except Exception as e:
        print(f" 배치 플래닝 실패: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/update_environment', methods=['POST'])
def update_environment(): 
    try:
        
        # 요청 데이터 파싱
        if not request.is_json:
            return jsonify({
                "success": False,
                "error": "Content-Type must be application/json"
            }), 400
            
        # Unity에서 전송한 JSON 데이터를 그대로 받기
        environment_data = request.get_json()
        
        # 입력 검증 (기본적인 구조 확인)
        if not environment_data:
            return jsonify({
                "success": False,
                "error": "No data received"
            }), 400
        
        # 필수 필드 검증
        required_fields = ['rooms', 'objects', 'agent']
        for field in required_fields:
            if field not in environment_data:
                return jsonify({
                    "success": False,
                    "error": f"Missing required field: {field}"
                }), 400
        
        print(f"\n[ENV UPDATE] Unity에서 환경 데이터 수신")
        print(f"  - Rooms: {len(environment_data.get('rooms', []))}개")
        print(f"  - Objects: {len(environment_data.get('objects', []))}개")
        print(f"  - Relations: {len(environment_data.get('relation_information', []))}개")
        print(f"  - Agent Room: {environment_data.get('agent', {}).get('current_room', 'unknown')}")
        
        # 기존 파일 백업
        env_file_path = os.path.join(os.path.dirname(__file__), 'OKB', 'lab_env.json')
       
        # Unity에서 받은 JSON 데이터를 그대로 lab_env.json에 덮어쓰기
        with open(env_file_path, 'w', encoding='utf-8') as f:
            json.dump(environment_data, f, indent=4, ensure_ascii=False)
        
        # TaskMaker 파이프라인 환경 재로드
        if hasattr(pipeline, 'spatial_grounder') and hasattr(pipeline.spatial_grounder, 'load_enhanced_environment'):
            pipeline.spatial_grounder.load_enhanced_environment()
        
        # AmbiguityResolver 환경 재로드
        if hasattr(pipeline, 'ambiguity_resolver') and hasattr(pipeline.ambiguity_resolver, 'reload_environment'):
            pipeline.ambiguity_resolver.reload_environment()
        
        # 모든 캐시 무효화하여 새로운 환경 정보 반영
        if hasattr(pipeline, 'invalidate_all_caches'):
            pipeline.invalidate_all_caches()
        
        print(f"[ENV UPDATE] 파이프라인 환경 재로드 완료\n")
        
        return jsonify({
            "success": True,
            "message": "Environment updated successfully",
            "environment_stats": {
                "rooms_count": len(environment_data.get('rooms', [])),
                "objects_count": len(environment_data.get('objects', [])),
                "relations_count": len(environment_data.get('relation_information', [])),
                "agent_room": environment_data.get('agent', {}).get('current_room', 'unknown')
            },
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        # 에러 로깅
        error_msg = str(e)
        traceback.print_exc()
        
        print(f" 환경 업데이트 실패: {error_msg}")
        
        return jsonify({
            "success": False,
            "error": error_msg,
            "error_type": type(e).__name__,
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/get_environment', methods=['GET'])
def get_environment():
    try:
        env_file_path = os.path.join(os.path.dirname(__file__), 'OKB', 'lab_env.json')
        
        if not os.path.exists(env_file_path):
            return jsonify({
                "success": False,
                "error": "lab_env.json file not found"
            }), 404
        
        # 현재 환경 파일 읽기
        with open(env_file_path, 'r', encoding='utf-8') as f:
            environment_data = json.load(f)
        
        
        return jsonify({
            "success": True,
            "environment_data": environment_data,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        error_msg = str(e)
        print(f" 환경 데이터 전송 실패: {error_msg}")
        
        return jsonify({
            "success": False,
            "error": error_msg,
            "error_type": type(e).__name__,
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """시스템 통계 조회"""
    return jsonify({
        "pipeline_stats": pipeline.stats,
        "planner_stats": pipeline.custom_planner.get_planning_statistics() if hasattr(pipeline, 'custom_planner') else {},
        "timestamp": datetime.now().isoformat()
    })

if __name__ == '__main__':
    # 서버 설정
    HOST = '0.0.0.0'  # 모든 IP에서 접근 허용
    PORT = 5000       # 포트 번호 (Unity에서 사용)
    DEBUG = True      # 개발 모드
    
    print("=" * 60)
    print("TaskMaker API Server")
    print("=" * 60)
    print(f"Host: {HOST}")
    print(f"Port: {PORT}")
    print(f"Debug: {DEBUG}")
    print(f"Endpoints:")
    print(f"  - POST /plan          : 단일 명령 처리")
    print(f"  - POST /batch_plan    : 배치 명령 처리")
    print(f"  - POST /update_environment : 환경 업데이트")
    print(f"  - GET  /get_environment    : 환경 조회")
    print(f"  - GET  /health        : 서버 상태 확인")
    print(f"  - GET  /stats         : 통계 조회")
    print("=" * 60)
    print("\n서버 시작 중...\n")
    
    # Flask 서버 실행
    app.run(host=HOST, port=PORT, debug=DEBUG)
