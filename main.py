import os
import time
import json
import threading
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Any
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from abc import ABC, abstractmethod

# LLM Provider imports
import openai
import anthropic
import requests
import groq

app = Flask(__name__)
CORS(app)

# Abstract base class for LLM providers
class LLMProvider(ABC):
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    @abstractmethod
    def generate_response(self, prompt: str, conversation_history: List[Dict] = None) -> str:
        pass
    
    @abstractmethod
    def get_available_models(self) -> List[str]:
        pass

# OpenAI Provider
class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.client = openai.OpenAI(api_key=api_key)
        self.models = [
            "gpt-4.1", "gpt-4o", "gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1-nano",
            "o4-mini", "o3", "o3-pro", "o3-mini", "o1", "o1-mini", "o1-pro",
            "chatgpt-4o-latest"
        ]
    
    def generate_response(self, prompt: str, conversation_history: List[Dict] = None) -> str:
        try:
            messages = []
            
            # Add system message
            messages.append({"role": "system", "content": prompt})
            
            # Add conversation history
            if conversation_history:
                for msg in conversation_history[-10:]:  # Last 10 messages for context
                    role = "assistant" if msg["speaker"] == "seller" else "user"
                    messages.append({"role": role, "content": msg["message"]})
            
            response = self.client.chat.completions.create(
                model="gpt-4o",  # Default model
                messages=messages,
                temperature=0.8,
                max_tokens=2000
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating OpenAI response: {str(e)}"
    
    def get_available_models(self) -> List[str]:
        return self.models

# Anthropic Provider
class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.client = anthropic.Anthropic(api_key=api_key)
        self.models = [
            "claude-opus-4-20250514", "claude-sonnet-4-20250514", 
            "claude-3-7-sonnet-20250219", "claude-3-5-haiku-20241022"
        ]
    
    def generate_response(self, prompt: str, conversation_history: List[Dict] = None) -> str:
        try:
            messages = []
            
            # Add conversation history
            if conversation_history:
                for msg in conversation_history[-10:]:  # Last 10 messages for context
                    role = "assistant" if msg["speaker"] == "seller" else "user"
                    messages.append({
                        "role": role,
                        "content": [{"type": "text", "text": msg["message"]}]
                    })
            
            # Add current context as user message if no history
            if not messages:
                messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": "Start the conversation."}]
                })
            
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",  # Default model
                max_tokens=2000,
                temperature=0.8,
                system=prompt,
                messages=messages
            )
            
            return response.content[0].text.strip()
        except Exception as e:
            return f"Error generating Anthropic response: {str(e)}"
    
    def get_available_models(self) -> List[str]:
        return self.models

# xAI Provider
class XAIProvider(LLMProvider):
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.api_url = "https://api.x.ai/v1/chat/completions"
        self.models = ["grok-4-0709", "grok-3", "grok-3-mini", "grok-3-fast"]
    
    def generate_response(self, prompt: str, conversation_history: List[Dict] = None) -> str:
        try:
            messages = []
            
            # Add system message
            messages.append({"role": "system", "content": prompt})
            
            # Add conversation history
            if conversation_history:
                for msg in conversation_history[-10:]:  # Last 10 messages for context
                    role = "assistant" if msg["speaker"] == "seller" else "user"
                    messages.append({"role": role, "content": msg["message"]})
            
            payload = {
                "model": "grok-4-0709",  # Default model
                "messages": messages,
                "temperature": 0.8,
                "max_tokens": 2000,
                "stream": False
            }
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            response = requests.post(self.api_url, json=payload, headers=headers)
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
        except Exception as e:
            return f"Error generating xAI response: {str(e)}"
    
    def get_available_models(self) -> List[str]:
        return self.models

# Groq Provider
class GroqProvider(LLMProvider):
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.client = groq.Groq(api_key=api_key)
        self.models = [
            "llama-3.3-70b-versatile", "llama-3.1-8b-instant", "gemma2-9b-it",
            "deepseek-r1-distill-llama-70b", "meta-llama/llama-4-maverick-17b-128e-instruct"
        ]
    
    def generate_response(self, prompt: str, conversation_history: List[Dict] = None) -> str:
        try:
            messages = []
            
            # Add system message
            messages.append({"role": "system", "content": prompt})
            
            # Add conversation history
            if conversation_history:
                for msg in conversation_history[-10:]:  # Last 10 messages for context
                    role = "assistant" if msg["speaker"] == "seller" else "user"
                    messages.append({"role": role, "content": msg["message"]})
            
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",  # Default model
                messages=messages,
                temperature=0.8,
                max_tokens=2000
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating Groq response: {str(e)}"
    
    def get_available_models(self) -> List[str]:
        return self.models

# Competition Run Management
class CompetitionRun:
    def __init__(self, run_id: str, agent1_config: Dict, agent2_config: Dict, api_keys: Dict):
        self.id = run_id
        self.agent1_config = agent1_config
        self.agent2_config = agent2_config
        self.api_keys = api_keys
        self.messages = []
        self.status = "initializing"
        self.start_time = datetime.now()
        self.current_turn = "agent1"
        self.round_count = 0
        self.max_rounds = 20  # Configurable limit
        
        # Initialize LLM providers
        self.agent1_provider = self._create_provider(agent1_config["llm"], api_keys)
        self.agent2_provider = self._create_provider(agent2_config["llm"], api_keys)
        
        # Start the competition
        self._start_competition()
    
    def _create_provider(self, provider_name: str, api_keys: Dict) -> Optional[LLMProvider]:
        try:
            if provider_name == "openai" and "openai" in api_keys:
                return OpenAIProvider(api_keys["openai"])
            elif provider_name == "anthropic" and "anthropic" in api_keys:
                return AnthropicProvider(api_keys["anthropic"])
            elif provider_name == "xai" and "xai" in api_keys:
                return XAIProvider(api_keys["xai"])
            elif provider_name == "groq" and "groq" in api_keys:
                return GroqProvider(api_keys["groq"])
            else:
                return None
        except Exception as e:
            print(f"Error creating {provider_name} provider: {e}")
            return None
    
    def _start_competition(self):
        if not self.agent1_provider or not self.agent2_provider:
            self.status = "error"
            self.add_message("system", "Error: Could not initialize LLM providers. Check API keys.")
            return
        
        self.status = "running"
        # Start with agent1 (seller)
        threading.Thread(target=self._generate_next_message, daemon=True).start()
    
    def _generate_next_message(self):
        if self.status != "running" or self.round_count >= self.max_rounds:
            return
        
        try:
            # Determine which agent should respond
            if self.current_turn == "agent1":
                provider = self.agent1_provider
                prompt = self.agent1_config["prompt"]
                speaker = "agent1"
            else:
                provider = self.agent2_provider
                prompt = self.agent2_config["prompt"]
                speaker = "agent2"
            
            # Generate response
            response = provider.generate_response(prompt, self.messages)
            
            # Add message
            self.add_message(speaker, response)
            
            # Switch turns
            self.current_turn = "agent2" if self.current_turn == "agent1" else "agent1"
            
            # Schedule next message after delay
            if self.status == "running" and self.round_count < self.max_rounds:
                threading.Timer(3.0, self._generate_next_message).start()
            else:
                self.status = "completed"
                
        except Exception as e:
            self.add_message("system", f"Error: {str(e)}")
            self.status = "error"
    
    def add_message(self, speaker: str, content: str):
        message = {
            "speaker": speaker,
            "message": content,
            "timestamp": datetime.now(),
            "round": self.round_count + 1
        }
        self.messages.append(message)
        
        if speaker in ["agent1", "agent2"]:
            self.round_count = len([m for m in self.messages if m["speaker"] in ["agent1", "agent2"]]) // 2
    
    def pause(self):
        self.status = "paused"
    
    def resume(self):
        if self.status == "paused":
            self.status = "running"
            threading.Thread(target=self._generate_next_message, daemon=True).start()
    
    def stop(self):
        self.status = "completed"
    
    def to_dict(self):
        return {
            "id": self.id,
            "agent1_config": self.agent1_config,
            "agent2_config": self.agent2_config,
            "messages": [
                {
                    **msg,
                    "timestamp": msg["timestamp"].isoformat()
                } for msg in self.messages
            ],
            "status": self.status,
            "start_time": self.start_time.isoformat(),
            "current_turn": self.current_turn,
            "round_count": self.round_count
        }

# Global run manager
class RunManager:
    def __init__(self):
        self.runs: Dict[str, CompetitionRun] = {}
        self.api_keys: Dict[str, str] = {}
    
    def create_run(self, agent1_config: Dict, agent2_config: Dict) -> str:
        run_id = str(uuid.uuid4())
        run = CompetitionRun(run_id, agent1_config, agent2_config, self.api_keys)
        self.runs[run_id] = run
        return run_id
    
    def get_run(self, run_id: str) -> Optional[CompetitionRun]:
        return self.runs.get(run_id)
    
    def get_all_runs(self) -> Dict[str, Dict]:
        return {run_id: run.to_dict() for run_id, run in self.runs.items()}
    
    def delete_run(self, run_id: str):
        if run_id in self.runs:
            self.runs[run_id].stop()
            del self.runs[run_id]
    
    def clear_all_runs(self):
        for run in self.runs.values():
            run.stop()
        self.runs.clear()
    
    def update_api_keys(self, api_keys: Dict[str, str]):
        self.api_keys = api_keys

# Global instances
run_manager = RunManager()

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/competition')
def competition():
    return render_template('index.html')

@app.route('/api-keys')
def api_keys():
    return render_template('index.html')

# API Endpoints
@app.route('/api/runs', methods=['GET'])
def get_runs():
    return jsonify(run_manager.get_all_runs())

@app.route('/api/runs', methods=['POST'])
def create_run():
    data = request.get_json()
    
    agent1_config = {
        "llm": data.get("agent1_llm"),
        "prompt": data.get("agent1_prompt")
    }
    
    agent2_config = {
        "llm": data.get("agent2_llm"),
        "prompt": data.get("agent2_prompt")
    }
    
    run_id = run_manager.create_run(agent1_config, agent2_config)
    return jsonify({"run_id": run_id, "status": "created"})

@app.route('/api/runs/<run_id>', methods=['GET'])
def get_run(run_id):
    run = run_manager.get_run(run_id)
    if run:
        return jsonify(run.to_dict())
    return jsonify({"error": "Run not found"}), 404

@app.route('/api/runs/<run_id>/pause', methods=['POST'])
def pause_run(run_id):
    run = run_manager.get_run(run_id)
    if run:
        run.pause()
        return jsonify({"status": "paused"})
    return jsonify({"error": "Run not found"}), 404

@app.route('/api/runs/<run_id>/resume', methods=['POST'])
def resume_run(run_id):
    run = run_manager.get_run(run_id)
    if run:
        run.resume()
        return jsonify({"status": "resumed"})
    return jsonify({"error": "Run not found"}), 404

@app.route('/api/runs/<run_id>/stop', methods=['POST'])
def stop_run(run_id):
    run = run_manager.get_run(run_id)
    if run:
        run.stop()
        return jsonify({"status": "stopped"})
    return jsonify({"error": "Run not found"}), 404

@app.route('/api/runs/<run_id>', methods=['DELETE'])
def delete_run(run_id):
    run_manager.delete_run(run_id)
    return jsonify({"status": "deleted"})

@app.route('/api/runs', methods=['DELETE'])
def clear_all_runs():
    run_manager.clear_all_runs()
    return jsonify({"status": "all_runs_cleared"})

@app.route('/api/api-keys', methods=['POST'])
def save_api_keys():
    data = request.get_json()
    run_manager.update_api_keys(data)
    return jsonify({"status": "api_keys_saved"})

@app.route('/api/api-keys/test', methods=['POST'])
def test_api_keys():
    data = request.get_json()
    results = {}
    
    # Test each provider
    for provider_name, api_key in data.items():
        if not api_key:
            results[provider_name] = {"status": "not_configured"}
            continue
            
        try:
            if provider_name == "openai":
                client = openai.OpenAI(api_key=api_key)
                client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": "Test"}],
                    max_tokens=1
                )
            elif provider_name == "anthropic":
                client = anthropic.Anthropic(api_key=api_key)
                client.messages.create(
                    model="claude-3-5-haiku-20241022",
                    max_tokens=1,
                    messages=[{"role": "user", "content": [{"type": "text", "text": "Test"}]}]
                )
            elif provider_name == "xai":
                headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                payload = {
                    "model": "grok-3-mini",
                    "messages": [{"role": "user", "content": "Test"}],
                    "max_tokens": 1
                }
                response = requests.post("https://api.x.ai/v1/chat/completions", 
                                       json=payload, headers=headers)
                response.raise_for_status()
            elif provider_name == "groq":
                client = groq.Groq(api_key=api_key)
                client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": "Test"}],
                    max_tokens=1
                )
            
            results[provider_name] = {"status": "connected"}
        except Exception as e:
            results[provider_name] = {"status": "error", "message": str(e)}
    
    return jsonify(results)

@app.route('/api/models/<provider>', methods=['GET'])
def get_models(provider):
    models = {
        "openai": ["gpt-4.1", "gpt-4o", "gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1-nano", "o4-mini", "o3", "o3-mini"],
        "anthropic": ["claude-opus-4-20250514", "claude-sonnet-4-20250514", "claude-3-7-sonnet-20250219", "claude-3-5-haiku-20241022"],
        "xai": ["grok-4-0709", "grok-3", "grok-3-mini", "grok-3-fast"],
        "groq": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "gemma2-9b-it", "deepseek-r1-distill-llama-70b"]
    }
    return jsonify(models.get(provider, []))

if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5001) 