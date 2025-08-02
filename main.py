#!/usr/bin/env python3
"""
Epistemic Grounding Experiment - AI-to-AI Negotiation Platform

A research platform for studying epistemic grounding and multi-agent negotiations
across different AI providers including OpenAI, Anthropic, xAI, and Groq.

Author: Ayushi
License: MIT
Repository: https://github.com/ayushi-uwc/epistemic_grounding_experiment
"""

import os
import time
import json
import threading
import uuid
import re
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple
from flask import Flask, render_template, jsonify, request, make_response
from flask_cors import CORS
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# LLM Provider imports
import openai
import anthropic
import requests
import groq

# Load configuration
try:
    from config import config, Config
    app_config = config.get(os.environ.get('FLASK_ENV', 'development'), config['default'])
except ImportError:
    # Fallback if config.py is not available
    class AppConfig:
        DEBUG = True
        PORT = 5001
        HOST = '0.0.0.0'
        API_KEYS = {}
    app_config = AppConfig

app = Flask(__name__)
CORS(app)
app.config.from_object(app_config)

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
        self.current_model = "gpt-4o"  # Default model
        self.models = [
            "o4-mini", "o3", "o3-mini", "o1", "o1-pro", 
            "gpt-4.1", "gpt-4o", "chatgpt-4o-latest", 
            "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o-mini"
        ]
    
    def generate_response(self, prompt: str, conversation_history: List[Dict] = None) -> str:
        import time
        max_retries = 3
        retry_delay = 1  # Start with 1 second delay
        
        for attempt in range(max_retries):
            try:
                messages = []
                
                # Reasoning models (o1, o3) have different requirements
                is_reasoning_model = self.current_model.startswith(('o1', 'o3', 'o4'))
                
                if is_reasoning_model:
                    # For reasoning models, don't use system message - incorporate into user message
                    if conversation_history:
                        for msg in conversation_history[-8:]:  # Fewer messages for reasoning models
                            role = "assistant" if msg["speaker"] == "seller" else "user"
                            messages.append({"role": role, "content": msg["message"]})
                    
                    # Add the prompt as part of the last user message or create new one
                    if messages and messages[-1]["role"] == "user":
                        messages[-1]["content"] = f"{prompt}\n\n{messages[-1]['content']}"
                    else:
                        messages.append({"role": "user", "content": prompt})
                    
                    # Reasoning models use max_completion_tokens and don't support temperature
                    response = self.client.chat.completions.create(
                        model=self.current_model,
                        messages=messages,
                        max_completion_tokens=2000
                    )
                else:
                    # Regular models use system messages and max_tokens
                    messages.append({"role": "system", "content": prompt})
                    
                    # Add conversation history
                    if conversation_history:
                        for msg in conversation_history[-10:]:  # Last 10 messages for context
                            role = "assistant" if msg["speaker"] == "seller" else "user"
                            messages.append({"role": role, "content": msg["message"]})
                    
                    response = self.client.chat.completions.create(
                        model=self.current_model,
                        messages=messages,
                        temperature=0.8,
                        max_tokens=2000
                    )
                
                return response.choices[0].message.content.strip()
                
            except openai.RateLimitError as e:
                if attempt < max_retries - 1:
                    print(f"Rate limit hit for OpenAI, retrying in {retry_delay} seconds (attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                else:
                    return f"Error: OpenAI rate limit exceeded after {max_retries} retries. Please wait before making more requests."
            except Exception as e:
                if "rate_limit" in str(e).lower() and attempt < max_retries - 1:
                    print(f"Rate limit detected in error, retrying in {retry_delay} seconds (attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                else:
                    return f"Error generating OpenAI response: {str(e)}"
        
        return "Error: Maximum retries exceeded"
    
    def get_available_models(self) -> List[str]:
        return self.models

# Anthropic Provider
class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.client = anthropic.Anthropic(api_key=api_key)
        self.current_model = "claude-sonnet-4-20250514"  # Default model
        self.models = [
            "claude-sonnet-4-20250514", "claude-3-7-sonnet-20250219"
        ]
    
    def generate_response(self, prompt: str, conversation_history: List[Dict] = None) -> str:
        import time
        max_retries = 3
        retry_delay = 1  # Start with 1 second delay
        
        for attempt in range(max_retries):
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
                    model=self.current_model,
                    max_tokens=2000,
                    temperature=0.8,
                    system=prompt,
                    messages=messages
                )
                
                return response.content[0].text.strip()
                
            except anthropic.RateLimitError as e:
                if attempt < max_retries - 1:
                    print(f"Rate limit hit for Anthropic, retrying in {retry_delay} seconds (attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                else:
                    return f"Error: Anthropic rate limit exceeded after {max_retries} retries. Please wait before making more requests."
            except Exception as e:
                if "rate_limit" in str(e).lower() and attempt < max_retries - 1:
                    print(f"Rate limit detected in error, retrying in {retry_delay} seconds (attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                else:
                    return f"Error generating Anthropic response: {str(e)}"
        
        return "Error: Maximum retries exceeded"
    
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

# Batch Execution Management
class BatchExecution:
    def __init__(self, batch_id: str, config: Dict, api_keys: Dict, run_manager_ref=None):
        self.id = batch_id
        self.batch_id = batch_id  # For consistency with access patterns
        self.config = config
        self.runs: Dict[str, CompetitionRun] = {}
        self.status = "initializing"
        self.start_time = datetime.now()
        self.completion_time = None
        self.total_combinations = 0
        self.completed_runs = 0
        self.run_manager_ref = run_manager_ref
        self.api_keys = api_keys
        self.batch_results = {}  # Store complete results for download
        
        # Start batch execution in background
        threading.Thread(target=self._start_batch_execution, daemon=True).start()
    
    def _start_batch_execution(self):
        try:
            cases = self.config.get('cases', [])
            if not cases:
                self.status = "error"
                return
                
            # Generate all N×N combinations
            combinations = []
            for buyer_case in cases:
                for seller_case in cases:
                    combinations.append({
                        'buyer_case': buyer_case,
                        'seller_case': seller_case
                    })
            
            self.total_combinations = len(combinations)
            self.status = "running"
            
            # Execute combinations in parallel (max 2 concurrent to avoid rate limits)
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_to_combo = {}
                
                # Submit combinations with small delays to avoid rate limits
                for i, combo in enumerate(combinations):
                    if i > 0 and i % 2 == 0:  # Add delay every 2 submissions
                        import time
                        time.sleep(0.5)  # 500ms delay
                    
                    future = executor.submit(self._execute_combination, combo)
                    future_to_combo[future] = combo
                
                for future in as_completed(future_to_combo):
                    combo = future_to_combo[future]
                    try:
                        run_id = future.result()
                        self.completed_runs += 1
                        print(f"Completed combination {self.completed_runs}/{self.total_combinations}: {combo['buyer_case']['name']} vs {combo['seller_case']['name']}")
                    except Exception as exc:
                        print(f'Combination {combo} generated an exception: {exc}')
                        self.completed_runs += 1  # Count failed runs too
            
            self.status = "completed"
            self.completion_time = datetime.now()
            print(f"Batch execution {self.id} completed: {self.completed_runs}/{self.total_combinations} runs")
            
        except Exception as e:
            print(f"Error in batch execution: {e}")
            self.status = "error"
    
    def _execute_combination(self, combination) -> str:
        # Create unique run ID for this combination
        run_id = f"{self.id}_{combination['buyer_case']['name']}_vs_{combination['seller_case']['name']}"
        
        # Build full prompts (system + case)
        buyer_prompt = self.config['buyer_system_prompt'] + "\n\n" + combination['buyer_case']['buyerPrompt']
        seller_prompt = self.config['seller_system_prompt'] + "\n\n" + combination['seller_case']['sellerPrompt']
        
        # Create agent configs
        agent1_config = {
            "llm": combination['buyer_case']['buyerLLM'],
            "prompt": buyer_prompt,
            "role": "buyer"
        }
        
        agent2_config = {
            "llm": combination['seller_case']['sellerLLM'], 
            "prompt": seller_prompt,
            "role": "seller"
        }
        
        # Prepare matrix execution info
        matrix_info = {
            "buyer_case": combination['buyer_case']['name'],
            "seller_case": combination['seller_case']['name'],
            "matrix_combination": f"{combination['buyer_case']['name']} vs {combination['seller_case']['name']}",
            "buyer_full_prompt": buyer_prompt,
            "seller_full_prompt": seller_prompt,
            "buyer_system_prompt": self.config['buyer_system_prompt'],
            "seller_system_prompt": self.config['seller_system_prompt'],
            "buyer_case_prompt": combination['buyer_case']['buyerPrompt'],
            "seller_case_prompt": combination['seller_case']['sellerPrompt'],
            "buyer_llm": combination['buyer_case']['buyerLLM'],
            "seller_llm": combination['seller_case']['sellerLLM']
        }
        
        # Create and start the run
        run = CompetitionRun(
            run_id, 
            agent1_config, 
            agent2_config, 
            self.api_keys,
            max_rounds=int(self.config.get('max_turns', 40)),
            turn_delay=float(self.config.get('turn_delay', 3)),
            first_speaker=self.config.get('first_speaker', 'seller'),
            termination_prompt=self.config.get('termination_prompt', ''),
            matrix_info=matrix_info
        )
        
        self.runs[run_id] = run
        
        # Register with global run manager if available
        if self.run_manager_ref:
            self.run_manager_ref.runs[run_id] = run
        
        # Wait for run completion and store results
        threading.Thread(target=self._monitor_and_store_results, args=(run_id, run), daemon=True).start()
            
        return run_id
    
    def _monitor_and_store_results(self, run_id: str, run: 'CompetitionRun'):
        """Monitor a run completion and store its results"""
        try:
            # Wait for run to complete
            while run.status in ['initializing', 'running']:
                time.sleep(1)
            
            # Extract offers and store complete results
            offers_data = run.extract_offers()
            
            # Store complete run data
            self.batch_results[run_id] = {
                **run.to_dict(),
                "offers_analysis": offers_data,
                "completion_timestamp": datetime.now().isoformat()
            }
            
            print(f"Stored results for {run_id}: Initial Buyer: ${offers_data.get('initial_buyer_offer', 'N/A')}, Initial Seller: ${offers_data.get('initial_seller_offer', 'N/A')}, Final: ${offers_data.get('agreed_price', 'No agreement')}")
            
        except Exception as e:
            print(f"Error monitoring run {run_id}: {e}")
    
    def to_dict(self):
        return {
            "id": self.id,
            "config": self.config,
            "status": self.status,
            "start_time": self.start_time.isoformat(),
            "total_combinations": self.total_combinations,
            "completed_runs": self.completed_runs,
            "progress": (self.completed_runs / self.total_combinations * 100) if self.total_combinations > 0 else 0,
            "runs": {run_id: run.to_dict() for run_id, run in self.runs.items()},
            "batch_results": self.batch_results,
            "results_summary": self._get_results_summary()
        }
    
    def _get_results_summary(self):
        """Generate a summary of batch results"""
        try:
            if not self.batch_results:
                return {"status": "no_results", "message": "Batch execution in progress or no results available"}
            
            successful_negotiations = 0
            total_agreements = 0
            price_ranges = {"min_agreed": None, "max_agreed": None}
            
            for run_data in self.batch_results.values():
                offers = run_data.get("offers_analysis", {})
                if offers.get("negotiation_successful"):
                    successful_negotiations += 1
                    agreed_price = offers.get("agreed_price")
                    if agreed_price:
                        total_agreements += 1
                        if price_ranges["min_agreed"] is None or agreed_price < price_ranges["min_agreed"]:
                            price_ranges["min_agreed"] = agreed_price
                        if price_ranges["max_agreed"] is None or agreed_price > price_ranges["max_agreed"]:
                            price_ranges["max_agreed"] = agreed_price
            
            return {
                "total_runs": len(self.batch_results),
                "successful_negotiations": successful_negotiations,
                "success_rate": (successful_negotiations / len(self.batch_results) * 100) if self.batch_results else 0,
                "price_range": price_ranges,
                "average_agreed_price": sum(
                    run_data.get("offers_analysis", {}).get("agreed_price", 0) 
                    for run_data in self.batch_results.values() 
                    if run_data.get("offers_analysis", {}).get("agreed_price")
                ) / total_agreements if total_agreements > 0 else None
            }
        except Exception as e:
            return {"status": "error", "message": f"Error generating summary: {str(e)}"}

# Competition Run Management
class CompetitionRun:
    def __init__(self, run_id: str, agent1_config: Dict, agent2_config: Dict, api_keys: Dict, 
                 max_rounds: int = 20, turn_delay: float = 3.0, first_speaker: str = "seller",
                 termination_prompt: str = "", matrix_info: Dict = None):
        self.id = run_id
        self.agent1_config = agent1_config  # buyer
        self.agent2_config = agent2_config  # seller
        self.api_keys = api_keys
        self.messages = []
        self.status = "initializing"
        self.start_time = datetime.now()
        self.round_count = 0
        self.max_rounds = max_rounds
        self.turn_delay = turn_delay
        self.termination_prompt = termination_prompt
        self.termination_reason = ""  # Store the reason for termination
        self.matrix_info = matrix_info or {}  # Store matrix execution details
        
        # Map UI first_speaker (buyer/seller) to internal agent (agent1/agent2)
        # agent1 = buyer, agent2 = seller
        if first_speaker == "buyer":
            self.current_turn = "agent1"
        elif first_speaker == "seller":
            self.current_turn = "agent2"
        else:
            # Default to seller if invalid value
            self.current_turn = "agent2"
        
        # Initialize LLM providers
        self.agent1_provider = self._create_provider(agent1_config["llm"], api_keys)
        self.agent2_provider = self._create_provider(agent2_config["llm"], api_keys)
        
        # Start the competition
        self._start_competition()
    
    def _get_provider_from_model(self, model_id: str) -> str:
        """Map model ID to provider name"""
        openai_models = ["o4-mini", "o3", "o3-mini", "o1", "o1-pro", "gpt-4.1", "gpt-4o", "chatgpt-4o-latest", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o-mini"]
        anthropic_models = ["claude-sonnet-4-20250514", "claude-3-7-sonnet-20250219"]
        
        if model_id in openai_models:
            return "openai"
        elif model_id in anthropic_models:
            return "anthropic"
        else:
            # Fallback: try to infer from model name
            if "gpt" in model_id.lower() or "o1" in model_id.lower() or "o3" in model_id.lower() or "o4" in model_id.lower() or "chatgpt" in model_id.lower():
                return "openai"
            elif "claude" in model_id.lower():
                return "anthropic"
            else:
                return "openai"  # Default fallback
    
    def _create_provider(self, provider_name: str, api_keys: Dict) -> Optional[LLMProvider]:
        try:
            # If provider_name looks like a model ID, extract the actual provider
            actual_provider = self._get_provider_from_model(provider_name)
            
            if actual_provider == "openai" and "openai" in api_keys:
                provider = OpenAIProvider(api_keys["openai"])
                provider.current_model = provider_name  # Set the specific model
                return provider
            elif actual_provider == "anthropic" and "anthropic" in api_keys:
                provider = AnthropicProvider(api_keys["anthropic"])
                provider.current_model = provider_name  # Set the specific model
                return provider
            else:
                return None
        except Exception as e:
            print(f"Error creating provider for model {provider_name}: {e}")
            return None
    
    def _start_competition(self):
        if not self.agent1_provider or not self.agent2_provider:
            self.status = "error"
            self.termination_reason = "Error: Could not initialize LLM providers. Check API keys."
            self.add_message("system", "Error: Could not initialize LLM providers. Check API keys.")
            return
        
        self.status = "running"
        # Start with the configured first speaker (current_turn already set in __init__)
        threading.Thread(target=self._generate_next_message, daemon=True).start()
    
    def _generate_next_message(self):
        # Check if we've reached max rounds (complete buyer+seller rounds)
        # max_rounds = max complete turns, each turn = buyer + seller messages
        agent_messages = len([m for m in self.messages if m["speaker"] in ["agent1", "agent2"]])
        max_total_messages = self.max_rounds * 2  # Each turn = 2 messages
        
        if self.status != "running" or agent_messages >= max_total_messages:
            self.status = "completed"
            self.termination_reason = f"Max turns reached: {self.max_rounds} complete turns ({max_total_messages} messages)"
            self.add_message("system", f"Negotiation completed: reached maximum {self.max_rounds} complete turns ({max_total_messages} messages)")
            return
        
        try:
            # Check custom termination condition if provided
            if self.termination_prompt:
                should_terminate, reason = self._should_terminate()
                if should_terminate:
                    self.status = "completed"
                    self.termination_reason = f"Custom termination: {reason}"
                    self.add_message("system", f"Negotiation terminated by custom condition: {reason}")
                    return
            
            # Determine which agent should respond
            if self.current_turn == "agent1":
                provider = self.agent1_provider
                prompt = self.agent1_config["prompt"]
                speaker = "agent1"
                role_name = self.agent1_config.get("role", "agent1")
            else:
                provider = self.agent2_provider
                prompt = self.agent2_config["prompt"]
                speaker = "agent2"
                role_name = self.agent2_config.get("role", "agent2")
            
            # Generate response
            response = provider.generate_response(prompt, self.messages)
            
            # Add message
            self.add_message(speaker, response, role_name)
            
            # Switch turns
            self.current_turn = "agent2" if self.current_turn == "agent1" else "agent1"
            
            # Schedule next message after delay
            agent_messages = len([m for m in self.messages if m["speaker"] in ["agent1", "agent2"]])
            max_total_messages = self.max_rounds * 2
            
            if self.status == "running" and agent_messages < max_total_messages:
                threading.Timer(self.turn_delay, self._generate_next_message).start()
            else:
                self.status = "completed"
                self.termination_reason = f"Max turns reached: {self.max_rounds} complete turns ({max_total_messages} messages)"
                self.add_message("system", f"Negotiation completed: reached maximum {self.max_rounds} complete turns ({max_total_messages} messages)")
                
        except Exception as e:
            self.add_message("system", f"Error: {str(e)}")
            self.status = "error"
            self.termination_reason = f"Error occurred: {str(e)}"
    
    def _should_terminate(self) -> tuple[bool, str]:
        """Check if the conversation should terminate based on custom termination prompt
        Returns: (should_terminate: bool, termination_reason: str)
        """
        if not self.termination_prompt or len(self.messages) < 4:
            return False, ""
        
        try:
            # Use the first available provider to evaluate termination
            provider = self.agent1_provider or self.agent2_provider
            if not provider:
                return False, ""
            
            # Get last few messages for context
            recent_messages = self.messages[-6:]  # More context for better evaluation
            conversation_context = "\n".join([
                f"{msg.get('role_name', msg['speaker'])}: {msg['message']}" 
                for msg in recent_messages 
                if msg['speaker'] not in ['system']
            ])
            
            termination_check = f"""{self.termination_prompt}

Recent conversation:
{conversation_context}

Please evaluate if the negotiation should end based on the criteria above. Respond with a JSON object containing:
1. "terminate": true/false - whether the negotiation should end
2. "reason": "string" - brief explanation of why termination is/isn't warranted

Example response:
{{"terminate": true, "reason": "Both parties have reached a mutual agreement on price and terms"}}

Your response:"""
            
            response = provider.generate_response(termination_check, [])
            
            # Try to parse JSON response
            import json
            try:
                # Clean response (remove markdown formatting if present)
                clean_response = response.strip()
                if clean_response.startswith('```json'):
                    clean_response = clean_response.replace('```json', '').replace('```', '').strip()
                elif clean_response.startswith('```'):
                    # Handle other markdown code blocks
                    clean_response = clean_response.replace('```', '').strip()
                
                # Try to extract JSON if it's embedded in text
                if '{' in clean_response and '}' in clean_response:
                    start = clean_response.find('{')
                    end = clean_response.rfind('}') + 1
                    json_part = clean_response[start:end]
                else:
                    json_part = clean_response
                
                result = json.loads(json_part)
                should_terminate = result.get('terminate', False)
                reason = result.get('reason', 'Custom termination condition met')
                
                # Log for debugging
                print(f"Termination check - JSON parsed successfully: terminate={should_terminate}, reason={reason}")
                
                return bool(should_terminate), reason
                
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                print(f"JSON parsing failed: {e}. Response was: {response[:200]}...")
                # Fallback: check if response contains positive termination indicators
                response_lower = response.lower()
                if any(word in response_lower for word in ['agreement reached', 'deal concluded', 'negotiation settled', 'final agreement']):
                    return True, f"Agreement detected in non-JSON response: {response[:100]}..."
                elif any(word in response_lower for word in ['should terminate', 'should end', 'negotiation over']):
                    return True, f"Termination condition met in non-JSON response: {response[:100]}..."
                else:
                    return False, f"Could not parse termination response: {response[:100]}..."
            
        except Exception as e:
            print(f"Error checking termination condition: {e}")
            return False, f"Error in termination check: {str(e)}"
    
    def add_message(self, speaker: str, content: str, role_name: str = None):
        message = {
            "speaker": speaker,
            "message": content,
            "timestamp": datetime.now(),
            "role_name": role_name or speaker
        }
        self.messages.append(message)
        
        # Update round count: 1 turn = 1 complete round (buyer + seller)
        # So round_count = total agent messages / 2
        if speaker in ["agent1", "agent2"]:
            agent_messages = len([m for m in self.messages if m["speaker"] in ["agent1", "agent2"]])
            self.round_count = (agent_messages + 1) // 2  # Round up for current round
            message["round"] = self.round_count
        else:
            message["round"] = self.round_count
    
    def pause(self):
        self.status = "paused"
    
    def resume(self):
        if self.status == "paused":
            self.status = "running"
            threading.Thread(target=self._generate_next_message, daemon=True).start()
    
    def stop(self):
        self.status = "completed"
        if not self.termination_reason:  # Only set if not already set
            self.termination_reason = "Manually stopped by user"
    
    def extract_offers(self) -> Dict[str, Any]:
        """Extract initial and final offers from the conversation"""
        try:
            buyer_messages = [msg for msg in self.messages if msg.get('role_name') == 'buyer' and msg['speaker'] in ['agent1', 'agent2']]
            seller_messages = [msg for msg in self.messages if msg.get('role_name') == 'seller' and msg['speaker'] in ['agent1', 'agent2']]
            
            # Extract monetary amounts using regex
            def extract_amounts(text: str) -> List[float]:
                # Look for patterns like $1,234, $1234.56, 1,234, 1234.56
                patterns = [
                    r'\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',  # $1,234.56
                    r'\$(\d+(?:\.\d{2})?)',                  # $1234.56
                    r'(\d{1,3}(?:,\d{3})*)\s*(?:dollars?|USD|\$)',  # 1,234 dollars
                    r'(\d+)\s*(?:thousand|k)',               # 20 thousand
                ]
                
                amounts = []
                for pattern in patterns:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    for match in matches:
                        try:
                            # Clean up the match and convert to float
                            clean_amount = match.replace(',', '')
                            if 'thousand' in text.lower() or 'k' in text.lower():
                                amounts.append(float(clean_amount) * 1000)
                            else:
                                amounts.append(float(clean_amount))
                        except ValueError:
                            continue
                
                return amounts
            
            # Get initial offers (first messages with amounts)
            initial_buyer_offer = None
            initial_seller_offer = None
            
            for msg in buyer_messages[:3]:  # Check first 3 buyer messages
                amounts = extract_amounts(msg['message'])
                if amounts and initial_buyer_offer is None:
                    initial_buyer_offer = amounts[0]
                    break
            
            for msg in seller_messages[:3]:  # Check first 3 seller messages
                amounts = extract_amounts(msg['message'])
                if amounts and initial_seller_offer is None:
                    initial_seller_offer = amounts[0]
                    break
            
            # Get final offers (last messages with amounts)
            final_buyer_offer = None
            final_seller_offer = None
            
            for msg in reversed(buyer_messages[-5:]):  # Check last 5 buyer messages
                amounts = extract_amounts(msg['message'])
                if amounts:
                    final_buyer_offer = amounts[-1]  # Take the last amount mentioned
                    break
            
            for msg in reversed(seller_messages[-5:]):  # Check last 5 seller messages
                amounts = extract_amounts(msg['message'])
                if amounts:
                    final_seller_offer = amounts[-1]  # Take the last amount mentioned
                    break
            
            # If negotiation was successful, final offers should be the same
            agreed_price = None
            if (final_buyer_offer and final_seller_offer and 
                abs(final_buyer_offer - final_seller_offer) < 100):  # Within $100
                agreed_price = final_seller_offer  # Use seller's final offer as agreed price
            
            return {
                "initial_buyer_offer": initial_buyer_offer,
                "initial_seller_offer": initial_seller_offer,
                "final_buyer_offer": final_buyer_offer,
                "final_seller_offer": final_seller_offer,
                "agreed_price": agreed_price,
                "negotiation_successful": agreed_price is not None,
                "price_movement": {
                    "buyer_movement": (final_buyer_offer - initial_buyer_offer) if (initial_buyer_offer and final_buyer_offer) else None,
                    "seller_movement": (final_seller_offer - initial_seller_offer) if (initial_seller_offer and final_seller_offer) else None
                }
            }
            
        except Exception as e:
            print(f"Error extracting offers: {e}")
            return {
                "initial_buyer_offer": None,
                "initial_seller_offer": None,
                "final_buyer_offer": None,
                "final_seller_offer": None,
                "agreed_price": None,
                "negotiation_successful": False,
                "price_movement": {"buyer_movement": None, "seller_movement": None},
                "extraction_error": str(e)
            }
    
    def to_dict(self):
        agent_messages = len([m for m in self.messages if m["speaker"] in ["agent1", "agent2"]])
        max_total_messages = self.max_rounds * 2
        
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
            "round_count": self.round_count,
            "max_rounds": self.max_rounds,
            "turn_delay": self.turn_delay,
            "termination_prompt": self.termination_prompt,
            "termination_reason": self.termination_reason,
            "matrix_info": self.matrix_info,
            "progress": {
                "current_complete_turns": self.round_count,
                "max_complete_turns": self.max_rounds,
                "total_messages": agent_messages,
                "max_total_messages": max_total_messages,
                "progress_percentage": (agent_messages / max_total_messages * 100) if max_total_messages > 0 else 0
            },
            "execution_settings": {
                "max_rounds": self.max_rounds,
                "turn_delay": self.turn_delay,
                "current_turn": self.current_turn,
                "agent1_role": self.agent1_config.get("role", "agent1"),
                "agent2_role": self.agent2_config.get("role", "agent2"),
                "termination_prompt": self.termination_prompt,
                "termination_reason": self.termination_reason,
                "turn_counting": "1 turn = buyer message + seller message"
            }
        }

# Global run manager
class RunManager:
    def __init__(self):
        self.runs: Dict[str, CompetitionRun] = {}
        self.batch_executions: Dict[str, BatchExecution] = {}
        self.api_keys: Dict[str, str] = {}
        self.saved_config: Dict = {}
    
    def create_run(self, agent1_config: Dict, agent2_config: Dict, **kwargs) -> str:
        run_id = str(uuid.uuid4())
        
        # For single runs, create basic matrix info if not provided
        if 'matrix_info' not in kwargs:
            kwargs['matrix_info'] = {
                "buyer_case": "single_run",
                "seller_case": "single_run", 
                "matrix_combination": "Single Run (Legacy Mode)",
                "buyer_full_prompt": agent1_config.get('prompt', ''),
                "seller_full_prompt": agent2_config.get('prompt', ''),
                "buyer_system_prompt": "N/A (Single Run)",
                "seller_system_prompt": "N/A (Single Run)",
                "buyer_case_prompt": "N/A (Single Run)",
                "seller_case_prompt": "N/A (Single Run)", 
                "buyer_llm": agent1_config.get('llm', 'unknown'),
                "seller_llm": agent2_config.get('llm', 'unknown')
            }
        
        # Use API keys from frontend (stored in self.api_keys)
        run = CompetitionRun(run_id, agent1_config, agent2_config, self.api_keys, **kwargs)
        self.runs[run_id] = run
        return run_id
    
    def create_batch_execution(self, config: Dict) -> str:
        batch_id = str(uuid.uuid4())
        # Pass API keys from frontend to batch execution
        batch = BatchExecution(batch_id, config, self.api_keys, run_manager_ref=self)
        self.batch_executions[batch_id] = batch
        
        # Note: Individual runs will be added to self.runs as they are created
        # in the background thread within BatchExecution
            
        return batch_id
    
    def get_run(self, run_id: str) -> Optional[CompetitionRun]:
        return self.runs.get(run_id)
    
    def get_batch_execution(self, batch_id: str) -> Optional[BatchExecution]:
        return self.batch_executions.get(batch_id)
    
    def get_all_runs(self) -> Dict[str, Dict]:
        return {run_id: run.to_dict() for run_id, run in self.runs.items()}
    
    def get_all_batch_executions(self) -> Dict[str, Dict]:
        return {batch_id: batch.to_dict() for batch_id, batch in self.batch_executions.items()}
    
    def delete_run(self, run_id: str):
        if run_id in self.runs:
            self.runs[run_id].stop()
            del self.runs[run_id]
    
    def delete_batch_execution(self, batch_id: str):
        if batch_id in self.batch_executions:
            batch = self.batch_executions[batch_id]
            # Stop and remove all individual runs
            for run_id in batch.runs.keys():
                if run_id in self.runs:
                    self.runs[run_id].stop()
                    del self.runs[run_id]
            del self.batch_executions[batch_id]
    
    def clear_all_runs(self):
        for run in self.runs.values():
            run.stop()
        self.runs.clear()
        self.batch_executions.clear()
    
    def update_api_keys(self, api_keys: Dict[str, str]):
        self.api_keys = api_keys
    
    def save_configuration(self, config: Dict):
        self.saved_config = config
    
    def load_configuration(self) -> Dict:
        return self.saved_config

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
    
    # Check if API keys are configured
    if not any(run_manager.api_keys.get(provider, '').strip() for provider in ['openai', 'anthropic', 'xai', 'groq']):
        return jsonify({"error": "No API keys configured. Please configure at least one LLM provider in API Keys settings."}), 400
    
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

@app.route('/api/models', methods=['GET'])
def get_available_models():
    """Get all available models for OpenAI and Anthropic only"""
    models = {
        "openai": {
            "provider_name": "OpenAI", 
            "models": [
                {"id": "o4-mini", "name": "o4-mini"},
                {"id": "o3", "name": "o3"},
                {"id": "o3-mini", "name": "o3-mini"},
                {"id": "o1", "name": "o1"},
                {"id": "o1-pro", "name": "o1-pro"},
                {"id": "gpt-4.1", "name": "GPT-4.1"},
                {"id": "gpt-4o", "name": "GPT-4o"},
                {"id": "chatgpt-4o-latest", "name": "ChatGPT-4o Latest"},
                {"id": "gpt-4.1-mini", "name": "GPT-4.1 Mini"},
                {"id": "gpt-4.1-nano", "name": "GPT-4.1 Nano"},
                {"id": "gpt-4o-mini", "name": "GPT-4o Mini"}
            ]
        },
        "anthropic": {
            "provider_name": "Anthropic",
            "models": [
                {"id": "claude-sonnet-4-20250514", "name": "Claude Sonnet 4"},
                {"id": "claude-3-7-sonnet-20250219", "name": "Claude Sonnet 3.7"}
            ]
        }
    }
    return jsonify(models)

@app.route('/api/models/<provider>', methods=['GET'])
def get_models(provider):
    """Legacy endpoint for backwards compatibility"""
    all_models = {
        "openai": ["o4-mini", "o3", "o3-mini", "o1", "o1-pro", "gpt-4.1", "gpt-4o", "chatgpt-4o-latest", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o-mini"],
        "anthropic": ["claude-sonnet-4-20250514", "claude-3-7-sonnet-20250219"]
    }
    return jsonify(all_models.get(provider, []))

# Batch Execution Endpoints
@app.route('/api/batch-execute', methods=['POST'])
def batch_execute():
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['buyer_system_prompt', 'seller_system_prompt', 'cases']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        if not data['cases']:
            return jsonify({"error": "At least one case is required"}), 400
        
        # Check if API keys are configured
        if not any(run_manager.api_keys.get(provider, '').strip() for provider in ['openai', 'anthropic', 'xai', 'groq']):
            return jsonify({"error": "No API keys configured. Please configure at least one LLM provider in API Keys settings."}), 400
        
        # Create batch execution
        batch_id = run_manager.create_batch_execution(data)
        
        return jsonify({
            "batch_id": batch_id,
            "status": "started",
            "total_combinations": len(data['cases']) * len(data['cases'])
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/batch-preview', methods=['POST'])
def batch_preview():
    try:
        data = request.get_json()
        
        cases = data.get('cases', [])
        if not cases:
            return jsonify({"error": "No cases provided"}), 400
        
        # Generate preview of all combinations
        preview = []
        for buyer_case in cases:
            for seller_case in cases:
                preview.append({
                    "buyer_case": buyer_case['name'],
                    "buyer_llm": buyer_case['buyerLLM'],
                    "seller_case": seller_case['name'],
                    "seller_llm": seller_case['sellerLLM'],
                    "combination_id": f"{buyer_case['name']}_vs_{seller_case['name']}"
                })
        
        return jsonify({
            "preview": preview,
            "total_combinations": len(preview),
            "settings": {
                "first_speaker": data.get('first_speaker', 'seller'),
                "max_turns": data.get('max_turns', 40),
                "turn_delay": data.get('turn_delay', 3),
                "termination_prompt": data.get('termination_prompt', '')
            }
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/batch-executions', methods=['GET'])
def get_batch_executions():
    return jsonify(run_manager.get_all_batch_executions())

@app.route('/api/batch-executions/<batch_id>', methods=['GET'])
def get_batch_execution(batch_id):
    batch = run_manager.get_batch_execution(batch_id)
    if batch:
        return jsonify(batch.to_dict())
    return jsonify({"error": "Batch execution not found"}), 404

@app.route('/api/batch-executions/<batch_id>', methods=['DELETE'])
def delete_batch_execution(batch_id):
    run_manager.delete_batch_execution(batch_id)
    return jsonify({"status": "deleted"})

@app.route('/api/batch-executions/latest/download', methods=['GET'])
def download_latest_batch_results():
    """Download the latest completed batch execution results as JSON"""
    try:
        # Find the latest completed batch execution
        latest_batch = None
        latest_timestamp = None
        
        for batch_id, batch in run_manager.batch_executions.items():
            if batch.status == "completed":
                if latest_timestamp is None or batch.completion_time > latest_timestamp:
                    latest_batch = batch
                    latest_timestamp = batch.completion_time
        
        if latest_batch is None:
            return jsonify({"error": "No completed batch executions found"}), 404
        
        # Use the same logic as the specific batch download  
        results_package = {
            "download_timestamp": datetime.now().isoformat(),
            "download_type": "latest_batch_results",
            "batch_info": {
                "batch_id": latest_batch.id,
                "status": latest_batch.status,
                "start_time": latest_batch.start_time.isoformat(),
                "completion_time": latest_batch.completion_time.isoformat() if latest_batch.completion_time else None,
                "total_combinations": latest_batch.total_combinations,
                "completed_runs": latest_batch.completed_runs,
                "config": latest_batch.config
            },
            "results_summary": latest_batch._get_results_summary(),
            "individual_runs": {},
            "aggregated_analysis": {}
        }
        
        # Add individual run results with offer analysis
        for run_id, run_data in latest_batch.batch_results.items():
            # Enhance run data with formatted conversation
            enhanced_run_data = run_data.copy()
            
            # Create a formatted conversation for easy reading
            conversation_flow = []
            buyer_messages = []
            seller_messages = []
            
            for msg in run_data.get("messages", []):
                if msg.get("speaker") in ["agent1", "agent2"] and msg.get("role_name"):
                    formatted_msg = {
                        "round": msg.get("round", 0),
                        "timestamp": msg.get("timestamp", ""),
                        "speaker": msg.get("role_name", "unknown"),
                        "message": msg.get("message", ""),
                        "speaker_id": msg.get("speaker", "")
                    }
                    
                    conversation_flow.append(formatted_msg)
                    
                    if msg.get("role_name") == "buyer":
                        buyer_messages.append(formatted_msg)
                    elif msg.get("role_name") == "seller":
                        seller_messages.append(formatted_msg)
            
            # Add conversation analysis
            enhanced_run_data["conversation_analysis"] = {
                "total_messages": len(conversation_flow),
                "buyer_message_count": len(buyer_messages),
                "seller_message_count": len(seller_messages),
                "conversation_flow": conversation_flow,
                "buyer_messages_only": buyer_messages,
                "seller_messages_only": seller_messages,
                "conversation_summary": {
                    "first_speaker": conversation_flow[0]["speaker"] if conversation_flow else "unknown",
                    "last_speaker": conversation_flow[-1]["speaker"] if conversation_flow else "unknown",
                    "rounds_completed": max([msg.get("round", 0) for msg in conversation_flow]) if conversation_flow else 0
                }
            }
            
            # Add human-readable dialogue format
            dialogue_transcript = []
            for msg in conversation_flow:
                dialogue_transcript.append(f"[Round {msg['round']}] {msg['speaker'].upper()}: {msg['message']}")
            
            enhanced_run_data["dialogue_transcript"] = "\n\n".join(dialogue_transcript)
            
            results_package["individual_runs"][run_id] = enhanced_run_data
        
        # Add aggregated analysis
        case_combinations = {}
        for run_id, run_data in latest_batch.batch_results.items():
            matrix_info = run_data.get("matrix_info", {})
            buyer_case = matrix_info.get("buyer_case", "unknown")
            seller_case = matrix_info.get("seller_case", "unknown")
            combination_key = f"{buyer_case}_vs_{seller_case}"
            
            if combination_key not in case_combinations:
                case_combinations[combination_key] = {
                    "buyer_case": buyer_case,
                    "seller_case": seller_case,
                    "runs": [],
                    "success_count": 0,
                    "total_runs": 0,
                    "avg_rounds": 0,
                    "price_range": {"min": None, "max": None},
                    "agreed_prices": []
                }
            
            case_combinations[combination_key]["runs"].append(run_id)
            case_combinations[combination_key]["total_runs"] += 1
            
            offers = run_data.get("offers_analysis", {})
            if offers.get("agreed_price"):
                case_combinations[combination_key]["success_count"] += 1
                case_combinations[combination_key]["agreed_prices"].append(offers["agreed_price"])
                
                if case_combinations[combination_key]["price_range"]["min"] is None:
                    case_combinations[combination_key]["price_range"]["min"] = offers["agreed_price"]
                    case_combinations[combination_key]["price_range"]["max"] = offers["agreed_price"]
                else:
                    case_combinations[combination_key]["price_range"]["min"] = min(case_combinations[combination_key]["price_range"]["min"], offers["agreed_price"])
                    case_combinations[combination_key]["price_range"]["max"] = max(case_combinations[combination_key]["price_range"]["max"], offers["agreed_price"])
            
            # Calculate average rounds
            total_rounds = sum([run_data.get("round_count", 0) for run_data in [latest_batch.batch_results[rid] for rid in case_combinations[combination_key]["runs"]]])
            case_combinations[combination_key]["avg_rounds"] = total_rounds / case_combinations[combination_key]["total_runs"] if case_combinations[combination_key]["total_runs"] > 0 else 0
        
        results_package["aggregated_analysis"] = case_combinations
        
        # Return as downloadable JSON
        response = make_response(json.dumps(results_package, indent=2))
        response.headers['Content-Type'] = 'application/json'
        response.headers['Content-Disposition'] = f'attachment; filename="latest_batch_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json"'
        
        return response
        
    except Exception as e:
        print(f"Error downloading latest batch results: {e}")
        return jsonify({"error": "Failed to download latest batch results"}), 500

@app.route('/api/batch-executions/<batch_id>/download', methods=['GET'])
def download_batch_results(batch_id):
    """Download complete batch execution results as JSON"""
    try:
        if batch_id not in run_manager.batch_executions:
            return jsonify({"error": "Batch execution not found"}), 404
        
        batch = run_manager.batch_executions[batch_id]
        
        # Create comprehensive results package
        results_package = {
            "download_timestamp": datetime.now().isoformat(),
            "download_type": "batch_results",
            "batch_info": {
                "batch_id": batch.id,
                "status": batch.status,
                "start_time": batch.start_time.isoformat(),
                "completion_time": batch.completion_time.isoformat() if batch.completion_time else None,
                "total_combinations": batch.total_combinations,
                "completed_runs": batch.completed_runs,
                "config": batch.config
            },
            "results_summary": batch._get_results_summary(),
            "individual_runs": {},
            "aggregated_analysis": {}
        }
        
        # Add individual run results with offer analysis
        for run_id, run_data in batch.batch_results.items():
            # Enhance run data with formatted conversation
            enhanced_run_data = run_data.copy()
            
            # Create a formatted conversation for easy reading
            conversation_flow = []
            buyer_messages = []
            seller_messages = []
            
            for msg in run_data.get("messages", []):
                if msg.get("speaker") in ["agent1", "agent2"] and msg.get("role_name"):
                    formatted_msg = {
                        "round": msg.get("round", 0),
                        "timestamp": msg.get("timestamp", ""),
                        "speaker": msg.get("role_name", "unknown"),
                        "message": msg.get("message", ""),
                        "speaker_id": msg.get("speaker", "")
                    }
                    
                    conversation_flow.append(formatted_msg)
                    
                    if msg.get("role_name") == "buyer":
                        buyer_messages.append(formatted_msg)
                    elif msg.get("role_name") == "seller":
                        seller_messages.append(formatted_msg)
            
            # Add conversation analysis
            enhanced_run_data["conversation_analysis"] = {
                "total_messages": len(conversation_flow),
                "buyer_message_count": len(buyer_messages),
                "seller_message_count": len(seller_messages),
                "conversation_flow": conversation_flow,
                "buyer_messages_only": buyer_messages,
                "seller_messages_only": seller_messages,
                "conversation_summary": {
                    "first_speaker": conversation_flow[0]["speaker"] if conversation_flow else "unknown",
                    "last_speaker": conversation_flow[-1]["speaker"] if conversation_flow else "unknown",
                    "rounds_completed": max([msg.get("round", 0) for msg in conversation_flow]) if conversation_flow else 0
                }
            }
            
            # Add human-readable dialogue format
            dialogue_transcript = []
            for msg in conversation_flow:
                dialogue_transcript.append(f"[Round {msg['round']}] {msg['speaker'].upper()}: {msg['message']}")
            
            enhanced_run_data["dialogue_transcript"] = "\n\n".join(dialogue_transcript)
            
            results_package["individual_runs"][run_id] = enhanced_run_data
        
        # Add aggregated analysis
        case_combinations = {}
        for run_id, run_data in batch.batch_results.items():
            matrix_info = run_data.get("matrix_info", {})
            combination_key = f"{matrix_info.get('buyer_case', 'unknown')}_vs_{matrix_info.get('seller_case', 'unknown')}"
            
            if combination_key not in case_combinations:
                case_combinations[combination_key] = {
                    "buyer_case": matrix_info.get('buyer_case'),
                    "seller_case": matrix_info.get('seller_case'),
                    "buyer_llm": matrix_info.get('buyer_llm'),
                    "seller_llm": matrix_info.get('seller_llm'),
                    "runs": [],
                    "success_rate": 0,
                    "average_agreed_price": None,
                    "price_range": {"min": None, "max": None}
                }
            
            offers = run_data.get("offers_analysis", {})
            case_combinations[combination_key]["runs"].append({
                "run_id": run_id,
                "successful": offers.get("negotiation_successful", False),
                "initial_buyer_offer": offers.get("initial_buyer_offer"),
                "initial_seller_offer": offers.get("initial_seller_offer"),
                "final_buyer_offer": offers.get("final_buyer_offer"),
                "final_seller_offer": offers.get("final_seller_offer"),
                "agreed_price": offers.get("agreed_price"),
                "termination_reason": run_data.get("termination_reason"),
                "round_count": run_data.get("round_count"),
                "message_count": len(run_data.get("messages", []))
            })
        
        # Calculate aggregated metrics for each case combination
        for combo_key, combo_data in case_combinations.items():
            successful_runs = [run for run in combo_data["runs"] if run["successful"]]
            combo_data["success_rate"] = (len(successful_runs) / len(combo_data["runs"]) * 100) if combo_data["runs"] else 0
            
            agreed_prices = [run["agreed_price"] for run in successful_runs if run["agreed_price"]]
            if agreed_prices:
                combo_data["average_agreed_price"] = sum(agreed_prices) / len(agreed_prices)
                combo_data["price_range"]["min"] = min(agreed_prices)
                combo_data["price_range"]["max"] = max(agreed_prices)
        
        results_package["aggregated_analysis"] = case_combinations
        
        # Return JSON with appropriate headers for download
        from flask import Response
        response = Response(
            json.dumps(results_package, indent=2, default=str),
            mimetype='application/json',
            headers={
                'Content-Disposition': f'attachment; filename=batch_results_{batch_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            }
        )
        
        return response
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Configuration Management Endpoints
@app.route('/api/save-config', methods=['POST'])
def save_config():
    try:
        data = request.get_json()
        run_manager.save_configuration(data)
        return jsonify({"success": True, "message": "Configuration saved successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/load-config', methods=['GET'])
def load_config():
    try:
        config = run_manager.load_configuration()
        return jsonify(config)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# API Keys Status Endpoint
@app.route('/api/api-keys-status', methods=['GET'])
def api_keys_status():
    """Check which API keys are configured in the frontend"""
    configured_keys = {
        'openai': bool(run_manager.api_keys.get('openai', '').strip()),
        'anthropic': bool(run_manager.api_keys.get('anthropic', '').strip()),
        'xai': bool(run_manager.api_keys.get('xai', '').strip()),
        'groq': bool(run_manager.api_keys.get('groq', '').strip())
    }
    
    return jsonify({
        "configured_keys": configured_keys,
        "total_configured": sum(configured_keys.values()),
        "missing_keys": [k for k, v in configured_keys.items() if not v],
        "ready_for_execution": sum(configured_keys.values()) > 0
    })

# Debug Endpoint for Execution Settings
@app.route('/api/debug/execution-settings', methods=['GET'])
def debug_execution_settings():
    """Debug endpoint to check execution settings"""
    active_runs = {}
    for run_id, run in run_manager.runs.items():
        if hasattr(run, 'execution_settings'):
            active_runs[run_id] = run.to_dict().get('execution_settings', {})
        else:
            active_runs[run_id] = {
                "max_rounds": getattr(run, 'max_rounds', 'unknown'),
                "turn_delay": getattr(run, 'turn_delay', 'unknown'),
                "current_turn": getattr(run, 'current_turn', 'unknown'),
                "termination_prompt": getattr(run, 'termination_prompt', 'unknown'),
                "termination_reason": getattr(run, 'termination_reason', 'unknown')
            }
    
    return jsonify({
        "total_active_runs": len(run_manager.runs),
        "active_runs_settings": active_runs,
        "batch_executions": len(run_manager.batch_executions)
    })

# Debug endpoint for termination testing
@app.route('/api/debug/test-termination', methods=['POST'])
def test_termination():
    """Test endpoint for custom termination logic"""
    data = request.get_json()
    prompt = data.get('termination_prompt', '')
    messages = data.get('messages', [])
    
    if not prompt or not messages:
        return jsonify({"error": "termination_prompt and messages are required"}), 400
    
    # Create a temporary run for testing
    temp_run = type('TempRun', (), {})()
    temp_run.termination_prompt = prompt
    temp_run.messages = messages
    temp_run.agent1_provider = None
    temp_run.agent2_provider = None
    
    # Get first available provider from run manager
    for provider_name in ['openai', 'anthropic', 'xai', 'groq']:
        if run_manager.api_keys.get(provider_name, '').strip():
            if provider_name == "openai":
                temp_run.agent1_provider = OpenAIProvider(run_manager.api_keys[provider_name])
            elif provider_name == "anthropic":
                temp_run.agent1_provider = AnthropicProvider(run_manager.api_keys[provider_name])
            elif provider_name == "xai":
                temp_run.agent1_provider = XAIProvider(run_manager.api_keys[provider_name])
            elif provider_name == "groq":
                temp_run.agent1_provider = GroqProvider(run_manager.api_keys[provider_name])
            break
    
    if not temp_run.agent1_provider:
        return jsonify({"error": "No LLM provider available for testing"}), 400
    
    # Test termination logic
    try:
        # Use the same logic as CompetitionRun
        should_terminate, reason = CompetitionRun._should_terminate(temp_run)
        return jsonify({
            "should_terminate": should_terminate,
            "reason": reason,
            "prompt_used": prompt
        })
    except Exception as e:
        return jsonify({"error": f"Test failed: {str(e)}"}), 500

if __name__ == "__main__":
    """
    Main entry point for the Epistemic Grounding Experiment platform.
    """
    print("🧠 Starting Epistemic Grounding Experiment Platform...")
    print(f"📊 Environment: {os.environ.get('FLASK_ENV', 'development')}")
    print(f"🌐 Access the platform at: http://localhost:{app_config.PORT}")
    print("⚙️  Configure API keys through the web interface or .env file")
    print("📚 Documentation: https://github.com/ayushi-uwc/epistemic_grounding_experiment")
    
    try:
        app.run(
            host=getattr(app_config, 'HOST', '0.0.0.0'),
            port=getattr(app_config, 'PORT', 5001),
            debug=getattr(app_config, 'DEBUG', True)
        )
    except KeyboardInterrupt:
        print("\n👋 Shutdown requested by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        print("💡 Check your configuration and try again") 