"""
Main evaluation script for SOBACO-EVAL
Evaluates LLMs on social bias and cultural awareness datasets
"""

import argparse
import json
import os
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
import requests
import asyncio
from tqdm.asyncio import tqdm as async_tqdm
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file if it exists
except ImportError:
    pass  # python-dotenv not installed, skip

# Optional API imports
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import google.generativeai as genai
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

from utils import (
    load_config, load_dataset, format_prompt, extract_answer,
    calculate_metrics, print_metrics, create_output_dir,
    get_dataset_language, parse_options
)

warnings.filterwarnings('ignore')


class LLMEvaluator:
    """Base class for LLM evaluation"""
    
    def __init__(self, model_name: str, config: Dict):
        self.model_name = model_name
        self.config = config
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load model - to be implemented by subclasses"""
        raise NotImplementedError
        
    def generate(self, prompt: str, system_prompt: str = None) -> str:
        """Generate response - to be implemented by subclasses"""
        raise NotImplementedError
        
    def unload_model(self):
        """Unload model from memory"""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
            del self.tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class LocalModelEvaluator(LLMEvaluator):
    """Evaluator for local Hugging Face models"""
    
    def load_model(self):
        """Load local model from Hugging Face"""
        model_config = self.config['local_models'][self.model_name]
        model_path = model_config['model_name']
        
        print(f"Loading model: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.tokenizer.padding_side = 'left'
        
        load_kwargs = {
            'pretrained_model_name_or_path': model_path,
            'device_map': model_config.get('device_map', 'auto'),
            'torch_dtype': torch.float16,
        }
        
        if model_config.get('load_in_8bit', False):
            load_kwargs['load_in_8bit'] = True
        
        self.model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
        print(f"Model loaded successfully!")
        
    def generate(self, prompt: str, system_prompt: str = None) -> str:
        """Generate response using local model"""
        model_config = self.config['local_models'][self.model_name]
        
        # Check if model has chat template (Instruct models)
        has_chat_template = hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template is not None
        
        if has_chat_template:
            # Format messages for chat models
            # Only include user message, no system prompt
            messages = [{"role": "user", "content": prompt}]
            
            # Apply chat template
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        else:
            # Base model without chat template - use prompt directly
            formatted_prompt = prompt
        
        # Tokenize with truncation and padding to prevent CUDA errors
        inputs = self.tokenizer(
            formatted_prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=2048,  # Prevent overly long inputs
            padding=False
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=model_config.get('max_tokens', 100),
                temperature=model_config.get('temperature', 0.1),
                top_p=model_config.get('top_p', 0.95),
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode only the generated part
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return response.strip()
    
    def generate_batch(self, prompts: List[str], system_prompt: str = None) -> List[str]:
        """Generate responses for multiple prompts using batching"""
        model_config = self.config['local_models'][self.model_name]
        
        # Check if model has chat template (Instruct models)
        has_chat_template = hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template is not None
        
        formatted_prompts = []
        for prompt in prompts:
            if has_chat_template:
                # Format messages for chat models
                messages = [{"role": "user", "content": prompt}]
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
            else:
                # Base model without chat template
                formatted_prompt = prompt
            formatted_prompts.append(formatted_prompt)
        
        # Tokenize batch with padding
        inputs = self.tokenizer(
            formatted_prompts, 
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            padding=True  # Enable padding for batch processing
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=model_config.get('max_tokens', 100),
                temperature=model_config.get('temperature', 0.1),
                top_p=model_config.get('top_p', 0.95),
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode each response in the batch
        responses = []
        for i, output in enumerate(outputs):
            # Decode only the generated part (skip input tokens)
            input_length = inputs['input_ids'][i].shape[0]
            generated_ids = output[input_length:]
            response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            responses.append(response.strip())
        
        return responses


class APIModelEvaluator(LLMEvaluator):
    """Evaluator for API-based models"""
    
    def load_model(self):
        """Initialize API client"""
        model_config = self.config['api_models'][self.model_name]
        provider = model_config['provider']
        
        if provider == 'openai':
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI package not installed. Install with: pip install openai")
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            self.client = OpenAI(api_key=api_key)
            
        elif provider == 'anthropic':
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("Anthropic package not installed. Install with: pip install anthropic")
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable not set")
            self.client = Anthropic(api_key=api_key)
            
        elif provider == 'google':
            if not GOOGLE_AVAILABLE:
                raise ImportError("Google AI package not installed. Install with: pip install google-generativeai")
            api_key = os.getenv('GOOGLE_API_KEY')
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable not set")
            genai.configure(api_key=api_key)
            self.client = genai
            
        print(f"API client initialized for {provider}")
        
    def generate(self, prompt: str, system_prompt: str = None) -> str:
        """Generate response using API"""
        model_config = self.config['api_models'][self.model_name]
        provider = model_config['provider']
        
        try:
            if provider == 'openai':
                # Only use user message, no system prompt
                messages = [{"role": "user", "content": prompt}]
                
                response = self.client.chat.completions.create(
                    model=model_config['model_name'],
                    messages=messages,
                    #max_tokens=model_config.get('max_tokens', 100),
                    temperature=model_config.get('temperature', 0.1),
                )
                return response.choices[0].message.content.strip()
                
            elif provider == 'anthropic':
                # Only use user message, no system prompt
                response = self.client.messages.create(
                    model=model_config['model_name'],
                    max_tokens=model_config.get('max_tokens', 100),
                    temperature=model_config.get('temperature', 0.1),
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text.strip()
                
            elif provider == 'google':
                # Use prompt directly without system prompt
                model = self.client.GenerativeModel(model_config['model_name'])
                response = model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=model_config.get('max_tokens', 100),
                        temperature=model_config.get('temperature', 0.1),
                    )
                )
                return response.text.strip()
                
        except Exception as e:
            print(f"Error generating response: {e}")
            return ""

class AsyncAPIModelEvaluator(LLMEvaluator):
    """Async evaluator for API-based models with concurrent requests"""
    
    def __init__(self, model_name: str, config: Dict, max_concurrent: int = 10):
        super().__init__(model_name, config)
        self.max_concurrent = max_concurrent
        self.semaphore = None
        
    def load_model(self):
        """Initialize API client"""
        model_config = self.config['api_models'][self.model_name]
        provider = model_config['provider']
        
        if provider == 'openai':
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI package not installed. Install with: pip install openai")
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(api_key=api_key)
            
        elif provider == 'anthropic':
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("Anthropic package not installed. Install with: pip install anthropic")
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable not set")
            from anthropic import AsyncAnthropic
            self.client = AsyncAnthropic(api_key=api_key)
            
        elif provider == 'google':
            # Google AI SDK doesn't have native async support yet
            # We'll use the sync client with asyncio.to_thread
            if not GOOGLE_AVAILABLE:
                raise ImportError("Google AI package not installed. Install with: pip install google-generativeai")
            api_key = os.getenv('GOOGLE_API_KEY')
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable not set")
            genai.configure(api_key=api_key)
            self.client = genai
            
        self.semaphore = asyncio.Semaphore(self.max_concurrent)
        print(f"Async API client initialized for {provider} (max concurrent: {self.max_concurrent})")
    
    async def generate_async(self, prompt: str, system_prompt: str = None) -> str:
        """Generate response using API asynchronously"""
        model_config = self.config['api_models'][self.model_name]
        provider = model_config['provider']
        
        async with self.semaphore:
            try:
                if provider == 'openai':
                    messages = [{"role": "user", "content": prompt}]
                    
                    response = await self.client.chat.completions.create(
                        model=model_config['model_name'],
                        messages=messages,
                        temperature=model_config.get('temperature', 0.1),
                    )
                    return response.choices[0].message.content.strip()
                    
                elif provider == 'anthropic':
                    response = await self.client.messages.create(
                        model=model_config['model_name'],
                        max_tokens=model_config.get('max_tokens', 100),
                        temperature=model_config.get('temperature', 0.1),
                        messages=[{"role": "user", "content": prompt}]
                    )
                    return response.content[0].text.strip()
                    
                elif provider == 'google':
                    # Run in thread pool since Google SDK doesn't support async
                    model = self.client.GenerativeModel(model_config['model_name'])
                    response = await asyncio.to_thread(
                        model.generate_content,
                        prompt,
                        generation_config=genai.types.GenerationConfig(
                            max_output_tokens=model_config.get('max_tokens', 100),
                            temperature=model_config.get('temperature', 0.1),
                        )
                    )
                    return response.text.strip()
                    
            except Exception as e:
                print(f"\nError generating response: {e}")
                return ""
    
    def generate(self, prompt: str, system_prompt: str = None) -> str:
        """Sync wrapper for backward compatibility"""
        return asyncio.run(self.generate_async(prompt, system_prompt))


async def evaluate_dataset_async(evaluator: AsyncAPIModelEvaluator, dataset_path: str, config: Dict, subset_percent: Optional[float] = None) -> pd.DataFrame:
    """
    Evaluate model on a dataset asynchronously
    
    Args:
        evaluator: AsyncAPIModelEvaluator instance
        dataset_path: Path to CSV dataset
        config: Configuration dictionary
        subset_percent: If provided, only evaluate on this percentage of the dataset
    
    Returns:
        DataFrame with predictions and results
    """
    # Load dataset
    df = load_dataset(dataset_path)
    
    # Use subset if specified
    if subset_percent is not None:
        original_size = len(df)
        subset_size = max(1, int(original_size * subset_percent))
        df = df.head(subset_size).copy()
        print(f"Using subset: {subset_size} samples ({subset_percent*100:.0f}% of {original_size})")
    
    # Determine target language for prompt selection
    language = get_dataset_language(dataset_path)
    
    # Get all available format templates for the target language
    template_variants = []
    for i in range(1, 4):
        template_key = f"format_template_{language}_{i}"
        if template_key in config['prompts']:
            template_variants.append(config['prompts'][template_key])
    
    if not template_variants:
        raise ValueError(f"No format templates found for language '{language}' in config prompts")
    else:
        prompt_template = random.choice(template_variants)
        print(f"Using {language} format template (randomly selected from {len(template_variants)} variant(s))")
    
    # Prepare all prompts
    prompts = []
    options_list = []
    for idx, row in df.iterrows():
        prompt = format_prompt(row, prompt_template, language)
        prompts.append((idx, prompt))
        options_list.append((idx, parse_options(row['options'])))
    
    # Add columns for predictions
    df['prediction'] = None
    df['raw_response'] = None
    
    # Create tasks for concurrent evaluation
    async def process_sample(idx: int, prompt: str, options: List[str]):
        try:
            response = await evaluator.generate_async(prompt)
            prediction = extract_answer(response, options)
            return idx, response, prediction
        except Exception as e:
            print(f"\nError at index {idx}: {e}")
            return idx, f"ERROR: {str(e)}", None
    
    # Create all tasks
    tasks = [
        process_sample(idx, prompt, options)
        for (idx, prompt), (_, options) in zip(prompts, options_list)
    ]
    
    # Execute tasks with progress bar
    results = []
    for coro in async_tqdm(
        asyncio.as_completed(tasks),
        total=len(tasks),
        desc=f"Evaluating {evaluator.model_name}"
    ):
        result = await coro
        results.append(result)
    
    # Update dataframe with results
    for idx, raw_response, prediction in results:
        df.at[idx, 'raw_response'] = raw_response
        df.at[idx, 'prediction'] = prediction
    
    # Save final results
    if config['evaluation'].get('save_predictions', True):
        output_dir = create_output_dir(config['output']['results_dir'])
        temp_path = output_dir / f"temp_{evaluator.model_name}_{Path(dataset_path).stem}.csv"
        df.to_csv(temp_path, index=False)
    
    return df

class OllamaModelEvaluator(LLMEvaluator):
    """Evaluator for local models via Ollama API"""
    
    def __init__(self, model_name: str, config: Dict, ollama_url: str = "http://localhost:11434"):
        super().__init__(model_name, config)
        self.ollama_url = ollama_url
        self.ollama_model_name = None
        
    def load_model(self):
        """Initialize Ollama connection and verify model availability"""
        model_config = self.config['local_models'][self.model_name]
        
        # Extract model name from Hugging Face path for Ollama
        # e.g., "meta-llama/Meta-Llama-3.1-8B-Instruct" -> "llama3.1:8b-instruct"
        hf_model_path = model_config['model_name']
        
        # Try to map common HF names to Ollama names
        # User can also specify 'ollama_model_name' in config to override
        if 'ollama_model_name' in model_config:
            self.ollama_model_name = model_config['ollama_model_name']
        else:
            # Simple heuristic mapping
            self.ollama_model_name = self._map_hf_to_ollama(hf_model_path)
        
        print(f"Using Ollama API at {self.ollama_url}")
        print(f"Model: {self.ollama_model_name}")
        
        # Verify Ollama is running and model is available
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            response.raise_for_status()
            available_models = [model['name'] for model in response.json().get('models', [])]
            
            # Check if exact model name exists
            if self.ollama_model_name not in available_models:
                print(f"Warning: Model '{self.ollama_model_name}' not found in Ollama.")
                print(f"Available models: {available_models}")
                print(f"Attempting to use anyway (Ollama might pull it automatically)")
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Could not connect to Ollama at {self.ollama_url}. "
                                f"Make sure Ollama is running. Error: {e}")
    
    def _map_hf_to_ollama(self, hf_path: str) -> str:
        """Map Hugging Face model path to Ollama model name"""
        hf_path_lower = hf_path.lower()
        
        # Common mappings
        if 'llama-3.1-8b-instruct' in hf_path_lower:
            return 'llama3.1:8b-instruct'
        elif 'llama-3.1-8b' in hf_path_lower:
            return 'llama3.1:8b'
        elif 'llama-3.1-70b-instruct' in hf_path_lower:
            return 'llama3.1:70b-instruct'
        elif 'llama-3.1-70b' in hf_path_lower:
            return 'llama3.1:70b'
        elif 'qwen3-4b' in hf_path_lower:
            return 'qwen3:4b-instruct'
        elif 'qwen3-30b' in hf_path_lower:
            return 'qwen3:30b-instruct'
        else:
            # Fallback: use the last part of the path
            model_name = hf_path.split('/')[-1].lower()
            print(f"Warning: Could not auto-map '{hf_path}' to Ollama model.")
            print(f"Using '{model_name}' - you may need to add 'ollama_model_name' to config.yaml")
            return model_name
    
    def generate(self, prompt: str, system_prompt: str = None) -> str:
        """Generate response using Ollama API"""
        model_config = self.config['local_models'][self.model_name]
        
        # Prepare request payload
        payload = {
            "model": self.ollama_model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": model_config.get('temperature', 0.1),
                "top_p": model_config.get('top_p', 0.95),
                "num_predict": model_config.get('max_tokens', 100),
            }
        }
        
        try:
            # Ensure URL doesn't have double slashes
            url = f"{self.ollama_url.rstrip('/')}/api/generate"
            
            response = requests.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=120  # 2 minute timeout for generation
            )
            
            # Debug output on failure
            if response.status_code != 200:
                print(f"Ollama API Error - Status: {response.status_code}")
                print(f"URL: {url}")
                print(f"Payload: {payload}")
                print(f"Response: {response.text}")
            
            response.raise_for_status()
            result = response.json()
            return result.get('response', '').strip()
            
        except requests.exceptions.RequestException as e:
            print(f"Error calling Ollama API: {e}")
            print(f"URL: {url}")
            print(f"Model: {self.ollama_model_name}")
            return ""
    
    def unload_model(self):
        """No need to unload for API-based models"""
        print(f"Finished using Ollama model: {self.ollama_model_name}")


def evaluate_dataset(evaluator: LLMEvaluator, dataset_path: str, config: Dict, subset_percent: Optional[float] = None, batch_size: int = 1) -> pd.DataFrame:
    """
    Evaluate model on a dataset
    
    Args:
        evaluator: LLMEvaluator instance
        dataset_path: Path to CSV dataset
        config: Configuration dictionary
        subset_percent: If provided, only evaluate on this percentage of the dataset (e.g., 0.1 for 10%)
        batch_size: Batch size for local models (default: 1, no batching)
    
    Returns:
        DataFrame with predictions and results
    """
    # Load dataset
    df = load_dataset(dataset_path)
    
    # Use subset if specified
    if subset_percent is not None:
        original_size = len(df)
        subset_size = max(1, int(original_size * subset_percent))
        df = df.head(subset_size).copy()
        print(f"Using subset: {subset_size} samples ({subset_percent*100:.0f}% of {original_size})")
    
    # Determine target language for prompt selection
    # For translated datasets (e.g., ko-jp), use the target language (ja)
    language = get_dataset_language(dataset_path)
    
    # Get all available format templates for the target language
    template_variants = []
    for i in range(1, 4):  # Try templates 1, 2, 3
        template_key = f"format_template_{language}_{i}"
        if template_key in config['prompts']:
            template_variants.append(config['prompts'][template_key])
    
    # If no language-specific templates found, try default template
    if not template_variants:
        raise ValueError(f"No format templates found for language '{language}' in config prompts")
    else:
        # Randomly select one of the available templates
        prompt_template = random.choice(template_variants)
        print(f"Using {language} format template (randomly selected from {len(template_variants)} variant(s))")
    
    # No system prompt is used
    system_prompt = None
    
    # Add columns for predictions
    df['prediction'] = None
    df['raw_response'] = None
    
    # Evaluation settings
    save_interval = config['evaluation'].get('save_interval', 100)
    output_dir = create_output_dir(config['output']['results_dir'])
    
    # Check if we can use batching (only for LocalModelEvaluator)
    use_batching = isinstance(evaluator, LocalModelEvaluator) and batch_size > 1
    
    if use_batching:
        print(f"Using batch processing with batch_size={batch_size}")
        
        # Prepare all prompts and options
        all_prompts = []
        all_options = []
        all_indices = []
        
        for idx, row in df.iterrows():
            prompt = format_prompt(row, prompt_template, language)
            options = parse_options(row['options'])
            all_prompts.append(prompt)
            all_options.append(options)
            all_indices.append(idx)
        
        # Process in batches
        for i in tqdm(range(0, len(all_prompts), batch_size), desc=f"Evaluating {evaluator.model_name}"):
            batch_prompts = all_prompts[i:i+batch_size]
            batch_options = all_options[i:i+batch_size]
            batch_indices = all_indices[i:i+batch_size]
            
            try:
                # Generate batch responses
                responses = evaluator.generate_batch(batch_prompts, system_prompt)
                
                # Process each response in batch
                for idx, response, options in zip(batch_indices, responses, batch_options):
                    df.at[idx, 'raw_response'] = response
                    prediction = extract_answer(response, options)
                    df.at[idx, 'prediction'] = prediction
                    
            except Exception as e:
                print(f"\nError in batch starting at index {batch_indices[0]}: {e}")
                # Fall back to individual processing for this batch
                for idx, prompt, options in zip(batch_indices, batch_prompts, batch_options):
                    try:
                        response = evaluator.generate(prompt, system_prompt)
                        df.at[idx, 'raw_response'] = response
                        prediction = extract_answer(response, options)
                        df.at[idx, 'prediction'] = prediction
                    except Exception as e2:
                        print(f"\nError at index {idx}: {e2}")
                        df.at[idx, 'raw_response'] = f"ERROR: {str(e2)}"
                        df.at[idx, 'prediction'] = None
            
            # Save intermediate results
            if config['evaluation'].get('save_predictions', True) and (i + batch_size) % save_interval < batch_size:
                temp_path = output_dir / f"temp_{evaluator.model_name}_{Path(dataset_path).stem}.csv"
                df.to_csv(temp_path, index=False)
    else:
        # Original sequential processing (for API models or batch_size=1)
        if batch_size > 1:
            print(f"Batching not supported for {type(evaluator).__name__}, using sequential processing")
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Evaluating {evaluator.model_name}"):
            # Format prompt
            prompt = format_prompt(row, prompt_template, language)
            
            # Generate response
            try:
                response = evaluator.generate(prompt, system_prompt)
                df.at[idx, 'raw_response'] = response
                
                # Extract answer
                options = parse_options(row['options'])
                prediction = extract_answer(response, options)
                df.at[idx, 'prediction'] = prediction
                
            except Exception as e:
                print(f"\nError at index {idx}: {e}")
                df.at[idx, 'raw_response'] = f"ERROR: {str(e)}"
                df.at[idx, 'prediction'] = None
            
            # Save intermediate results
            if config['evaluation'].get('save_predictions', True) and (idx + 1) % save_interval == 0:
                temp_path = output_dir / f"temp_{evaluator.model_name}_{Path(dataset_path).stem}.csv"
                df.to_csv(temp_path, index=False)
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLMs on SOBACO datasets")
    parser.add_argument('--model', nargs='+', required=True, help='Model name(s) to evaluate')
    parser.add_argument('--dataset', type=str, help='Path to specific dataset CSV')
    parser.add_argument('--all-datasets', action='store_true', help='Evaluate on all datasets in csv/')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--output-dir', type=str, help='Override output directory')
    parser.add_argument('--subset', type=float, default=None, 
                       help='Evaluate on a subset of the data (e.g., 0.1 for 10%%, 0.2 for 20%%)')
    parser.add_argument('--use-ollama', action='store_true', 
                       help='Use Ollama API for local models instead of loading them directly')
    parser.add_argument('--ollama-url', type=str, default='http://localhost:11434',
                       help='Ollama server URL (default: http://localhost:11434)')
    parser.add_argument('--async-api', action='store_true',
                       help='Use async evaluation for API models (faster with concurrent requests)')
    parser.add_argument('--max-concurrent', type=int, default=10,
                       help='Maximum concurrent API requests for async mode (default: 10)')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size for local model evaluation (default: 1, no batching). Higher values improve GPU utilization.')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    if args.output_dir:
        config['output']['results_dir'] = args.output_dir
    
    # Determine datasets to evaluate
    datasets = []
    if args.all_datasets:
        csv_dir = Path('csv')
        datasets = list(csv_dir.glob('*.csv'))
    elif args.dataset:
        datasets = [Path(args.dataset)]
    else:
        print("Error: Please specify --dataset or --all-datasets")
        return
    
    print(f"Will evaluate on {len(datasets)} dataset(s): {[d.name for d in datasets]}")
    print(f"Models to evaluate: {args.model}")
    if args.use_ollama:
        print(f"Using Ollama API at {args.ollama_url}")
    if args.async_api:
        print(f"Using async API evaluation with max {args.max_concurrent} concurrent requests")
    
    # Evaluate each model on each dataset
    for model_name in args.model:
        print(f"\n{'='*60}")
        print(f"Evaluating model: {model_name}")
        print(f"{'='*60}")
        
        # Determine if local or API model
        is_api_model = model_name in config.get('api_models', {})
        is_local_model = model_name in config.get('local_models', {})
        
        if not is_api_model and not is_local_model:
            print(f"Error: Model '{model_name}' not found in config")
            continue
        
        # Create appropriate evaluator
        if is_api_model and args.async_api:
            evaluator = AsyncAPIModelEvaluator(model_name, config, args.max_concurrent)
            use_async = True
        elif is_api_model:
            evaluator = APIModelEvaluator(model_name, config)
            use_async = False
        elif is_local_model and args.use_ollama:
            evaluator = OllamaModelEvaluator(model_name, config, args.ollama_url)
            use_async = False
        else:
            evaluator = LocalModelEvaluator(model_name, config)
            use_async = False
        
        # Load model
        try:
            evaluator.load_model()
        except Exception as e:
            print(f"Error loading model: {e}")
            continue
        
        # Evaluate on each dataset
        for dataset_path in datasets:
            print(f"\nEvaluating on: {dataset_path.name}")
            
            try:
                # Run evaluation (async or sync)
                if use_async:
                    results_df = asyncio.run(evaluate_dataset_async(evaluator, str(dataset_path), config, args.subset))
                else:
                    results_df = evaluate_dataset(evaluator, str(dataset_path), config, args.subset, args.batch_size)
                
                # Calculate metrics
                metrics = calculate_metrics(results_df)
                
                # Print results
                print_metrics(metrics, f"{model_name} on {dataset_path.stem}")
                
                # Save results
                output_dir = create_output_dir(config['output']['results_dir'])
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if config['output'].get('create_timestamp', True) else ""
                base_name = f"{model_name}_{dataset_path.stem}"
                if timestamp:
                    base_name = f"{base_name}_{timestamp}"
                
                # Save CSV
                csv_path = output_dir / f"{base_name}.csv"
                results_df.to_csv(csv_path, index=False)
                print(f"Results saved to: {csv_path}")
                
                # Save metrics JSON
                json_path = output_dir / f"{base_name}_metrics.json"
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        'model': model_name,
                        'dataset': str(dataset_path),
                        'timestamp': datetime.now().isoformat(),
                        'metrics': metrics
                    }, f, indent=2, ensure_ascii=False)
                print(f"Metrics saved to: {json_path}")
                
            except Exception as e:
                print(f"Error evaluating {dataset_path.name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Unload model to free memory
        evaluator.unload_model()
    
    print(f"\n{'='*60}")
    print("Evaluation complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
