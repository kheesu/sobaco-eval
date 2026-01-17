"""
Main evaluation script for SOBACO-EVAL
Evaluates LLMs on social bias and cultural awareness datasets using vLLM and OpenAI
"""

import argparse
import json
import os
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import warnings
import asyncio
from tqdm.asyncio import tqdm as async_tqdm

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("Warning: vLLM not installed. Local execution will fail.")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from openai import OpenAI, AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from utils import (
    load_config, load_dataset, format_prompt, extract_answer,
    calculate_metrics, print_metrics, create_output_dir,
    get_dataset_language, parse_options
)

warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


class LLMEvaluator:
    """Base class for LLM evaluation"""
    
    def __init__(self, model_name: str, config: Dict):
        self.model_name = model_name
        self.config = config
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        raise NotImplementedError
        
    def generate(self, prompt: str, system_prompt: str = None) -> str:
        raise NotImplementedError
        
    def unload_model(self):
        """Unload model from memory"""
        if hasattr(self, 'model') and self.model is not None:
            import gc
            del self.model
            if hasattr(self, 'tokenizer'):
                del self.tokenizer
            self.model = None
            self.tokenizer = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class VLLMModelEvaluator(LLMEvaluator):
    """Evaluator using vLLM for high-throughput local inference"""
    
    def load_model(self):
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM is not installed. Please install with: pip install vllm")

        model_config = self.config['local_models'][self.model_name]
        model_path = model_config['model_name']
        
        tensor_parallel_size = self.config.get('tensor_parallel_size', 1)
        dtype = self.config.get('dtype', 'float16')
        gpu_memory_utilization = self.config.get('gpu_memory_utilization', 0.9)
        
        print(f"Loading vLLM model: {model_path}")
        print(f"Configuration: TP={tensor_parallel_size}, Dtype={dtype}")

        self.model = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            enforce_eager=False, 
        )
        
        self.tokenizer = self.model.get_tokenizer()

    def generate_batch(self, prompts: List[str], system_prompt: str = None) -> List[str]:
        """
        Optimized generation for MQA (Single Token Output).
        """
        model_config = self.config['local_models'][self.model_name]
        
        has_chat_template = hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template is not None
        
        final_prompts = []
        for prompt in prompts:
            if has_chat_template:
                messages = [{"role": "user", "content": prompt}]
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                final_prompts.append(formatted_prompt)
            else:
                final_prompts.append(prompt)

        stop_token_ids = [self.tokenizer.eos_token_id]
        if self.tokenizer.pad_token_id is not None:
            stop_token_ids.append(self.tokenizer.pad_token_id)
            
        if hasattr(self.tokenizer, "convert_tokens_to_ids"):
            try:
                eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                if isinstance(eot_id, int) and eot_id not in stop_token_ids:
                    stop_token_ids.append(eot_id)
            except:
                pass

        sampling_params = SamplingParams(
            temperature=0.0,      
            top_p=1.0,            
            max_tokens=1,        
            stop_token_ids=stop_token_ids,
            
            # OPTIONAL: Guided Decoding
            # guided_choice=["A", "B", "C", "D", "a", "b", "c", "d"],
        )

        outputs = self.model.generate(final_prompts, sampling_params, use_tqdm=False)

        results = []
        for output in outputs:
            generated_text = output.outputs[0].text.strip()
            results.append(generated_text)
            
        return results

    def generate(self, prompt: str, system_prompt: str = None) -> str:
        """Wrapper for single generation"""
        return self.generate_batch([prompt], system_prompt)[0]


class APIModelEvaluator(LLMEvaluator):
    """Evaluator for OpenAI API"""
    
    def load_model(self):
        """Initialize OpenAI client"""
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not installed. Install with: pip install openai")
        
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
            
        self.client = OpenAI(api_key=api_key)
        
    def generate(self, prompt: str, system_prompt: str = None) -> str:
        """Generate response using OpenAI"""
        model_config = self.config['api_models'][self.model_name]
        
        try:
            messages = [{"role": "user", "content": prompt}]
            
            response = self.client.chat.completions.create(
                model=model_config['model_name'],
                messages=messages,
                temperature=model_config.get('temperature', 0.1),
            )
            return response.choices[0].message.content.strip()
                
        except Exception as e:
            print(f"Error generating response: {e}")
            return ""


class AsyncAPIModelEvaluator(LLMEvaluator):
    """Async evaluator for OpenAI API"""
    
    def __init__(self, model_name: str, config: Dict, max_concurrent: int = 10):
        super().__init__(model_name, config)
        self.max_concurrent = max_concurrent
        self.semaphore = None
        
    def load_model(self):
        """Initialize Async OpenAI client"""
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not installed. Install with: pip install openai")
            
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
            
        self.client = AsyncOpenAI(api_key=api_key)
        self.semaphore = asyncio.Semaphore(self.max_concurrent)
    
    async def generate_async(self, prompt: str, system_prompt: str = None) -> str:
        """Generate response using API asynchronously"""
        model_config = self.config['api_models'][self.model_name]
        
        async with self.semaphore:
            try:
                messages = [{"role": "user", "content": prompt}]
                
                response = await self.client.chat.completions.create(
                    model=model_config['model_name'],
                    messages=messages,
                    temperature=model_config.get('temperature', 0.1),
                )
                return response.choices[0].message.content.strip()
                    
            except Exception as e:
                print(f"\nError generating response: {e}")
                return ""
    
    def generate(self, prompt: str, system_prompt: str = None) -> str:
        """Sync wrapper"""
        return asyncio.run(self.generate_async(prompt, system_prompt))


async def evaluate_dataset_async(evaluator: AsyncAPIModelEvaluator, dataset_path: str, config: Dict, subset_percent: Optional[float] = None, template_index: Optional[int] = None) -> pd.DataFrame:
    """
    Evaluate model on a dataset asynchronously (For API models)
    """
    df = load_dataset(dataset_path)
    
    if subset_percent is not None:
        original_size = len(df)
        subset_size = max(1, int(original_size * subset_percent))
        df = df.head(subset_size).copy()
        print(f"Using subset: {subset_size} samples")
    
    language = get_dataset_language(dataset_path)
    
    # Template selection logic
    template_variants = []
    for i in range(1, 4):
        template_key = f"format_template_{language}_{i}"
        if template_key in config['prompts']:
            template_variants.append(config['prompts'][template_key])
    
    if not template_variants:
        raise ValueError(f"No format templates found for language '{language}'")
    
    if template_index is not None:
        prompt_template = template_variants[template_index - 1]
    else:
        prompt_template = random.choice(template_variants)
    
    # Prepare prompts
    prompts = []
    options_list = []
    for idx, row in df.iterrows():
        prompt = format_prompt(row, prompt_template, language)
        prompts.append((idx, prompt))
        options_list.append((idx, parse_options(row['options'])))
    
    df['prediction'] = None
    df['raw_response'] = None
    
    async def process_sample(idx: int, prompt: str, options: List[str]):
        try:
            response = await evaluator.generate_async(prompt)
            prediction = extract_answer(response, options)
            return idx, response, prediction
        except Exception as e:
            return idx, f"ERROR: {str(e)}", None
    
    tasks = [
        process_sample(idx, prompt, options)
        for (idx, prompt), (_, options) in zip(prompts, options_list)
    ]
    
    results = []
    for coro in async_tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"Evaluating {evaluator.model_name}"):
        result = await coro
        results.append(result)
    
    for idx, raw_response, prediction in results:
        df.at[idx, 'raw_response'] = raw_response
        df.at[idx, 'prediction'] = prediction
        
    return df


def evaluate_dataset(evaluator: LLMEvaluator, dataset_path: str, config: Dict, subset_percent: Optional[float] = None, batch_size: int = 16, template_index: Optional[int] = None) -> pd.DataFrame:
    """
    Evaluate model on a dataset.
    For VLLMModelEvaluator, this handles batch processing logic.
    """
    df = load_dataset(dataset_path)
    
    if subset_percent is not None:
        original_size = len(df)
        subset_size = max(1, int(original_size * subset_percent))
        df = df.head(subset_size).copy()
        print(f"Using subset: {subset_size} samples")
    
    language = get_dataset_language(dataset_path)
    
    # Template logic
    template_variants = []
    for i in range(1, 4):
        template_key = f"format_template_{language}_{i}"
        if template_key in config['prompts']:
            template_variants.append(config['prompts'][template_key])
            
    if not template_variants:
        raise ValueError(f"No format templates found for language '{language}'")
    
    if template_index is not None:
        prompt_template = template_variants[template_index - 1]
    else:
        prompt_template = random.choice(template_variants)
    
    df['prediction'] = None
    df['raw_response'] = None
    
    # Determine if we use batch processing (vLLM)
    is_vllm = isinstance(evaluator, VLLMModelEvaluator)
    
    all_prompts = []
    all_options = []
    all_indices = []
    
    for idx, row in df.iterrows():
        prompt = format_prompt(row, prompt_template, language)
        options = parse_options(row['options'])
        all_prompts.append(prompt)
        all_options.append(options)
        all_indices.append(idx)
    
    save_interval = config['evaluation'].get('save_interval', 100)
    output_dir = create_output_dir(config['output']['results_dir'])

    if is_vllm:
        # For vLLM, we can use a larger batch size (or "chunk size") 
        # because vLLM handles the actual micro-batching internally.
        # We process in chunks to avoid creating massive lists if the dataset is huge.
        chunk_size = max(batch_size, 100) # Ensure chunk is at least 100 or user specified
        print(f"Processing with vLLM (Chunk size: {chunk_size})...")
        
        for i in tqdm(range(0, len(all_prompts), chunk_size), desc=f"Evaluating {evaluator.model_name}"):
            chunk_prompts = all_prompts[i:i+chunk_size]
            chunk_options = all_options[i:i+chunk_size]
            chunk_indices = all_indices[i:i+chunk_size]
            
            try:
                responses = evaluator.generate_batch(chunk_prompts)
                
                for idx, response, options in zip(chunk_indices, responses, chunk_options):
                    df.at[idx, 'raw_response'] = response
                    prediction = extract_answer(response, options)
                    df.at[idx, 'prediction'] = prediction
                    
            except Exception as e:
                print(f"Error in vLLM batch: {e}")
                # Fallback? vLLM typically fails hard if OOM, so standard loop won't help much.
    
    else:
        # Sequential processing for APIs (if not using Async)
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Evaluating {evaluator.model_name}"):
            prompt = format_prompt(row, prompt_template, language)
            try:
                response = evaluator.generate(prompt)
                df.at[idx, 'raw_response'] = response
                
                options = parse_options(row['options'])
                prediction = extract_answer(response, options)
                df.at[idx, 'prediction'] = prediction
            except Exception as e:
                print(f"Error at index {idx}: {e}")

            if config['evaluation'].get('save_predictions', True) and (idx + 1) % save_interval == 0:
                 temp_path = output_dir / f"temp_{evaluator.model_name}_{Path(dataset_path).stem}.csv"
                 df.to_csv(temp_path, index=False)

    return df


def evaluate_on_all_templates(evaluator: LLMEvaluator, dataset_path: str, config: Dict, 
                             subset_percent: Optional[float] = None, batch_size: int = 16,
                             use_async: bool = False) -> Dict:
    """
    Evaluate model on all three templates and return averaged results
    """
    language = get_dataset_language(dataset_path)
    num_templates = 0
    for i in range(1, 4):
        if f"format_template_{language}_{i}" in config['prompts']:
            num_templates += 1
    
    print(f"\nEvaluating on all {num_templates} templates for {dataset_path}")
    
    template_results = {}
    template_metrics = {}
    
    for template_idx in range(1, num_templates + 1):
        print(f"\n--- Template {template_idx}/{num_templates} ---")
        try:
            if use_async:
                results_df = asyncio.run(evaluate_dataset_async(
                    evaluator, str(dataset_path), config, subset_percent, template_index=template_idx
                ))
            else:
                results_df = evaluate_dataset(
                    evaluator, str(dataset_path), config, subset_percent, batch_size, template_index=template_idx
                )
            
            metrics = calculate_metrics(results_df)
            template_results[f"template_{template_idx}"] = results_df
            template_metrics[f"template_{template_idx}"] = metrics
            
            print_metrics(metrics, f"{evaluator.model_name} (Template {template_idx})")
            
        except Exception as e:
            print(f"Error evaluating template {template_idx}: {e}")
            continue
            
    # Calculate averages
    if template_metrics:
        print("\n--- AVERAGED RESULTS ---")
        averaged_metrics = {}
        # Get keys from first available metric
        metric_keys = list(template_metrics[list(template_metrics.keys())[0]].keys())
        
        # Helper to recurse and average
        def get_avg(keys, metrics_dict_list):
            res = {}
            for k in keys:
                vals = [m[k] for m in metrics_dict_list if k in m]
                if not vals: continue
                if isinstance(vals[0], dict):
                    res[k] = get_avg(vals[0].keys(), vals)
                elif isinstance(vals[0], (int, float)):
                    res[k] = sum(vals) / len(vals)
                else:
                    res[k] = vals[0]
            return res

        values_list = list(template_metrics.values())
        averaged_metrics = get_avg(metric_keys, values_list)
        
        print_metrics(averaged_metrics, f"{evaluator.model_name} (AVERAGED)")
        template_metrics['averaged'] = averaged_metrics

    return {'results': template_results, 'metrics': template_metrics}


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLMs on SOBACO datasets")
    parser.add_argument('--model', nargs='+', required=True, help='Model name(s) to evaluate')
    parser.add_argument('--dataset', type=str, help='Path to specific dataset CSV')
    parser.add_argument('--all-datasets', action='store_true', help='Evaluate on all datasets in csv/')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--output-dir', type=str, help='Override output directory')
    parser.add_argument('--subset', type=float, default=None, help='Evaluate on a subset (0.1 = 10%%)')
    parser.add_argument('--async-api', action='store_true', help='Use async evaluation for API models')
    parser.add_argument('--max-concurrent', type=int, default=10, help='Max concurrent API requests')
    parser.add_argument('--batch-size', type=int, default=1000, 
                       help='Chunk size for vLLM processing. Higher is better for vLLM.')
    parser.add_argument('--tensor-parallel-size', type=int, default=1, 
                       help='Number of GPUs to use per model instance (vLLM)')
    parser.add_argument('--dtype', type=str, default='float16', 
                       help='Data type for weights (float16, bfloat16, auto)')
    
    parser.add_argument('--all-templates', action='store_true', default=True, help='Evaluate all templates')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    config = load_config(args.config)
    if args.output_dir:
        config['output']['results_dir'] = args.output_dir
    
    # Inject vLLM args into config for the evaluator to access
    config['tensor_parallel_size'] = args.tensor_parallel_size
    config['dtype'] = args.dtype

    datasets = []
    if args.all_datasets:
        csv_dir = Path('csv')
        datasets = list(csv_dir.glob('*.csv'))
    elif args.dataset:
        datasets = [Path(args.dataset)]
    else:
        print("Error: Please specify --dataset or --all-datasets")
        return

    print(f"Evaluating {len(datasets)} dataset(s)")
    
    for model_name in args.model:
        print(f"\n{'='*60}\nEvaluating model: {model_name}\n{'='*60}")
        
        is_api_model = model_name in config.get('api_models', {})
        is_local_model = model_name in config.get('local_models', {})
        
        if not is_api_model and not is_local_model:
            print(f"Error: Model '{model_name}' not found in config")
            continue
        
        # Initialize Evaluator
        if is_api_model:
            if args.async_api:
                evaluator = AsyncAPIModelEvaluator(model_name, config, args.max_concurrent)
                use_async = True
            else:
                evaluator = APIModelEvaluator(model_name, config)
                use_async = False
        else:
            # Assume Local/vLLM
            evaluator = VLLMModelEvaluator(model_name, config)
            use_async = False
        
        try:
            evaluator.load_model()
        except Exception as e:
            print(f"Error loading model: {e}")
            continue
            
        for dataset_path in datasets:
            if args.all_templates:
                all_template_results = evaluate_on_all_templates(
                    evaluator, str(dataset_path), config, args.subset, args.batch_size, use_async
                )
                
                output_dir = create_output_dir(config['output']['results_dir'])
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_name = f"{model_name}_{dataset_path.stem}_all_templates_{timestamp}"
                
                json_path = output_dir / f"{base_name}_metrics.json"
                
                comprehensive_results = {
                    'model': model_name,
                    'dataset': str(dataset_path),
                    'timestamp': datetime.now().isoformat(),
                    'evaluation_type': 'all_templates',
                    'results': all_template_results['metrics']
                }
                
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(comprehensive_results, f, indent=2, ensure_ascii=False)
                print(f"Saved metrics to {json_path}")
            else:
                if use_async:
                    results_df = asyncio.run(evaluate_dataset_async(evaluator, str(dataset_path), config, args.subset))
                else:
                    results_df = evaluate_dataset(evaluator, str(dataset_path), config, args.subset, args.batch_size)
                
                metrics = calculate_metrics(results_df)
                print_metrics(metrics, f"{model_name} on {dataset_path.stem}")

        evaluator.unload_model()

if __name__ == "__main__":
    main()