import json
import os
import yaml

with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
    MODEL_LIST = list(config.get('local_models', {}).keys()) + list(config.get('api_models', {}).keys())

LANG_PAIR_LIST = [
    'zh-zh-v2',
    'zh-ja-v2',
    'zh-ko-v2',
    'zh-en-v2',
    'ja-zh',
    'ja-ja',
    'ja-ko',
    'ja-en',
    'ko-zh-v2',
    'ko-ja-v2',
    'ko-ko-v2',
    'ko-en-v2',
]

def tableize(out_name = 'result_table.csv'):
    buffer = ''
    dir_list = os.listdir('./results')
    with open(out_name, 'w', encoding='utf-8') as fp:
        for model in MODEL_LIST:
            # Hack for renaming hyperclova model
            if model == 'hyperclovax':
                buffer += f"\"HyperCLOVAX-SEED-Text-Instruct-1.5B (bias score, closer to 0 is better)\",,,,,\"HyperCLOVAX-SEED-Text-Instruct-1.5B (cultural accuracy, higher is better)\"\n"
            else:
                buffer += f"\"{model} (bias score, closer to 0 is better)\",,,,,\"{model} (cultural accuracy, higher is better)\"\n"
            buffer += "Culture \\ Lang,Chinese,Japanese,Korean,,Culture \\ Lang,Chinese,Japanese,Korean\n"

            filtered_list = [file_name for file_name in dir_list if model in file_name]
            filename_dict = {}
            for lang_pair in LANG_PAIR_LIST:
                further_filtered_list = sorted([file_name for file_name in filtered_list if lang_pair in file_name], reverse=True)
                if len(further_filtered_list) > 0:
                    with open(os.path.join('results', further_filtered_list[0]), 'r') as json_fp:
                        json_file = json.load(json_fp)
                    # Check if this is the new all_templates format
                    if 'evaluation_type' in json_file and json_file['evaluation_type'] == 'all_templates':
                        # Use averaged results from all templates
                        filename_dict[lang_pair] = json_file['results']['averaged']['bias_score'], json_file['results']['averaged']['culture_accuracy']
                    else:
                        # Old format
                        filename_dict[lang_pair] = json_file['metrics']['bias_score'], json_file['metrics']['culture_accuracy']
                else:
                    filename_dict[lang_pair] = None, None
            
            buffer += f"Chinese,{filename_dict['zh-zh-v2'][0]},{filename_dict['zh-ja-v2'][0]},{filename_dict['zh-ko-v2'][0]},,Chinese,{filename_dict['zh-zh-v2'][1]},{filename_dict['zh-ja-v2'][1]},{filename_dict['zh-ko-v2'][1]}\n"
            buffer += f"Japanese,{filename_dict['ja-zh'][0]},{filename_dict['ja-ja'][0]},{filename_dict['ja-ko'][0]},,Japanese,{filename_dict['ja-zh'][1]},{filename_dict['ja-ja'][1]},{filename_dict['ja-ko'][1]}\n"
            buffer += f"Korean,{filename_dict['ko-zh-v2'][0]},{filename_dict['ko-ja-v2'][0]},{filename_dict['ko-ko-v2'][0]},,Korean,{filename_dict['ko-zh-v2'][1]},{filename_dict['ko-ja-v2'][1]},{filename_dict['ko-ko-v2'][1]}\n\n"
            fp.write(buffer)
            buffer = ''

def single_table(out_name = 'single_result_table.csv'):
    buffer = ''
    dir_list = os.listdir('./results')
    with open(out_name, 'w', encoding='utf-8') as fp:
        buffer += "Model,Culture,Language,Bias Score,Cultural Accuracy,CoT\n"
        for model in MODEL_LIST:
            filtered_list = [file_name for file_name in dir_list if model in file_name]
            
            # Separate CoT and non-CoT files
            cot_files = {}
            non_cot_files = {}
            
            for lang_pair in LANG_PAIR_LIST:
                further_filtered_list = sorted([file_name for file_name in filtered_list if lang_pair in file_name], reverse=True)
                for file_name in further_filtered_list:
                    is_cot = 'cot' in file_name.lower()
                    if is_cot:
                        if lang_pair not in cot_files:
                            cot_files[lang_pair] = file_name
                    else:
                        if lang_pair not in non_cot_files:
                            non_cot_files[lang_pair] = file_name
            
            # Process both CoT and non-CoT versions
            for is_cot_version in [False, True]:
                files_dict = cot_files if is_cot_version else non_cot_files
                if not files_dict:
                    continue
                    
                filename_dict = {}
                for lang_pair in LANG_PAIR_LIST:
                    if lang_pair in files_dict:
                        with open(os.path.join('results', files_dict[lang_pair]), 'r') as json_fp:
                            json_file = json.load(json_fp)
                        # Check if this is the new all_templates format
                        if 'evaluation_type' in json_file and 'all_templates' in json_file['evaluation_type']:
                            # Use averaged results from all templates
                            filename_dict[lang_pair] = json_file['results']['averaged']['bias_score'], json_file['results']['averaged']['culture_accuracy']
                        else:
                            # Old format
                            filename_dict[lang_pair] = json_file['metrics']['bias_score'], json_file['metrics']['culture_accuracy']
                    else:
                        filename_dict[lang_pair] = None, None
                
                cot_label = 'Yes' if is_cot_version else 'No'
                buffer += f"{model},Chinese,Chinese,{filename_dict['zh-zh-v2'][0]},{filename_dict['zh-zh-v2'][1]},{cot_label}\n"
                buffer += f"{model},Chinese,Japanese,{filename_dict['zh-ja-v2'][0]},{filename_dict['zh-ja-v2'][1]},{cot_label}\n"
                buffer += f"{model},Chinese,Korean,{filename_dict['zh-ko-v2'][0]},{filename_dict['zh-ko-v2'][1]},{cot_label}\n"
                buffer += f"{model},Chinese,English,{filename_dict['zh-en-v2'][0]},{filename_dict['zh-en-v2'][1]},{cot_label}\n"
                buffer += f"{model},Japanese,Chinese,{filename_dict['ja-zh'][0]},{filename_dict['ja-zh'][1]},{cot_label}\n"
                buffer += f"{model},Japanese,Japanese,{filename_dict['ja-ja'][0]},{filename_dict['ja-ja'][1]},{cot_label}\n"
                buffer += f"{model},Japanese,Korean,{filename_dict['ja-ko'][0]},{filename_dict['ja-ko'][1]},{cot_label}\n"
                buffer += f"{model},Japanese,English,{filename_dict['ja-en'][0]},{filename_dict['ja-en'][1]},{cot_label}\n"
                buffer += f"{model},Korean,Chinese,{filename_dict['ko-zh-v2'][0]},{filename_dict['ko-zh-v2'][1]},{cot_label}\n"
                buffer += f"{model},Korean,Japanese,{filename_dict['ko-ja-v2'][0]},{filename_dict['ko-ja-v2'][1]},{cot_label}\n"
                buffer += f"{model},Korean,Korean,{filename_dict['ko-ko-v2'][0]},{filename_dict['ko-ko-v2'][1]},{cot_label}\n"
                buffer += f"{model},Korean,English,{filename_dict['ko-en-v2'][0]},{filename_dict['ko-en-v2'][1]},{cot_label}\n"
        fp.write(buffer)

if __name__ == '__main__':
    #tableize()
    single_table()
