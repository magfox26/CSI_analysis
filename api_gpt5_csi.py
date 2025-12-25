import datetime
import json
import openai
import time
import base64
import tqdm
from pathlib import Path
from io import BytesIO
import os
import argparse
import sys
import requests

with open('/mnt/workspace/xintong/api_key.txt', 'r') as f:
    lines = f.readlines()

API_KEY = lines[0].strip()
BASE_URL = lines[1].strip()

openai.api_key = API_KEY
openai.base_url = BASE_URL


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def call_api(prompt, image_path):
    base64_image = encode_image(image_path)
    response = openai.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}, 
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    return response.choices[0].message.content


CSI_PROMPT = """You are a professional Culture-Specific Items (CSIs) analyzer for a multilingual image-based translation dataset.

Your analysis directly supports a downstream translation task, where certain expressions must be translated using substitution or cultural adaptation, rather than literal or transliterated translation.

========================
IMPORTANT SCOPE
========================

- The source language is Chinese.
- Focus ONLY on text written in Chinese as it appears in the image.
- Ignore all other languages, symbols, or visual elements unless they directly affect the interpretation of Chinese text.
- Exhaustively identify ALL Culture-Specific Items (CSIs) present in the image text.
- Do NOT stop after finding a single CSI.

========================
Core Translation-Oriented Principle
========================

Base every decision on the following question:

Would a non-Chinese audience, without Chinese cultural, historical, or institutional background knowledge, likely misunderstand, misinterpret, or fail to appropriately translate this expression if it were rendered literally?

ONLY expressions that clearly trigger this problem should be considered CSIs.

========================
CSI Definition
========================

A Culture-Specific Item (CSI) is a minimal textual unit (word, phrase, abbreviation, or fixed expression) whose FORM OF EXPRESSION—not merely its dictionary meaning—is shaped by Chinese-specific cultural, historical, institutional, or community conventions, such that:

- A literal or surface-level translation would be misleading, confusing, pragmatically incorrect, or culturally opaque; AND
- Accurate translation would typically require substitution, functional replacement, or explanatory adaptation, rather than direct lexical correspondence.

========================
Necessary Conditions (ALL must be met)
========================

An expression should be labeled as a CSI ONLY IF all of the following are true:

1. Chinese-Specific Dependence  
   The expression relies on knowledge, conventions, or interpretive frameworks that are specific to Chinese-speaking communities.

2. Translation Failure Risk  
   A non-Chinese reader cannot reliably infer the intended referent, function, or communicative effect from a literal translation alone.

3. Substitution Requirement  
   Proper translation would normally involve replacing the expression with a culturally or functionally equivalent concept, or adding interpretive clarification.

4. Not Mere Lexical Difference  
   The difficulty arises from cultural or institutional specificity, not simply from different conventional wordings for a universally shared concept.

========================
Explicit Non-CSI Guidance
========================

Do NOT label an expression as a CSI if:

- It denotes a universally shared concept or function whose meaning and usage are directly transferable across cultures;
- It can be translated naturally and correctly through standard lexical translation or widely established equivalents;
- Its interpretation does NOT depend on Chinese-specific cultural, historical, or institutional knowledge.

Cultural association alone is NOT sufficient.
An item qualifies as a CSI ONLY IF its correct interpretation or translation would likely fail for a non-Chinese audience without cultural or institutional context.

========================
CSI Types (Choose EXACTLY ONE)
========================

Each identified CSI must be assigned exactly ONE of the following types:

- csi_cultural_reference  
  Culture-, history-, or community-specific references rooted in Chinese social life, traditions, daily practices, belief systems, or shared background knowledge.

- csi_idiomatic_and_slang_expression  
  Idiomatic, figurative, colloquial, internet-mediated, or discourse-specific expressions characteristic of Chinese usage and not directly interpretable cross-culturally.

- csi_social_pragmatics  
  Expressions shaped by Chinese-specific social norms, interactional conventions, address systems, politeness strategies, role-based language, or institutional discourse.

- csi_localization_norm  
  Expressions governed by Chinese institutional, regulatory, or conventional norms, including measurement units, date/time formats, address writing styles, administrative labels, public signage logic, or system-internal naming practices.

========================
Extraction Constraints
========================

- Extract minimal meaningful units only; do NOT output full sentences.
- Preserve the exact surface form as shown in the image.
- Treat each CSI independently; do NOT merge multiple items.
- Be conservative: if genuinely uncertain, do NOT extract the item as a CSI.

========================
Output Format (STRICT JSON ONLY)
========================

Return ONLY ONE of the following formats.

Case 1: NO CSI found

{
  "status": "Not contain",
  "reason": "<用中文说明为什么图中文字在翻译层面不涉及文化特定表达>"
}

Case 2: CSI found

{
  "status": "Contain",
  "items": [
    {
      "text": "<exact Chinese text>",
      "type": "<one CSI type>",
      "reason": "<用中文说明该表达是否只在中文文化或制度语境中成立，以及为什么直译或字面理解会对非中文使用者造成误解或信息缺失>"
    }
  ]
}

Rules:

- If status is "Not contain", the reason field is REQUIRED.
- If status is "Contain", items must be a non-empty list.
- Do NOT output any text outside the JSON object.

Now analyze the image and strictly follow the rules above."""


def process_dataset(image_folder, dataset_name, retries=3, retry_wait=2):
    results = {}
    error_log = {}
    
    image_folder_path = Path(image_folder)
    image_files = list(image_folder_path.glob("*.jpg")) + \
                  list(image_folder_path.glob("*.png")) + \
                  list(image_folder_path.glob("*.jpeg"))
    
    print(f"Processing dataset: {dataset_name}")
    print(f"Found {len(image_files)} images in {image_folder}")
    
    for image_path in tqdm.tqdm(image_files):
        image_name = image_path.name 
        last_error = None
        
        for attempt in range(1, retries + 1):
            try:
                output = call_api(CSI_PROMPT, str(image_path))
                
                try:
                    clean_output = output.strip()
                    if clean_output.startswith("```json"):
                        clean_output = clean_output[7:]
                    if clean_output.startswith("```"):
                        clean_output = clean_output[3:]
                    if clean_output.endswith("```"):
                        clean_output = clean_output[:-3]
                    clean_output = clean_output.strip()
                    
                    parsed_output = json.loads(clean_output)
                    results[image_name] = parsed_output
                except json.JSONDecodeError as je:
                    print(f"[{image_name}] JSON解析失败: {je}")
                    results[image_name] = {
                        "status": "Error",
                        "raw_output": output,
                        "error": f"JSON decode error: {str(je)}"
                    }
                
                break 
                
            except Exception as e:
                last_error = str(e)
                print(f"[{image_name}] 第 {attempt} 次失败：{e}")
                if attempt < retries:
                    time.sleep(retry_wait)
                else:
                    print(f"[{image_name}] 已重试 {retries} 次仍失败")
                    results[image_name] = {
                        "status": "Error",
                        "error": last_error
                    }
                    error_log[image_name] = last_error
    
    return results, error_log


if __name__ == '__main__':
    model_name = "gpt-5-2025-08-07-GlobalStandard"
    print(f"Using model: {model_name}")
    
    root = f"/mnt/workspace/xintong/pjh/All_result/csi_analysis_results/"
    today = datetime.date.today()
    
    Path(root).mkdir(parents=True, exist_ok=True)
    print(f"结果保存地址: {root}")
    
    datasets = [
        {
            "name": "AibTrans",
            "path": "/mnt/workspace/xintong/dataset/practice_ds_500/"
        },
        {
            "name": "OCRMT30K",
            "path": "/mnt/workspace/xintong/dataset/OCRMT30K-refine/whole_image_v2/"
        }
    ]
    
    for dataset in datasets:
        dataset_name = dataset["name"]
        image_folder = dataset["path"]
        
        print(f"\n{'='*60}")
        print(f"开始处理数据集: {dataset_name}")
        print(f"{'='*60}\n")
        
        results, error_log = process_dataset(image_folder, dataset_name)
        
        output_path = os.path.join(root, f"{model_name}_csi_analysis_{dataset_name}.json")
        print(f"\n保存结果到: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        
        if len(error_log) > 0:
            error_log_path = os.path.join(root, f"{model_name}_csi_analysis_{dataset_name}_error_log.json")
            print(f"保存错误日志到: {error_log_path}")
            with open(error_log_path, 'w', encoding='utf-8') as f:
                json.dump(error_log, f, ensure_ascii=False, indent=4)
        
        print(f"\n数据集 {dataset_name} 处理完成!")
        print(f"成功处理: {len(results)} 张图片")
        print(f"失败: {len(error_log)} 张图片")
    
    print(f"\n{'='*60}")
    print("所有数据集处理完成!")
    print(f"{'='*60}")