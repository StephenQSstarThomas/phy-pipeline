import json
import re
import asyncio
import os
from typing import List, Dict, Any, Tuple, Optional
import hashlib
from datetime import datetime
import logging

# ===================== API SETUP ==============================================
# Choose between official and proxy API
USE_PROXY = True  # Set to False to use official OpenAI API

if USE_PROXY:
    # Proxy API setup
    from openai import AsyncOpenAI
    
    BASE_URL = "https://api.gpt.ge/v1"
    API_KEY = "sk-KeZHX30fqof4BhNy8179370aDd13434e9aA251B8Ef53E9B9"
    
    client = AsyncOpenAI(
        api_key=API_KEY,
        base_url=BASE_URL
    )
    DEFAULT_MODEL = "gpt-4o"
else:
    # Official API setup
    from litellm import completion
    os.environ["OPENAI_API_KEY"] = "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"
    DEFAULT_MODEL = "gpt-4o"

# ===============================================================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MixedQAProcessor:
    """处理题目和解答混合在一起的markdown文件"""
    
    def __init__(self):
        self.raw_sections = []
        self.processed_problems = []
        
    def read_file(self, file_path: str) -> str:
        """读取文件内容"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to read {file_path}: {e}")
            return ""
    
    def extract_problem_sections(self, content: str) -> List[Dict[str, Any]]:
        """提取可能的题目段落"""
        sections = []
        
        # 首先按章节分割（寻找 "# 第X章" 格式）
        chapter_pattern = r'(# 第\d+章[^\n]*\n)'
        chapters = re.split(chapter_pattern, content)
        
        current_chapter = None
        for i, part in enumerate(chapters):
            if re.match(r'# 第\d+章', part):
                current_chapter = part.strip()
                continue
            
            if current_chapter and part.strip():
                # 在每章内寻找题目（格式：数字.数字）
                problem_pattern = r'\n(\d+\.\d+)\s+([^\n]+)\n(.*?)(?=\n\d+\.\d+|\Z)'
                
                matches = re.finditer(problem_pattern, part, re.MULTILINE | re.DOTALL)
                
                for match in matches:
                    problem_num = match.group(1).strip()
                    title = match.group(2).strip()
                    content_text = match.group(3).strip()
                    
                    # 过滤掉明显是目录的条目（内容太短且没有解答特征）
                    if len(content_text) > 100:  # 基本长度过滤
                        sections.append({
                            'chapter': current_chapter,
                            'number': problem_num,
                            'title': title,
                            'content': content_text,
                            'raw_content': f"{problem_num} {title}\n{content_text}"
                        })
        
        logger.info(f"Extracted {len(sections)} potential problem sections")
        return sections
    
    async def call_llm(self, messages: List[Dict[str, str]], model: str = DEFAULT_MODEL) -> str:
        """调用LLM API"""
        try:
            if USE_PROXY:
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=4000
                )
                return response.choices[0].message.content
            else:
                response = completion(
                    model=f"openai/{model}",
                    messages=messages,
                    temperature=0.1,
                    max_tokens=4000
                )
                return response.choices[0].message.content
                
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise
    
    async def analyze_and_split_section(self, section: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """使用LLM分析并分割单个题目段落为题目和解答"""
        
        prompt = f"""
你是一个物理学专家。请分析以下文本段落，判断它是否包含一个完整的物理题目和解答。

文本内容：
{section['raw_content']}

任务：
1. 判断这是否是一个真正的物理题目和解答（而不是目录、标题或其他内容）
2. 如果是，请将内容分割为题目部分和解答部分
3. 修正任何LaTeX格式问题
4. 确保数学公式使用正确的$...$或$$...$$格式

请返回以下JSON格式：
{{
    "is_valid_problem": true/false,
    "confidence": 0.0-1.0,
    "problem_number": "{section['number']}",
    "question": "纯题目内容（如果有的话）",
    "answer": "纯解答内容（如果有的话）",
    "analysis": "简要分析说明",
    "corrections_applied": ["修正列表"]
}}

判断标准：
- 真正的题目通常以"题X.X"开头，包含物理问题描述
- 解答部分通常包含"解"字，有详细的推导过程
- 目录条目通常很短，只有标题没有详细内容
- 数学公式应该用LaTeX格式

专注于物理学内容的准确性和格式的规范性。
"""

        messages = [
            {"role": "system", "content": "你是物理学和数学内容处理专家，擅长识别和分割题目与解答。"},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response_text = await self.call_llm(messages)
            
            # 提取JSON
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_text = json_match.group()
                result = json.loads(json_text)
                
                # 验证结果
                if result.get('is_valid_problem', False) and result.get('confidence', 0) > 0.7:
                    return result
                else:
                    logger.info(f"Section {section['number']} rejected: not a valid problem or low confidence")
                    return None
            else:
                logger.error(f"No valid JSON found in LLM response for section {section['number']}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to analyze section {section['number']}: {e}")
            return None
    
    async def process_all_sections(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """处理所有段落"""
        valid_problems = []
        
        # 为了避免API限制，分批处理
        batch_size = 5
        total_batches = (len(sections) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(sections))
            batch_sections = sections[start_idx:end_idx]
            
            logger.info(f"Processing batch {batch_idx + 1}/{total_batches} ({len(batch_sections)} sections)")
            
            # 并行处理一个批次
            tasks = [self.analyze_and_split_section(section) for section in batch_sections]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 收集有效结果
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Error processing section {batch_sections[i]['number']}: {result}")
                elif result is not None:
                    valid_problems.append(result)
            
            # 批次间延迟
            if batch_idx < total_batches - 1:
                await asyncio.sleep(2)
        
        logger.info(f"Successfully processed {len(valid_problems)} valid problems")
        return valid_problems
    
    def convert_to_json_format(self, problems: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """转换为最终的JSON格式"""
        json_results = []
        
        for problem in problems:
            # 提取图片引用（如果有的话）
            question_text = problem['question']
            answer_text = problem['answer']
            
            # 查找图片引用
            figure_refs = {}
            img_pattern = r'!\[.*?\]\((.*?)\)'
            
            # 从问题中提取图片
            img_matches = re.findall(img_pattern, question_text)
            for i, img_url in enumerate(img_matches):
                fig_id = f"fig_{i+1:03d}"
                figure_refs[fig_id] = img_url
                # 替换为标准引用格式
                question_text = question_text.replace(f"![]({img_url})", f"<{fig_id}>", 1)
            
            # 从答案中提取图片
            img_matches = re.findall(img_pattern, answer_text)
            for i, img_url in enumerate(img_matches):
                fig_id = f"fig_{len(figure_refs)+i+1:03d}"
                figure_refs[fig_id] = img_url
                answer_text = answer_text.replace(f"![]({img_url})", f"<{fig_id}>", 1)
            
            json_item = {
                "id": f"chap1-{problem['problem_number'].replace('.', '-')}",
                "question": question_text,
                "answer": answer_text,
                "class": "MATHEMATICAL",
                "figure": figure_refs,
                "metadata": {
                    "original_problem_number": problem['problem_number'],
                    "confidence": problem.get('confidence', 0.0),
                    "corrections_applied": problem.get('corrections_applied', []),
                    "analysis": problem.get('analysis', ''),
                    "processing_timestamp": datetime.now().isoformat()
                }
            }
            
            json_results.append(json_item)
        
        return json_results
    
    def save_results(self, results: List[Dict[str, Any]], output_file: str):
        """保存结果到JSON文件"""
        try:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Results saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")

async def main():
    """主处理函数"""
    logger.info("Starting mixed QA processing for 大题典.md...")
    
    processor = MixedQAProcessor()
    
    # Step 1: 读取输入文件
    logger.info("Step 1: Reading input file")
    content = processor.read_file("test-example/大题典.md")
    
    if not content:
        logger.error("Failed to read input file")
        return
    
    # Step 2: 提取可能的题目段落
    logger.info("Step 2: Extracting potential problem sections")
    sections = processor.extract_problem_sections(content)
    
    if not sections:
        logger.error("No potential problem sections found")
        return
    
    # 限制处理数量（用于测试）
    test_limit = 10
    sections = sections[:test_limit]
    logger.info(f"Processing first {test_limit} sections for testing")
    
    # Step 3: 使用LLM分析和分割
    logger.info("Step 3: Analyzing and splitting sections with LLM")
    valid_problems = await processor.process_all_sections(sections)
    
    if not valid_problems:
        logger.error("No valid problems found after LLM analysis")
        return
    
    # Step 4: 转换为JSON格式
    logger.info("Step 4: Converting to JSON format")
    json_results = processor.convert_to_json_format(valid_problems)
    
    # Step 5: 保存结果
    logger.info("Step 5: Saving results")
    output_file = "output/mixed_qa_processed.json"
    processor.save_results(json_results, output_file)
    
    logger.info(f"Processing complete! Generated {len(json_results)} QA pairs")
    
    # 打印摘要
    print(f"\n=== PROCESSING SUMMARY ===")
    print(f"Potential sections extracted: {len(sections)}")
    print(f"Valid problems identified: {len(valid_problems)}")
    print(f"Final JSON results: {len(json_results)}")
    print(f"Output saved to: {output_file}")
    
    # 显示前几个结果的预览
    if json_results:
        print(f"\n=== SAMPLE RESULTS ===")
        for i, result in enumerate(json_results[:3]):
            print(f"\nProblem {i+1}: {result['id']}")
            print(f"Question preview: {result['question'][:100]}...")
            print(f"Answer preview: {result['answer'][:100]}...")

if __name__ == "__main__":
    asyncio.run(main()) 