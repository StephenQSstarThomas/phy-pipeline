import json
import re
import asyncio
import aiofiles
import os
from typing import List, Dict, Any, Optional, Tuple
import hashlib
from datetime import datetime
import logging
from dataclasses import dataclass

# ===================== OFFICIAL LITELLM SETUP ==============================================
from litellm import completion

## set ENV variables for official API
os.environ["OPENAI_API_KEY"] = "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"

# ===================== PROXY API SETUP ==================================================
from openai import AsyncOpenAI

BASE_URL = "https://api.gpt.ge/v1"
PROXY_API_KEY = "sk-KeZHX30fqof4BhNy8179370aDd13434e9aA251B8Ef53E9B9"

async_client = AsyncOpenAI(
    api_key=PROXY_API_KEY,
    base_url=BASE_URL
)

model_list = [
    "gpt-4o",
    "gemini-2.0-flash-thinking-exp-1219",
    "deepseek-reasoner",
    "deepseek-chat",
]

# =====================================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QAPair:
    """Data class for question-answer pairs"""
    question: str
    answer: str
    problem_id: str
    confidence: float = 0.0
    source_files: List[str] = None
    figures: Dict[str, str] = None
    metadata: Dict[str, Any] = None

class QAMatcher:
    """Main class for QA matching and content correction"""
    
    def __init__(self, use_proxy: bool = True):
        self.use_proxy = use_proxy
        self.matched_pairs = []
        
    async def extract_problems_from_markdown(self, md_file_path: str) -> List[Dict[str, Any]]:
        """Extract individual problems from markdown file"""
        async with aiofiles.open(md_file_path, 'r', encoding='utf-8') as f:
            content = await f.read()
        
        # Pattern to match problem sections (e.g., "1.1", "# 1.2", etc.)
        problem_pattern = r'(?:^|\n)(?:#\s*)?(\d+\.\d+)\s+([^\n]+)\n(.*?)(?=(?:\n(?:#\s*)?\d+\.\d+|\Z))'
        
        problems = []
        matches = re.finditer(problem_pattern, content, re.MULTILINE | re.DOTALL)
        
        for match in matches:
            problem_num = match.group(1)
            title = match.group(2).strip()
            content_text = match.group(3).strip()
            
            problems.append({
                'id': f"chap1-{problem_num.replace('.', '-')}",
                'number': problem_num,
                'title': title,
                'content': content_text,
                'source': md_file_path
            })
        
        logger.info(f"Extracted {len(problems)} problems from {md_file_path}")
        return problems
    
    async def extract_solutions_from_markdown(self, solutions_file_path: str) -> List[Dict[str, Any]]:
        """Extract solutions from solutions markdown file"""
        async with aiofiles.open(solutions_file_path, 'r', encoding='utf-8') as f:
            content = await f.read()
        
        # Pattern to match solution sections
        solution_pattern = r'(?:^|\n)#\s*(\d+\.\d+)\s*([^\n]*)\n(.*?)(?=(?:\n#\s*\d+\.\d+|\n#\s*Chapter|\Z))'
        
        solutions = []
        matches = re.finditer(solution_pattern, content, re.MULTILINE | re.DOTALL)
        
        for match in matches:
            solution_num = match.group(1)
            title = match.group(2).strip()
            solution_content = match.group(3).strip()
            
            solutions.append({
                'id': f"chap1-{solution_num.replace('.', '-')}",
                'number': solution_num,
                'title': title,
                'content': solution_content,
                'source': solutions_file_path
            })
        
        logger.info(f"Extracted {len(solutions)} solutions from {solutions_file_path}")
        return solutions
    
    async def call_llm_official(self, messages: List[Dict[str, str]], model: str = "gpt-4o") -> str:
        """Call LLM using official LiteLLM API"""
        try:
            response = completion(
                model=f"openai/{model}",
                messages=messages,
                temperature=0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Official API call failed: {e}")
            raise
    
    async def call_llm_proxy(self, messages: List[Dict[str, str]], model: str = "gpt-4o") -> str:
        """Call LLM using proxy API"""
        try:
            response = await async_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.1,
                stream=False,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Proxy API call failed: {e}")
            raise
    
    async def match_qa_with_llm(self, problems: List[Dict], solutions: List[Dict]) -> List[QAPair]:
        """Use LLM to intelligently match problems with solutions"""
        
        prompt = f"""
You are an expert in physics and mathematics. Your task is to match physics problems with their corresponding solutions.

Given the following problems and solutions, create accurate matches and perform content correction.

PROBLEMS ({len(problems)} total):
{json.dumps(problems[:5], indent=2, ensure_ascii=False)}...

SOLUTIONS ({len(solutions)} total):
{json.dumps(solutions[:5], indent=2, ensure_ascii=False)}...

Instructions:
1. Match each problem with its corresponding solution based on problem number and content similarity
2. Correct any LaTeX formatting inconsistencies (e.g., ensure proper math delimiters)
3. Fix any obvious typos or notation errors
4. Ensure mathematical expressions are properly formatted
5. Return the matches in the specified JSON format

Output format:
{{
    "matches": [
        {{
            "problem_id": "chap1-1-1",
            "question": "corrected question text with proper LaTeX",
            "answer": "corrected solution text with proper LaTeX", 
            "confidence": 0.95,
            "corrections_made": ["list of corrections applied"]
        }}
    ]
}}

Focus on accuracy and proper mathematical formatting.
"""

        messages = [
            {"role": "system", "content": "You are an expert physics and mathematics content processor."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            if self.use_proxy:
                response_text = await self.call_llm_proxy(messages)
            else:
                response_text = await self.call_llm_official(messages)
            
            # Parse LLM response
            response_data = json.loads(response_text)
            matches = response_data.get("matches", [])
            
            qa_pairs = []
            for match in matches:
                qa_pair = QAPair(
                    question=match["question"],
                    answer=match["answer"],
                    problem_id=match["problem_id"],
                    confidence=match.get("confidence", 0.0),
                    metadata={
                        "corrections_made": match.get("corrections_made", []),
                        "processed_timestamp": datetime.now().isoformat()
                    }
                )
                qa_pairs.append(qa_pair)
            
            logger.info(f"Successfully matched {len(qa_pairs)} QA pairs using LLM")
            return qa_pairs
            
        except Exception as e:
            logger.error(f"LLM matching failed: {e}")
            return await self.fallback_rule_based_matching(problems, solutions)
    
    async def fallback_rule_based_matching(self, problems: List[Dict], solutions: List[Dict]) -> List[QAPair]:
        """Fallback rule-based matching when LLM fails"""
        logger.info("Using fallback rule-based matching")
        
        qa_pairs = []
        
        # Create lookup dictionary for solutions
        solutions_dict = {sol['number']: sol for sol in solutions}
        
        for problem in problems:
            problem_num = problem['number']
            
            if problem_num in solutions_dict:
                solution = solutions_dict[problem_num]
                
                qa_pair = QAPair(
                    question=problem['content'],
                    answer=solution['content'],
                    problem_id=problem['id'],
                    confidence=0.8,  # Lower confidence for rule-based
                    source_files=[problem['source'], solution['source']],
                    metadata={
                        "matching_method": "rule_based",
                        "problem_title": problem.get('title', ''),
                        "solution_title": solution.get('title', '')
                    }
                )
                qa_pairs.append(qa_pair)
        
        return qa_pairs
    
    async def correct_content_with_llm(self, qa_pair: QAPair) -> QAPair:
        """Use LLM to correct and improve content quality"""
        
        correction_prompt = f"""
You are an expert editor for physics and mathematics content. Please review and correct the following question-answer pair:

QUESTION:
{qa_pair.question}

ANSWER:
{qa_pair.answer}

Tasks:
1. Fix any LaTeX formatting issues (ensure proper $...$ or $$...$$ delimiters)
2. Correct mathematical notation inconsistencies
3. Fix any obvious typos or grammatical errors
4. Ensure mathematical expressions are properly formatted
5. Maintain the original meaning and content
6. Do NOT add new content or change the mathematical substance

Return the corrected version in JSON format:
{{
    "corrected_question": "...",
    "corrected_answer": "...",
    "corrections_applied": ["list of specific corrections made"]
}}
"""

        messages = [
            {"role": "system", "content": "You are an expert mathematics and physics content editor."},
            {"role": "user", "content": correction_prompt}
        ]
        
        try:
            if self.use_proxy:
                response_text = await self.call_llm_proxy(messages)
            else:
                response_text = await self.call_llm_official(messages)
            
            correction_data = json.loads(response_text)
            
            # Update the QA pair with corrections
            qa_pair.question = correction_data["corrected_question"]
            qa_pair.answer = correction_data["corrected_answer"]
            
            if qa_pair.metadata is None:
                qa_pair.metadata = {}
            qa_pair.metadata["corrections_applied"] = correction_data.get("corrections_applied", [])
            qa_pair.metadata["content_corrected"] = True
            
            return qa_pair
            
        except Exception as e:
            logger.error(f"Content correction failed: {e}")
            return qa_pair
    
    def extract_figures(self, content: str) -> Dict[str, str]:
        """Extract figure references from content"""
        figure_pattern = r'<fig_(\d+)>'
        figures = {}
        
        matches = re.findall(figure_pattern, content)
        for match in matches:
            fig_id = f"fig_{match}"
            figures[fig_id] = f"placeholder_url_for_{fig_id}"
        
        return figures
    
    def generate_unique_id(self, content: str) -> str:
        """Generate unique ID based on content hash"""
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()[:8]
        return f"chap1-{content_hash}"
    
    async def convert_to_json_format(self, qa_pairs: List[QAPair]) -> List[Dict[str, Any]]:
        """Convert QA pairs to the required JSON format"""
        
        json_results = []
        
        for qa_pair in qa_pairs:
            # Extract figures from question and answer
            question_figures = self.extract_figures(qa_pair.question)
            answer_figures = self.extract_figures(qa_pair.answer)
            all_figures = {**question_figures, **answer_figures}
            
            json_item = {
                "id": qa_pair.problem_id or self.generate_unique_id(qa_pair.question),
                "question": qa_pair.question,
                "answer": qa_pair.answer,
                "class": "MATHEMATICAL",  # Default classification
                "figure": all_figures,
                "metadata": {
                    "confidence": qa_pair.confidence,
                    "source_files": qa_pair.source_files or [],
                    "processing_timestamp": datetime.now().isoformat(),
                    **(qa_pair.metadata or {})
                }
            }
            
            json_results.append(json_item)
        
        return json_results
    
    async def save_results(self, results: List[Dict[str, Any]], output_file: str):
        """Save results to JSON file"""
        async with aiofiles.open(output_file, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(results, indent=2, ensure_ascii=False))
        
        logger.info(f"Results saved to {output_file}")

# Main processing function
async def process_qa_matching(
    problems_md_path: str,
    solutions_md_path: str,
    output_json_path: str,
    use_proxy: bool = True,
    apply_corrections: bool = True
):
    """Main function to process QA matching and conversion"""
    
    matcher = QAMatcher(use_proxy=use_proxy)
    
    # Step 1: Extract problems and solutions
    logger.info("Step 1: Extracting problems and solutions from markdown files")
    problems = await matcher.extract_problems_from_markdown(problems_md_path)
    solutions = await matcher.extract_solutions_from_markdown(solutions_md_path)
    
    # Step 2: Match QA pairs using LLM
    logger.info("Step 2: Matching problems with solutions using LLM")
    qa_pairs = await matcher.match_qa_with_llm(problems, solutions)
    
    # Step 3: Apply content corrections (optional)
    if apply_corrections:
        logger.info("Step 3: Applying content corrections")
        corrected_pairs = []
        for qa_pair in qa_pairs:
            corrected_pair = await matcher.correct_content_with_llm(qa_pair)
            corrected_pairs.append(corrected_pair)
        qa_pairs = corrected_pairs
    
    # Step 4: Convert to JSON format
    logger.info("Step 4: Converting to JSON format")
    json_results = await matcher.convert_to_json_format(qa_pairs)
    
    # Step 5: Save results
    logger.info("Step 5: Saving results")
    await matcher.save_results(json_results, output_json_path)
    
    logger.info(f"Processing complete! Generated {len(json_results)} QA pairs")
    return json_results

# Example usage
if __name__ == "__main__":
    async def main():
        results = await process_qa_matching(
            problems_md_path="test-example/chap1.md",
            solutions_md_path="test-example/Solutions_June_2014.md", 
            output_json_path="output/processed_chap1.json",
            use_proxy=True,  # Set to False to use official OpenAI API
            apply_corrections=True
        )
        
        print(f"Successfully processed {len(results)} QA pairs")

    # Run the async main function
    asyncio.run(main()) 