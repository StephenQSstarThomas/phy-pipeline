import json
import re
import asyncio
import os
from typing import List, Dict, Any
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

class SimpleQAProcessor:
    """Simplified QA processor for chap1 physics problems"""
    
    def __init__(self):
        self.problems = []
        self.solutions = []
        
    def read_file(self, file_path: str) -> str:
        """Read file content"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to read {file_path}: {e}")
            return ""
    
    def extract_problems_from_chap1(self, content: str) -> List[Dict[str, Any]]:
        """Extract problems specifically from chap1.md format"""
        problems = []
        
        # Split by problem numbers (1.1, 1.2, etc.)
        problem_sections = re.split(r'\n\s*(\d+\.\d+)\s+', content)
        
        for i in range(1, len(problem_sections), 2):
            if i + 1 < len(problem_sections):
                problem_num = problem_sections[i].strip()
                problem_content = problem_sections[i + 1].strip()
                
                if problem_content:
                    problems.append({
                        'id': f"chap1-{problem_num.replace('.', '-')}",
                        'number': problem_num,
                        'content': problem_content,
                        'source': 'chap1.md'
                    })
        
        logger.info(f"Extracted {len(problems)} problems from chap1.md")
        return problems
    
    def extract_solutions_from_manual(self, content: str) -> List[Dict[str, Any]]:
        """Extract solutions from Solutions_June_2014.md format"""
        solutions = []
        
        # Look for solution patterns like "# 1.1", "1.2", etc.
        solution_pattern = r'#\s*(\d+\.\d+)\s*([^\n]*?)\n(.*?)(?=\n#\s*\d+\.\d+|\n#\s*Chapter|\Z)'
        
        matches = re.finditer(solution_pattern, content, re.MULTILINE | re.DOTALL)
        
        for match in matches:
            solution_num = match.group(1).strip()
            title = match.group(2).strip()
            solution_content = match.group(3).strip()
            
            if solution_content:
                solutions.append({
                    'id': f"chap1-{solution_num.replace('.', '-')}",
                    'number': solution_num,
                    'title': title,
                    'content': solution_content,
                    'source': 'Solutions_June_2014.md'
                })
        
        logger.info(f"Extracted {len(solutions)} solutions from Solutions_June_2014.md")
        return solutions
    
    async def call_llm(self, messages: List[Dict[str, str]], model: str = DEFAULT_MODEL) -> str:
        """Call LLM with proper API handling"""
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
    
    async def match_and_correct_single_pair(self, problem: Dict, solution: Dict) -> Dict[str, Any]:
        """Process a single problem-solution pair"""
        
        prompt = f"""
You are an expert physics and mathematics editor. You have a physics problem and its corresponding solution that need to be matched and corrected.

PROBLEM {problem['number']}:
{problem['content']}

SOLUTION {solution['number']}:
{solution['content']}

Tasks:
1. Verify that the problem and solution match (they should have the same number)
2. Correct any LaTeX formatting issues (ensure proper $ and $$ delimiters)
3. Fix mathematical notation inconsistencies
4. Fix any obvious typos or formatting errors
5. Ensure the content is properly formatted for JSON output
6. Do NOT change the mathematical substance or add new content

Return the result in this exact JSON format:
{{
    "id": "chap1-{problem['number'].replace('.', '-')}",
    "question": "corrected problem text with proper LaTeX formatting",
    "answer": "corrected solution text with proper LaTeX formatting",
    "class": "MATHEMATICAL",
    "figure": {{}},
    "metadata": {{
        "original_problem_number": "{problem['number']}",
        "original_solution_number": "{solution['number']}",
        "corrections_applied": ["list of corrections made"],
        "processing_timestamp": "{datetime.now().isoformat()}"
    }}
}}

Focus on:
- Proper LaTeX math delimiters ($...$ for inline, $$...$$ for display)
- Consistent mathematical notation
- Clean formatting without extra spaces or line breaks
- Preserving the original mathematical content exactly
"""

        messages = [
            {"role": "system", "content": "You are an expert physics and mathematics content editor. Return only valid JSON."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response_text = await self.call_llm(messages)
            
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_text = json_match.group()
                result = json.loads(json_text)
                return result
            else:
                raise ValueError("No valid JSON found in LLM response")
                
        except Exception as e:
            logger.error(f"Failed to process pair {problem['number']}: {e}")
            # Return fallback format
            return {
                "id": f"chap1-{problem['number'].replace('.', '-')}",
                "question": problem['content'],
                "answer": solution['content'],
                "class": "MATHEMATICAL", 
                "figure": {},
                "metadata": {
                    "original_problem_number": problem['number'],
                    "original_solution_number": solution['number'],
                    "corrections_applied": ["Fallback: No corrections applied due to processing error"],
                    "processing_timestamp": datetime.now().isoformat(),
                    "error": str(e)
                }
            }
    
    def match_problems_with_solutions(self, problems: List[Dict], solutions: List[Dict]) -> List[tuple]:
        """Match problems with solutions by number"""
        matches = []
        
        # Create solution lookup by number
        solutions_dict = {sol['number']: sol for sol in solutions}
        
        for problem in problems:
            problem_num = problem['number']
            if problem_num in solutions_dict:
                matches.append((problem, solutions_dict[problem_num]))
                logger.info(f"Matched problem {problem_num} with solution")
            else:
                logger.warning(f"No solution found for problem {problem_num}")
        
        logger.info(f"Successfully matched {len(matches)} problem-solution pairs")
        return matches
    
    async def process_all_matches(self, matches: List[tuple]) -> List[Dict[str, Any]]:
        """Process all matched pairs"""
        results = []
        
        for i, (problem, solution) in enumerate(matches):
            logger.info(f"Processing pair {i+1}/{len(matches)}: {problem['number']}")
            
            try:
                result = await self.match_and_correct_single_pair(problem, solution)
                results.append(result)
                
                # Add small delay to avoid rate limiting
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Failed to process pair {problem['number']}: {e}")
                continue
        
        return results
    
    def save_results(self, results: List[Dict[str, Any]], output_file: str):
        """Save results to JSON file"""
        try:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Results saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")

async def main():
    """Main processing function"""
    logger.info("Starting QA processing...")
    
    processor = SimpleQAProcessor()
    
    # Step 1: Read input files
    logger.info("Step 1: Reading input files")
    problems_content = processor.read_file("test-example/chap1.md")
    solutions_content = processor.read_file("test-example/Solutions_June_2014.md")
    
    if not problems_content or not solutions_content:
        logger.error("Failed to read input files")
        return
    
    # Step 2: Extract problems and solutions
    logger.info("Step 2: Extracting problems and solutions")
    problems = processor.extract_problems_from_chap1(problems_content)
    solutions = processor.extract_solutions_from_manual(solutions_content)
    
    # Step 3: Match problems with solutions
    logger.info("Step 3: Matching problems with solutions")
    matches = processor.match_problems_with_solutions(problems, solutions)
    
    if not matches:
        logger.error("No matches found between problems and solutions")
        return
    
    # Step 4: Process matches with LLM
    logger.info("Step 4: Processing matches with LLM correction")
    results = await processor.process_all_matches(matches)
    
    # Step 5: Save results
    logger.info("Step 5: Saving results")
    output_file = "output/processed_chap1_corrected.json"
    processor.save_results(results, output_file)
    
    logger.info(f"Processing complete! Generated {len(results)} corrected QA pairs")
    
    # Print summary
    print(f"\n=== PROCESSING SUMMARY ===")
    print(f"Problems extracted: {len(problems)}")
    print(f"Solutions extracted: {len(solutions)}")
    print(f"Successful matches: {len(matches)}")
    print(f"Final results: {len(results)}")
    print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    asyncio.run(main()) 