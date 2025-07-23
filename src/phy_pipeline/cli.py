"""
Command Line Interface for PHY-Pipeline.

This module provides the main CLI entry point for the PHY-Pipeline package.
"""

import typer
from typing import Optional
from pathlib import Path
import asyncio

from .simple_qa_processor import SimpleQAProcessor
from .mixed_qa_processor import MixedQAProcessor
from .qa_matcher import QAMatcher

app = typer.Typer(
    name="phy-pipeline",
    help="PHY-Pipeline: AI-powered physics problem processing pipeline",
    add_completion=False,
)


@app.command()
def process_simple(
    question_file: Path = typer.Argument(..., help="Path to question markdown file"),
    answer_file: Path = typer.Argument(..., help="Path to answer markdown file"),
    output_file: Path = typer.Option("output.json", help="Output JSON file path"),
    use_proxy: bool = typer.Option(True, help="Use proxy API instead of official OpenAI API"),
):
    """Process separated question and answer files."""
    processor = SimpleQAProcessor()
    
    # Read files
    questions = processor.read_file(str(question_file))
    answers = processor.read_file(str(answer_file))
    
    # Process
    processor.parse_questions(questions)
    processor.parse_solutions(answers)
    
    # Match and process
    asyncio.run(processor.match_and_process())
    
    # Save results
    processor.save_to_json(str(output_file))
    
    typer.echo(f"✅ Processing complete! Results saved to {output_file}")
    typer.echo(f"📊 Processed {len(processor.problems)} problems")


@app.command()
def process_mixed(
    input_file: Path = typer.Argument(..., help="Path to mixed markdown file"),
    output_file: Path = typer.Option("mixed_output.json", help="Output JSON file path"),
    use_proxy: bool = typer.Option(True, help="Use proxy API instead of official OpenAI API"),
):
    """Process mixed question and answer file."""
    processor = MixedQAProcessor()
    
    # Read and process file
    content = processor.read_file(str(input_file))
    processor.parse_mixed_content(content)
    
    # Process sections
    asyncio.run(processor.process_all_sections())
    
    # Save results
    processor.save_to_json(str(output_file))
    
    typer.echo(f"✅ Processing complete! Results saved to {output_file}")
    typer.echo(f"📊 Processed {len(processor.processed_problems)} problems")


@app.command()
def match(
    question_file: Path = typer.Argument(..., help="Path to question markdown file"),
    answer_file: Path = typer.Argument(..., help="Path to answer markdown file"),
    output_file: Path = typer.Option("matched_output.json", help="Output JSON file path"),
    confidence_threshold: float = typer.Option(0.7, help="Confidence threshold for matching"),
):
    """Match questions with answers using AI and rule-based methods."""
    matcher = QAMatcher()
    
    # Load files
    questions = matcher.load_questions(str(question_file))
    answers = matcher.load_answers(str(answer_file))
    
    # Perform matching
    matched_pairs = asyncio.run(matcher.match_qa_pairs(questions, answers))
    
    # Filter by confidence
    high_confidence = [pair for pair in matched_pairs if pair.confidence >= confidence_threshold]
    
    # Save results
    matcher.save_results(high_confidence, str(output_file))
    
    typer.echo(f"✅ Matching complete! Results saved to {output_file}")
    typer.echo(f"📊 Matched {len(high_confidence)} high-confidence pairs out of {len(matched_pairs)} total")


@app.command()
def version():
    """Show version information."""
    from . import __version__
    typer.echo(f"PHY-Pipeline version: {__version__}")


def main():
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main() 