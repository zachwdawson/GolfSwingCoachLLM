import logging
from typing import Dict, Any, List, Optional
from openai import OpenAI
from app.core.config import settings
from app.schemas.swing_flaw import SwingFlaw

logger = logging.getLogger(__name__)


def format_metrics_for_prompt(metrics: Dict[str, Dict[str, Any]]) -> str:
    """Format swing metrics into a readable string for the prompt."""
    if not metrics:
        return "No metrics available."
    
    position_names = {
        "address": "Address",
        "top": "Top of Backswing",
        "mid_ds": "Mid-Downswing",
        "impact": "Impact",
        "finish": "Finish"
    }
    
    formatted = []
    for position_key, position_name in position_names.items():
        if position_key in metrics:
            position_metrics = metrics[position_key]
            if position_metrics:
                formatted.append(f"\n{position_name}:")
                for metric_name, value in position_metrics.items():
                    if value is not None:
                        # Format metric name for readability
                        display_name = metric_name.replace("_deg", "").replace("_", " ").title()
                        if metric_name.endswith("_deg") or "angle" in metric_name.lower():
                            formatted.append(f"  - {display_name}: {value:.1f}°")
                        elif metric_name.endswith("_norm"):
                            formatted.append(f"  - {display_name}: {value:.2f}")
                        else:
                            formatted.append(f"  - {display_name}: {value:.2f}")
    
    return "\n".join(formatted) if formatted else "No metrics available."


def format_swing_flaws_for_prompt(swing_flaws: List[SwingFlaw]) -> str:
    """Format swing flaws into a readable string for the prompt."""
    if not swing_flaws:
        return "No specific swing flaws identified."
    
    formatted = []
    for i, flaw in enumerate(swing_flaws, 1):
        formatted.append(f"\n{i}. {flaw.title} (Similarity: {flaw.similarity * 100:.1f}%)")
        if flaw.level:
            formatted.append(f"   Level: {flaw.level}")
        if flaw.contact:
            formatted.append(f"   Contact Type: {flaw.contact}")
        if flaw.ball_shape:
            formatted.append(f"   Ball Shape: {flaw.ball_shape}")
        if flaw.cues:
            formatted.append(f"   Cues:")
            for cue in flaw.cues:
                formatted.append(f"     - {cue}")
        if flaw.drills:
            formatted.append(f"   Recommended Drills:")
            for drill in flaw.drills:
                if isinstance(drill, dict):
                    if "drill explanation" in drill:
                        formatted.append(f"     - {drill['drill explanation']}")
                        if "drill video" in drill and drill["drill video"]:
                            formatted.append(f"       Video: {drill['drill video']}")
    
    return "\n".join(formatted)


def generate_practice_plan(
    metrics: Dict[str, Dict[str, Any]],
    swing_flaws: List[SwingFlaw],
    description: Optional[str] = None,
    ball_shape: Optional[str] = None,
    contact: Optional[str] = None
) -> Optional[str]:
    """
    Generate a structured practice plan using OpenAI API.
    
    Args:
        metrics: Dictionary of swing metrics by position
        swing_flaws: List of identified swing flaws
        description: User-provided description of their swing
        ball_shape: Ball flight shape (hook, slice, normal)
        contact: Contact type (fat, thin, normal, inconsistent)
    
    Returns:
        Structured markdown practice plan or None if generation fails
    """
    # Check if API key is configured
    if not settings.openai_api_key:
        logger.warning("OpenAI API key not configured. Skipping practice plan generation.")
        return None
    
    try:
        client = OpenAI(api_key=settings.openai_api_key, timeout=30.0)
        
        # Format input data for prompt
        metrics_text = format_metrics_for_prompt(metrics)
        flaws_text = format_swing_flaws_for_prompt(swing_flaws)
        
        # Build the prompt
        prompt_parts = [
            "You are an expert golf swing coach. Based on the following swing analysis data, "
            "explain any flaws in the swing and provide a practice plan to improve the swing with the provided drills.",
            "",
            "## Swing Metrics:",
            metrics_text,
            "",
            "## Identified Swing Flaws:",
            flaws_text,
        ]
        
        if description:
            prompt_parts.extend([
                "",
                "## User Description:",
                description
            ])
        
        if ball_shape or contact:
            prompt_parts.append("")
            prompt_parts.append("## Additional Information:")
            if ball_shape:
                prompt_parts.append(f"- Ball Flight: {ball_shape}")
            if contact:
                prompt_parts.append(f"- Contact Type: {contact}")
        
        prompt_parts.extend([
            "",
            "## Instructions:",
            "Generate a structured practice plan in markdown format with the following sections:",
            # "1. **Executive Summary** - A brief overview of the swing analysis",
            "1. **Key Swing Issues Identified** - Summary of the main problems found",
            "2. **Practice Plan** - Specific drills, cues, and exercises organized by priority",
            # "4. **Next Steps** - Recommended progression and follow-up actions",
            "",
            "Make the plan actionable, specific, and encouraging. Focus on the most critical issues first. "
            "Include specific drills from the identified swing flaws when available.",
        ])
        
        prompt = "\n".join(prompt_parts)
        
        logger.info("Calling OpenAI API to generate practice plan...")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert golf swing coach who creates detailed, actionable practice plans based on swing analysis data."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=2000
        )
        
        practice_plan = response.choices[0].message.content
        logger.info("Successfully generated practice plan")
        return practice_plan
        
    except Exception as e:
        logger.error(f"Error generating practice plan with OpenAI: {e}", exc_info=True)
        return None

