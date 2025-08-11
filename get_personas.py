import json
from groq import Groq
from typing import List, Dict, Any
from collections import defaultdict
import logging
from datetime import datetime, time
import pprint
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PersonaRefinementSystem:
    def __init__(self, api_key: str, model: str = "llama-3.1-8b-instant"):
        self.client = Groq(api_key=api_key)
        self.model = model
        self.personas = []
        self.refinement_history = []
        
    def load_leads_data(self, file_path: str) -> List[Dict[str, Any]]:
        """Load leads data from JSON file"""
        try:
            with open(file_path, 'r') as file:
                leads_data = json.load(file)
            logger.info(f"Loaded {len(leads_data)} leads from {file_path}")
            return leads_data
        except Exception as e:
            logger.error(f"Error loading leads data: {e}")
            return []
    
    def get_persona_generation_prompt(self, leads_data: List[Dict[str, Any]], 
                                    existing_personas: List[Dict[str, Any]] = None,
                                    iteration: int = 1) -> str:
        """Generate the detailed prompt for persona creation/refinement"""
        
        base_prompt = f"""
You are an expert customer segmentation and persona creation specialist. Your task is to analyze customer lead data and create exactly 20 distinct, highly detailed customer personas.

## INPUT DATA STRUCTURE
Each lead contains:
- name: Customer identifier
- lead-score: Numerical score indicating lead quality/potential
- PERSONALITY TRAITS: Key behavioral and psychological characteristics
- TOP_PRIORITIES: Primary goals and objectives
- PAIN POINTS: Key challenges and frustrations
- PERSONALISED STRATEGY: Tailored approach recommendations

## LEAD DATA TO ANALYZE:
{json.dumps(leads_data[:50], indent=1)}

{'## EXISTING PERSONAS (FOR REFINEMENT):' if existing_personas else '## INITIAL PERSONA CREATION:'}
{json.dumps(existing_personas, indent=1) if existing_personas else 'This is the initial creation - no existing personas to refine.'}

## TASK REQUIREMENTS:

### 1. PERSONA STRUCTURE
Create exactly 10 personas, each with:
- **persona_id**: Unique identifier (P001-P010)
- **persona_name**: Descriptive name (e.g., "Tech-Savvy Efficiency Seeker")
- **personality_traits**: 5-7 key psychological/behavioral traits
- **top_priorities**: 3-5 primary goals and objectives
- **pain_points**: 3-5 major challenges and frustrations
- **lead_score_range**: Typical lead score range for this persona
- **communication_preferences**: How they prefer to be approached
- **decision_making_style**: How they make purchasing decisions
- **value_drivers**: What motivates them to buy
- **preferred_solutions**: Types of solutions they gravitate toward
- **engagement_strategy**: Tailored approach for this persona
- **representative_leads**: 3-5 lead names that best represent this persona

### 2. CLUSTERING METHODOLOGY
- Group leads with similar personality traits, priorities, and pain points
- Ensure each persona represents a meaningfully different customer segment
- Balance granularity (specific enough to be actionable) with coverage (broad enough to be scalable)
- Consider lead scores as a secondary clustering factor

### 3. QUALITY CRITERIA
- **Distinctiveness**: Each persona should be clearly different from others
- **Actionability**: Personas should enable specific marketing/sales strategies
- **Data-driven**: Based on actual patterns in the lead data
- **Comprehensive**: Cover the full spectrum of your customer base
- **Realistic**: Represent actual customer types, not idealized versions

### 4. REFINEMENT FOCUS {'(Iteration ' + str(iteration) + ')' if iteration > 1 else ''}
{self._get_refinement_instructions(iteration, existing_personas)}

## OUTPUT FORMAT
Return a JSON object with no other text. The structure should be:
{{
  "iteration": {iteration},
  "generation_timestamp": "{datetime.now().isoformat()}",
  "methodology_notes": "Brief explanation of your clustering approach",
  "personas": [
    {{
      "persona_id": "P001",
      "persona_name": "...",
      "personality_traits": ["...", "...", "..."],
      "top_priorities": ["...", "...", "..."],
      "pain_points": ["...", "...", "..."],
      "lead_score_range": "XX-XX",
      "communication_preferences": "...",
      "decision_making_style": "...",
      "preferred_solutions": ["...", "...", "..."],
      "engagement_strategy": "...",
    }}
    // ... 9 more personas
  ]
}}

Analyze the data thoroughly and create personas that will drive effective, targeted marketing and sales strategies.
"""
        return base_prompt
    
    def _get_refinement_instructions(self, iteration: int, existing_personas: List[Dict[str, Any]]) -> str:
        """Get specific refinement instructions based on iteration"""
        if iteration == 1:
            return "Focus on creating comprehensive, distinct personas based on clear data patterns."
        elif iteration == 2:
            return """
- Review persona distinctiveness - merge overly similar personas
- Enhance personas with insufficient detail or unclear targeting
- Ensure balanced coverage across different lead score ranges
- Refine engagement strategies to be more specific and actionable
            """
        elif iteration == 3:
            return """
- Fine-tune personality trait descriptions for better precision
- Optimize pain point articulation for clearer solution mapping
- Enhance demographic profiles with more specific details
- Refine value drivers to be more compelling and specific
            """
        else:
            return """
- Focus on subtle refinements and edge case handling
- Ensure personas are production-ready for marketing/sales teams
- Validate that all personas have clear, actionable strategies
- Optimize for maximum business impact and usability
            """
    
    def generate_personas(self, leads_data: List[Dict[str, Any]], 
                         existing_personas: List[Dict[str, Any]] = None,
                         iteration: int = 1) -> Dict[str, Any]:
        """Generate or refine personas using LLM"""
        
        prompt = self.get_persona_generation_prompt(leads_data, existing_personas, iteration)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert customer segmentation specialist. Always respond with valid JSON format.", },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=100_000,
                response_format={"type": "json_object"}
            )

            # Clean the response to ensure it's valid JSON
            content = response.choices[0].message.content.strip()
            
            # Remove any markdown formatting if present
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            
            result = json.loads(content)
            logger.info(f"Generated personas for iteration {iteration}")
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            logger.error(f"Response content: {response.choices[0].message.content}")
            return None
        except Exception as e:
            logger.error(f"Error generating personas: {e}")
            return None
    
    def evaluate_persona_quality(self, personas: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate the quality of generated personas"""
        
        evaluation_prompt = f"""
Evaluate these 10 customer personas on the following criteria:

PERSONAS TO EVALUATE:
{json.dumps(personas, indent=2)}

EVALUATION CRITERIA:
1. **Distinctiveness** (1-10): How different are the personas from each other?
2. **Completeness** (1-10): How comprehensive is the information for each persona?
3. **Actionability** (1-10): How useful are these for marketing/sales teams?
4. **Data Alignment** (1-10): How well do they reflect the input data patterns?
5. **Business Value** (1-10): How likely are these to drive business results?

Provide:
- Overall score (1-10)
- Scores for each criteria
- Specific improvement recommendations
- Identification of weakest personas that need refinement

Return JSON format:
{{
  "overall_score": X,
  "criteria_scores": {{
    "distinctiveness": X,
    "completeness": X,
    "actionability": X,
    "data_alignment": X,
    "business_value": X
  }},
  "improvement_recommendations": ["...", "...", "..."],
  "weakest_personas": ["P001", "P002", "..."],
  "strengths": ["...", "...", "..."],
  "areas_for_improvement": ["...", "...", "..."]
}}
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator of customer personas. Always respond with valid JSON format."},
                    {"role": "user", "content": evaluation_prompt}
                ],
                temperature=0.3,
                max_tokens=100_000,
                response_format={"type": "json_object"}
            )
            
            # Clean the response to ensure it's valid JSON
            content = response.choices[0].message.content.strip()
            
            # Remove any markdown formatting if present
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            
            return json.loads(content)
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error in evaluation: {e}")
            logger.error(f"Response content: {response.choices[0].message.content}")
            return None
        except Exception as e:
            logger.error(f"Error evaluating personas: {e}")
            return None
    
    def save_results(self, results: Dict[str, Any], file_path: str):
        """Save results to JSON file"""
        try:
            with open(file_path, 'w') as file:
                json.dump(results, file, indent=2)
            logger.info(f"Results saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def run_refinement_loop(self, leads_file: str, max_iterations: int = 5, 
                          target_score: float = 8.5) -> Dict[str, Any]:
        """Run the iterative refinement process"""
        
        # Load initial data
        leads_data = self.load_leads_data(leads_file)
        if not leads_data:
            return None
        
        current_personas = None
        best_personas = None
        best_score = 0
        x = 0
        
        for iteration in range(1, max_iterations + 1):
            logger.info(f"Starting iteration {iteration}")
            
            # Generate personas
            result = self.generate_personas(leads_data[5*x:5*(x+1)], current_personas, iteration)
            x = (x+1) % (len(leads_data) // 5)

            if not result:
                logger.error(f"Failed to generate personas in iteration {iteration}")
                continue

            time.sleep(60)
            
            # Evaluate quality
            evaluation = self.evaluate_persona_quality(result['personas'])
            if not evaluation:
                logger.error(f"Failed to evaluate personas in iteration {iteration}")
                continue
            
            # Combine results
            iteration_result = {
                **result,
                "evaluation": evaluation,
                "iteration": iteration
            }
            
            # Save iteration results
            self.save_results(iteration_result, f"personas_iteration_{iteration}.json")
            
            # Track best result
            current_score = evaluation['overall_score']
            if current_score > best_score:
                best_score = current_score
                best_personas = iteration_result
            
            # Check if target reached
            if current_score >= target_score:
                logger.info(f"Target score {target_score} reached in iteration {iteration}")
                break
            
            # Prepare for next iteration
            current_personas = result['personas']
            self.refinement_history.append(iteration_result)
            
            logger.info(f"Iteration {iteration} completed. Score: {current_score:.2f}")

            time.sleep(60)

        
        # Save final best result
        if best_personas:
            self.save_results(best_personas, "personas_final.json")
            logger.info(f"Refinement completed. Best score: {best_score:.2f}")
        
        return best_personas


# Usage example
def main():
    # Initialize system with Groq
    system = PersonaRefinementSystem(api_key="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")
    
    final_personas = system.run_refinement_loop(
        leads_file="leads.json",
        max_iterations=5,
        target_score=8.5
    )
    
    if final_personas:
        print(f"Final personas generated with score: {final_personas['evaluation']['overall_score']}")
        print(f"Saved to: personas_final.json")
    else:
        print("Failed to generate satisfactory personas")


if __name__ == "__main__":
    main()