#!/usr/bin/env python3
"""
Main Attack Script for SafeKV Time Channel Attack

This script orchestrates the complete time channel attack against SafeKV systems.
It generates candidate sets, establishes baselines, and executes attacks to
demonstrate the vulnerability of shared KV-Cache systems.

Author: Security Research Team
Date: 2024
"""

import asyncio
import json
import argparse
import logging
import time
from typing import Dict, List, Optional
from pathlib import Path

from attack_model import TimeChannelAttacker
from candidate_generator import CandidateGenerator, CandidateSet

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SafeKVTimeChannelAttack:
    """
    Main orchestrator for SafeKV time channel attacks.
    """
    
    def __init__(self, 
                 api_url: str,
                 output_dir: str = "attack_results",
                 baseline_requests: int = 15,
                 request_delay: float = 0.2):
        """
        Initialize the attack orchestrator.
        
        Args:
            api_url: URL of the target LLM API
            output_dir: Directory to save attack results
            baseline_requests: Number of requests for baseline establishment
            request_delay: Delay between requests
        """
        self.api_url = api_url
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.attacker = TimeChannelAttacker(
            api_url=api_url,
            baseline_requests=baseline_requests,
            request_delay=request_delay
        )
        self.generator = CandidateGenerator()
        
        # Attack state
        self.baseline_established = False
        self.attack_results = {}
        
    async def setup_attack(self):
        """Set up the attack by establishing baseline and generating candidates."""
        logger.info("Setting up SafeKV time channel attack...")
        
        # Generate candidate sets
        logger.info("Generating candidate sets...")
        candidate_sets = self.generator.generate_all_candidate_sets()
        
        # Save candidate sets
        candidate_file = self.output_dir / "candidate_sets.json"
        self.generator.save_candidate_sets(str(candidate_file))
        
        # Generate random prompts for baseline and interleaving
        logger.info("Generating random prompts for baseline...")
        baseline_prompts = self.attacker.generate_random_prompts(30)
        
        # Establish baseline
        logger.info("Establishing baseline TTFT...")
        try:
            await self.attacker.establish_baseline(baseline_prompts)
            self.baseline_established = True
            logger.info("Baseline established successfully")
        except Exception as e:
            logger.error(f"Failed to establish baseline: {e}")
            raise
        
        return candidate_sets, baseline_prompts
    
    async def execute_attack_session(self, 
                                   session_id: str,
                                   candidate_set: CandidateSet,
                                   random_prompts: List[str]) -> Dict:
        """
        Execute a single attack session.
        
        Args:
            session_id: Unique session identifier
            candidate_set: Set of candidates to test
            random_prompts: Random prompts for interleaving
            
        Returns:
            Attack results
        """
        logger.info(f"Executing attack session: {session_id}")
        logger.info(f"Target: {candidate_set.name} ({len(candidate_set.candidates)} candidates)")
        
        try:
            # Execute attack
            session = await self.attacker.attack_candidate_set(
                session_id=session_id,
                target_pattern=candidate_set.name,
                candidates=candidate_set.candidates,
                random_prompts=random_prompts
            )
            
            # Analyze results
            analysis = self.attacker.analyze_attack_results(session_id)
            
            # Add candidate set metadata
            analysis['candidate_set'] = {
                'name': candidate_set.name,
                'description': candidate_set.description,
                'category': candidate_set.category,
                'expected_pattern': candidate_set.expected_pattern,
                'confidence_threshold': candidate_set.confidence_threshold
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Attack session {session_id} failed: {e}")
            return {
                'session_id': session_id,
                'error': str(e),
                'success': False
            }
    
    async def run_comprehensive_attack(self, 
                                     target_categories: Optional[List[str]] = None,
                                     max_candidates_per_set: int = 15):
        """
        Run comprehensive attack against all or selected candidate categories.
        
        Args:
            target_categories: List of categories to attack (None for all)
            max_candidates_per_set: Maximum candidates per set to limit attack time
        """
        logger.info("Starting comprehensive SafeKV time channel attack...")
        
        # Setup attack
        candidate_sets, baseline_prompts = await self.setup_attack()
        
        # Filter candidate sets if specified
        if target_categories:
            candidate_sets = {k: v for k, v in candidate_sets.items() 
                            if k in target_categories}
            logger.info(f"Targeting specific categories: {target_categories}")
        
        # Limit candidates per set
        for name, candidate_set in candidate_sets.items():
            if len(candidate_set.candidates) > max_candidates_per_set:
                candidate_set.candidates = candidate_set.candidates[:max_candidates_per_set]
                logger.info(f"Limited {name} to {max_candidates_per_set} candidates")
        
        # Generate additional random prompts for interleaving
        interleaving_prompts = self.attacker.generate_random_prompts(20)
        
        # Execute attacks for each candidate set
        attack_results = {}
        for category, candidate_set in candidate_sets.items():
            session_id = f"attack_{category}_{int(time.time())}"
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Attacking category: {category}")
            logger.info(f"{'='*60}")
            
            result = await self.execute_attack_session(
                session_id=session_id,
                candidate_set=candidate_set,
                random_prompts=interleaving_prompts
            )
            
            attack_results[category] = result
            
            # Add delay between attack sessions
            await asyncio.sleep(1.0)
        
        # Save results
        self.save_attack_results(attack_results)
        
        # Generate summary report
        self.generate_summary_report(attack_results)
        
        return attack_results
    
    def save_attack_results(self, results: Dict):
        """Save attack results to files."""
        # Save detailed results
        results_file = self.output_dir / "attack_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save summary
        summary = self.attacker.get_attack_summary()
        summary_file = self.output_dir / "attack_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Attack results saved to {self.output_dir}")
    
    def generate_summary_report(self, results: Dict):
        """Generate a human-readable summary report."""
        report_file = self.output_dir / "attack_report.txt"
        
        with open(report_file, 'w') as f:
            f.write("SafeKV Time Channel Attack Report\n")
            f.write("=" * 50 + "\n\n")
            
            # Overall summary
            summary = self.attacker.get_attack_summary()
            f.write("Overall Attack Summary:\n")
            f.write(f"  Total Sessions: {summary['total_sessions']}\n")
            f.write(f"  Successful Sessions: {summary['successful_sessions']}\n")
            f.write(f"  Success Rate: {summary['success_rate']:.2%}\n")
            f.write(f"  Total Requests: {summary['total_requests']}\n")
            f.write(f"  Total Hit Candidates: {summary['total_hit_candidates']}\n")
            f.write(f"  Baseline TTFT: {summary['baseline_ttft']:.3f}s\n")
            f.write(f"  Hit Threshold: {summary['hit_threshold']:.3f}s\n\n")
            
            # Per-category results
            f.write("Per-Category Results:\n")
            f.write("-" * 30 + "\n")
            
            for category, result in results.items():
                if 'error' in result:
                    f.write(f"{category}: ERROR - {result['error']}\n")
                    continue
                
                stats = result.get('statistics', {})
                hit_candidates = result.get('hit_candidates', [])
                likely_private = result.get('likely_private_info', [])
                
                f.write(f"{category}:\n")
                f.write(f"  Hit Rate: {stats.get('hit_rate', 0):.2%}\n")
                f.write(f"  Hit Candidates: {len(hit_candidates)}\n")
                f.write(f"  Likely Private Info: {len(likely_private)}\n")
                
                if hit_candidates:
                    f.write(f"  Sample Hits: {hit_candidates[:3]}\n")
                if likely_private:
                    f.write(f"  Private Info: {likely_private[:3]}\n")
                f.write("\n")
            
            # Recommendations
            f.write("Security Recommendations:\n")
            f.write("-" * 30 + "\n")
            f.write("1. Implement SafeKV privacy detection mechanisms\n")
            f.write("2. Use KV-Cache partitioning for sensitive data\n")
            f.write("3. Add timing randomization to prevent side channel attacks\n")
            f.write("4. Monitor cache access patterns for suspicious activity\n")
            f.write("5. Implement rate limiting and request throttling\n")
        
        logger.info(f"Attack report saved to {report_file}")
    
    async def run_demo_attack(self):
        """Run a demonstration attack with a small set of candidates."""
        logger.info("Running demonstration attack...")
        
        # Create a small demo candidate set
        demo_candidates = [
            "user@example.com",
            "admin@company.org", 
            "password123",
            "admin123",
            "123-45-6789",
            "987-65-4321"
        ]
        
        demo_set = CandidateSet(
            name="demo_set",
            description="Demonstration candidate set",
            candidates=demo_candidates,
            category="demo",
            expected_pattern="demo"
        )
        
        # Setup
        _, baseline_prompts = await self.setup_attack()
        
        # Execute demo attack
        session_id = f"demo_attack_{int(time.time())}"
        result = await self.execute_attack_session(
            session_id=session_id,
            candidate_set=demo_set,
            random_prompts=baseline_prompts[:5]
        )
        
        # Print results
        print("\nDemo Attack Results:")
        print(json.dumps(result, indent=2))
        
        return result

def main():
    """Main entry point for the attack script."""
    parser = argparse.ArgumentParser(description="SafeKV Time Channel Attack")
    parser.add_argument("--api-url", 
                       default="http://localhost:30000/generate",
                       help="Target LLM API URL")
    parser.add_argument("--output-dir", 
                       default="attack_results",
                       help="Output directory for results")
    parser.add_argument("--categories", 
                       nargs="+",
                       help="Specific categories to attack")
    parser.add_argument("--max-candidates", 
                       type=int, default=15,
                       help="Maximum candidates per set")
    parser.add_argument("--baseline-requests", 
                       type=int, default=15,
                       help="Number of baseline requests")
    parser.add_argument("--request-delay", 
                       type=float, default=0.2,
                       help="Delay between requests")
    parser.add_argument("--demo", 
                       action="store_true",
                       help="Run demonstration attack only")
    
    args = parser.parse_args()
    
    # Initialize attack orchestrator
    attack = SafeKVTimeChannelAttack(
        api_url=args.api_url,
        output_dir=args.output_dir,
        baseline_requests=args.baseline_requests,
        request_delay=args.request_delay
    )
    
    async def run_attack():
        if args.demo:
            await attack.run_demo_attack()
        else:
            await attack.run_comprehensive_attack(
                target_categories=args.categories,
                max_candidates_per_set=args.max_candidates
            )
    
    # Run the attack
    try:
        asyncio.run(run_attack())
    except KeyboardInterrupt:
        logger.info("Attack interrupted by user")
    except Exception as e:
        logger.error(f"Attack failed: {e}")
        raise

if __name__ == "__main__":
    main() 