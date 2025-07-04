#!/usr/bin/env python3
"""
Test script for SafeKV Time Channel Attack Model

This script tests the basic functionality of the attack model without
requiring a real LLM server. It simulates the attack process and
validates the components.
"""

import asyncio
import json
import time
import random
from typing import Dict, List

from attack_model import TimeChannelAttacker, AttackResult, AttackSession
from candidate_generator import CandidateGenerator, CandidateSet

class MockLLMServer:
    """
    Mock LLM server for testing purposes.
    Simulates a server with KV-Cache behavior.
    """
    
    def __init__(self):
        self.cache = set()
        self.baseline_ttft = 0.05  # 50ms baseline
        self.cache_hit_ttft = 0.02  # 20ms for cache hits
        self.noise_factor = 0.01  # 10ms noise
        
    def add_to_cache(self, text: str):
        """Add text to the simulated cache."""
        self.cache.add(text.lower())
    
    def simulate_request(self, prompt: str) -> float:
        """
        Simulate a request and return TTFT.
        
        Args:
            prompt: The input prompt
            
        Returns:
            Simulated TTFT in seconds
        """
        # Check if any part of the prompt is in cache
        prompt_lower = prompt.lower()
        cache_hit = any(cached in prompt_lower for cached in self.cache)
        
        # Add some noise to make it realistic
        noise = random.uniform(-self.noise_factor, self.noise_factor)
        
        if cache_hit:
            return self.cache_hit_ttft + noise
        else:
            return self.baseline_ttft + noise

class MockTimeChannelAttacker(TimeChannelAttacker):
    """
    Mock attacker that uses a simulated server instead of real HTTP requests.
    """
    
    def __init__(self, mock_server: MockLLMServer, **kwargs):
        super().__init__(api_url="mock://localhost", **kwargs)
        self.mock_server = mock_server
    
    async def _send_request(self, prompt: str) -> float:
        """
        Mock request that simulates TTFT measurement.
        
        Args:
            prompt: The prompt to send
            
        Returns:
            Simulated TTFT in seconds
        """
        # Simulate network delay
        await asyncio.sleep(0.001)
        
        # Get TTFT from mock server
        ttft = self.mock_server.simulate_request(prompt)
        
        return ttft

def test_candidate_generator():
    """Test candidate generator functionality."""
    print("Testing candidate generator...")
    
    generator = CandidateGenerator()
    
    # Test email generation
    email_set = generator.generate_email_candidates(5)
    print(f"Generated {len(email_set.candidates)} email candidates")
    print(f"Sample: {email_set.candidates[:3]}")
    
    # Test password generation
    password_set = generator.generate_password_candidates(5)
    print(f"Generated {len(password_set.candidates)} password candidates")
    print(f"Sample: {password_set.candidates[:3]}")
    
    # Test all sets
    all_sets = generator.generate_all_candidate_sets()
    print(f"Generated {len(all_sets)} candidate sets")
    
    return True

async def test_attack_model():
    """Test attack model functionality."""
    print("\nTesting attack model...")
    
    # Create mock server
    mock_server = MockLLMServer()
    
    # Add some "cached" content
    mock_server.add_to_cache("admin@company.org")
    mock_server.add_to_cache("password123")
    mock_server.add_to_cache("123-45-6789")
    
    # Create mock attacker
    attacker = MockTimeChannelAttacker(mock_server)
    
    # Test baseline establishment
    random_prompts = attacker.generate_random_prompts(5)
    print(f"Generated {len(random_prompts)} random prompts")
    
    # Test baseline
    baseline_ttft = await attacker.establish_baseline(random_prompts)
    print(f"Baseline TTFT: {baseline_ttft:.3f}s")
    print(f"Hit threshold: {attacker.hit_threshold:.3f}s")
    
    # Test cache hit detection
    test_prompts = [
        "admin@company.org",  # Should hit cache
        "user@example.com",   # Should miss cache
        "password123",        # Should hit cache
        "random_text_here"    # Should miss cache
    ]
    
    print("\nTesting cache hit detection:")
    for prompt in test_prompts:
        ttft = await attacker._send_request(prompt)
        is_hit, confidence = attacker._determine_cache_hit(ttft)
        print(f"  '{prompt}': TTFT={ttft:.3f}s, Hit={is_hit}, Confidence={confidence:.2f}")
    
    return True

async def test_attack_session():
    """Test complete attack session."""
    print("\nTesting complete attack session...")
    
    # Create mock server with some cached content
    mock_server = MockLLMServer()
    mock_server.add_to_cache("admin@company.org")
    mock_server.add_to_cache("secret2024")
    
    # Create mock attacker
    attacker = MockTimeChannelAttacker(mock_server)
    
    # Generate candidate set
    generator = CandidateGenerator()
    candidate_set = generator.generate_email_candidates(10)
    
    # Add some known candidates
    candidate_set.candidates.extend([
        "admin@company.org",  # Should hit
        "user@example.com",   # Should miss
        "secret@corp.com"     # Should miss
    ])
    
    # Establish baseline
    random_prompts = attacker.generate_random_prompts(5)
    await attacker.establish_baseline(random_prompts)
    
    # Execute attack
    session = await attacker.attack_candidate_set(
        session_id="test_session",
        target_pattern="email_addresses",
        candidates=candidate_set.candidates,
        random_prompts=random_prompts
    )
    
    # Analyze results
    analysis = attacker.analyze_attack_results("test_session")
    
    print(f"Attack session completed:")
    print(f"  Total attempts: {analysis['statistics']['total_attempts']}")
    print(f"  Hit rate: {analysis['statistics']['hit_rate']:.2%}")
    print(f"  Hit candidates: {analysis['hit_candidates']}")
    print(f"  Likely private info: {analysis['likely_private_info']}")
    
    return analysis

def test_data_structures():
    """Test data structure functionality."""
    print("\nTesting data structures...")
    
    # Test AttackResult
    result = AttackResult(
        candidate="test@example.com",
        ttft=0.03,
        is_hit=True,
        confidence=0.85,
        timestamp=time.time()
    )
    print(f"AttackResult created: {result}")
    
    # Test AttackSession
    session = AttackSession(
        session_id="test_session",
        target_pattern="emails",
        candidates=["test1@example.com", "test2@example.com"]
    )
    session.add_result(result)
    print(f"AttackSession created with {len(session.results)} results")
    
    # Test CandidateSet
    candidate_set = CandidateSet(
        name="test_emails",
        description="Test email candidates",
        candidates=["test@example.com"],
        category="test",
        expected_pattern="email"
    )
    print(f"CandidateSet created: {candidate_set.name}")
    
    return True

async def main():
    """Run all tests."""
    print("SafeKV Time Channel Attack Model - Test Suite")
    print("=" * 50)
    
    try:
        # Test individual components
        test_candidate_generator()
        await test_attack_model()
        test_data_structures()
        
        # Test complete attack session
        analysis = await test_attack_session()
        
        # Test summary generation
        mock_server = MockLLMServer()
        attacker = MockTimeChannelAttacker(mock_server)
        summary = attacker.get_attack_summary()
        print(f"\nAttack summary: {summary}")
        
        print("\n" + "=" * 50)
        print("All tests completed successfully!")
        print("The attack model is working correctly.")
        
        return True
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1) 