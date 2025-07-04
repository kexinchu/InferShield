#!/usr/bin/env python3
"""
Candidate Generator for SafeKV Time Channel Attacks

This module generates various types of candidate sets for time channel attacks
against SafeKV systems. It creates realistic candidate sets for different types
of private information that might be cached in KV-Cache.

Author: Security Research Team
Date: 2024
"""

import random
import string
import re
from typing import List, Dict, Set, Optional
from dataclasses import dataclass
import json

@dataclass
class CandidateSet:
    """Represents a set of candidates for attack"""
    name: str
    description: str
    candidates: List[str]
    category: str
    expected_pattern: str
    confidence_threshold: float = 0.8

class CandidateGenerator:
    """
    Generator for various types of candidate sets used in time channel attacks.
    """
    
    def __init__(self):
        """Initialize the candidate generator"""
        self.common_names = [
            "john", "jane", "michael", "sarah", "david", "emma", "james", "olivia",
            "robert", "sophia", "william", "ava", "richard", "isabella", "joseph",
            "mia", "thomas", "charlotte", "christopher", "amelia", "charles", "harper",
            "daniel", "evelyn", "matthew", "abigail", "anthony", "emily", "mark",
            "elizabeth", "donald", "sofia", "steven", "madison", "paul", "avery",
            "andrew", "ella", "joshua", "scarlett", "kenneth", "grace", "kevin",
            "chloe", "brian", "victoria", "george", "riley", "edward", "aria"
        ]
        
        self.common_domains = [
            "gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "icloud.com",
            "company.com", "corp.com", "enterprise.com", "business.com", "org.com",
            "university.edu", "college.edu", "school.edu", "institute.edu"
        ]
        
        self.common_passwords = [
            "password", "123456", "password123", "admin", "admin123", "root",
            "secret", "secret123", "qwerty", "abc123", "letmein", "welcome",
            "monkey", "dragon", "master", "shadow", "superman", "batman"
        ]
        
        self.ssn_patterns = [
            "123-45-6789", "987-65-4321", "111-22-3333", "444-55-6666",
            "777-88-9999", "000-11-2222", "333-44-5555", "666-77-8888"
        ]
        
        self.credit_card_patterns = [
            "4111-1111-1111-1111", "5555-5555-5555-4444", "4000-0000-0000-0002",
            "3782-822463-10005", "3714-496353-98431", "3787-344936-71000"
        ]
        
        self.phone_patterns = [
            "(555) 123-4567", "(555) 987-6543", "(555) 111-2222",
            "(555) 333-4444", "(555) 555-6666", "(555) 777-8888"
        ]
        
        self.address_patterns = [
            "123 Main St", "456 Oak Ave", "789 Pine Rd", "321 Elm St",
            "654 Maple Dr", "987 Cedar Ln", "147 Birch Way", "258 Spruce Ct"
        ]
    
    def generate_email_candidates(self, count: int = 20) -> CandidateSet:
        """
        Generate email address candidates.
        
        Args:
            count: Number of candidates to generate
            
        Returns:
            CandidateSet with email addresses
        """
        candidates = []
        
        # Common email patterns
        for name in self.common_names[:count//2]:
            domain = random.choice(self.common_domains)
            candidates.append(f"{name}@{domain}")
        
        # Add some variations
        for i in range(count - len(candidates)):
            name = random.choice(self.common_names)
            domain = random.choice(self.common_domains)
            # Add numbers or special characters
            if random.random() < 0.3:
                name += str(random.randint(1, 999))
            elif random.random() < 0.2:
                name += random.choice([".", "_", "-"])
            candidates.append(f"{name}@{domain}")
        
        return CandidateSet(
            name="email_addresses",
            description="Email address candidates for attack",
            candidates=candidates,
            category="personal_info",
            expected_pattern=r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
        )
    
    def generate_password_candidates(self, count: int = 20) -> CandidateSet:
        """
        Generate password candidates.
        
        Args:
            count: Number of candidates to generate
            
        Returns:
            CandidateSet with passwords
        """
        candidates = []
        
        # Common passwords
        candidates.extend(self.common_passwords[:count//2])
        
        # Generate variations
        for i in range(count - len(candidates)):
            base = random.choice(self.common_passwords)
            # Add numbers, uppercase, special chars
            if random.random() < 0.4:
                base += str(random.randint(1, 999))
            if random.random() < 0.3:
                base = base.capitalize()
            if random.random() < 0.2:
                base += random.choice(["!", "@", "#", "$", "%"])
            candidates.append(base)
        
        return CandidateSet(
            name="passwords",
            description="Password candidates for attack",
            candidates=candidates,
            category="credentials",
            expected_pattern=r"password|secret|admin|root|qwerty|123456"
        )
    
    def generate_ssn_candidates(self, count: int = 20) -> CandidateSet:
        """
        Generate Social Security Number candidates.
        
        Args:
            count: Number of candidates to generate
            
        Returns:
            CandidateSet with SSNs
        """
        candidates = []
        
        # Common SSN patterns
        candidates.extend(self.ssn_patterns[:count//2])
        
        # Generate random SSNs
        for i in range(count - len(candidates)):
            area = random.randint(1, 999)
            group = random.randint(1, 99)
            serial = random.randint(1, 9999)
            candidates.append(f"{area:03d}-{group:02d}-{serial:04d}")
        
        return CandidateSet(
            name="social_security_numbers",
            description="SSN candidates for attack",
            candidates=candidates,
            category="government_id",
            expected_pattern=r"\d{3}-\d{2}-\d{4}"
        )
    
    def generate_credit_card_candidates(self, count: int = 20) -> CandidateSet:
        """
        Generate credit card number candidates.
        
        Args:
            count: Number of candidates to generate
            
        Returns:
            CandidateSet with credit card numbers
        """
        candidates = []
        
        # Common patterns
        candidates.extend(self.credit_card_patterns[:count//2])
        
        # Generate random credit card numbers
        for i in range(count - len(candidates)):
            # Generate a 16-digit number with proper formatting
            digits = ''.join([str(random.randint(0, 9)) for _ in range(16)])
            formatted = f"{digits[:4]}-{digits[4:8]}-{digits[8:12]}-{digits[12:]}"
            candidates.append(formatted)
        
        return CandidateSet(
            name="credit_card_numbers",
            description="Credit card number candidates for attack",
            candidates=candidates,
            category="financial",
            expected_pattern=r"\d{4}-\d{4}-\d{4}-\d{4}"
        )
    
    def generate_phone_candidates(self, count: int = 20) -> CandidateSet:
        """
        Generate phone number candidates.
        
        Args:
            count: Number of candidates to generate
            
        Returns:
            CandidateSet with phone numbers
        """
        candidates = []
        
        # Common patterns
        candidates.extend(self.phone_patterns[:count//2])
        
        # Generate random phone numbers
        for i in range(count - len(candidates)):
            area_code = random.randint(200, 999)
            prefix = random.randint(100, 999)
            line = random.randint(1000, 9999)
            candidates.append(f"({area_code}) {prefix}-{line}")
        
        return CandidateSet(
            name="phone_numbers",
            description="Phone number candidates for attack",
            candidates=candidates,
            category="contact_info",
            expected_pattern=r"\(\d{3}\) \d{3}-\d{4}"
        )
    
    def generate_address_candidates(self, count: int = 20) -> CandidateSet:
        """
        Generate address candidates.
        
        Args:
            count: Number of candidates to generate
            
        Returns:
            CandidateSet with addresses
        """
        candidates = []
        
        # Common patterns
        candidates.extend(self.address_patterns[:count//2])
        
        # Generate random addresses
        street_numbers = [str(i) for i in range(1, 1000)]
        street_names = ["Main", "Oak", "Pine", "Elm", "Maple", "Cedar", "Birch", "Spruce",
                       "Washington", "Lincoln", "Jefferson", "Adams", "Madison", "Monroe"]
        street_types = ["St", "Ave", "Rd", "Dr", "Ln", "Way", "Ct", "Blvd", "Pl"]
        
        for i in range(count - len(candidates)):
            number = random.choice(street_numbers)
            name = random.choice(street_names)
            street_type = random.choice(street_types)
            candidates.append(f"{number} {name} {street_type}")
        
        return CandidateSet(
            name="addresses",
            description="Address candidates for attack",
            candidates=candidates,
            category="location",
            expected_pattern=r"\d+\s+[A-Za-z]+\s+(St|Ave|Rd|Dr|Ln|Way|Ct|Blvd|Pl)"
        )
    
    def generate_api_key_candidates(self, count: int = 20) -> CandidateSet:
        """
        Generate API key candidates.
        
        Args:
            count: Number of candidates to generate
            
        Returns:
            CandidateSet with API keys
        """
        candidates = []
        
        # Common API key patterns
        common_keys = [
            "sk-1234567890abcdef1234567890abcdef1234567890abcdef",
            "pk_1234567890abcdef1234567890abcdef1234567890abcdef",
            "api_key_1234567890abcdef1234567890abcdef1234567890abcdef",
            "token_1234567890abcdef1234567890abcdef1234567890abcdef"
        ]
        
        candidates.extend(common_keys[:count//2])
        
        # Generate random API keys
        for i in range(count - len(candidates)):
            # Generate a 48-character hex string
            key = ''.join(random.choices('0123456789abcdef', k=48))
            prefix = random.choice(["sk-", "pk_", "api_key_", "token_"])
            candidates.append(f"{prefix}{key}")
        
        return CandidateSet(
            name="api_keys",
            description="API key candidates for attack",
            candidates=candidates,
            category="credentials",
            expected_pattern=r"(sk-|pk_|api_key_|token_)[a-f0-9]{48}"
        )
    
    def generate_company_info_candidates(self, count: int = 20) -> CandidateSet:
        """
        Generate company information candidates.
        
        Args:
            count: Number of candidates to generate
            
        Returns:
            CandidateSet with company information
        """
        candidates = []
        
        companies = [
            "Apple Inc", "Microsoft Corporation", "Google LLC", "Amazon.com Inc",
            "Facebook Inc", "Tesla Inc", "Netflix Inc", "Twitter Inc",
            "Uber Technologies", "Airbnb Inc", "Spotify Technology", "Slack Technologies"
        ]
        
        # Company names
        candidates.extend(companies[:count//3])
        
        # Project codes
        for i in range(count//3):
            project = f"Project-{chr(65 + i % 26)}{random.randint(100, 999)}"
            candidates.append(project)
        
        # Internal codes
        for i in range(count - len(candidates)):
            internal_code = f"INT-{random.randint(1000, 9999)}-{chr(65 + random.randint(0, 25))}"
            candidates.append(internal_code)
        
        return CandidateSet(
            name="company_info",
            description="Company information candidates for attack",
            candidates=candidates,
            category="business",
            expected_pattern=r"(Project-|INT-|Inc|Corporation|LLC)"
        )
    
    def generate_medical_info_candidates(self, count: int = 20) -> CandidateSet:
        """
        Generate medical information candidates.
        
        Args:
            count: Number of candidates to generate
            
        Returns:
            CandidateSet with medical information
        """
        candidates = []
        
        conditions = [
            "diabetes", "hypertension", "asthma", "depression", "anxiety",
            "arthritis", "cancer", "heart disease", "stroke", "obesity"
        ]
        
        medications = [
            "insulin", "metformin", "lisinopril", "atorvastatin", "aspirin",
            "ibuprofen", "acetaminophen", "omeprazole", "albuterol", "sertraline"
        ]
        
        # Medical conditions
        candidates.extend(conditions[:count//2])
        
        # Medications
        candidates.extend(medications[:count//2])
        
        # Generate variations
        for i in range(count - len(candidates)):
            if random.random() < 0.5:
                condition = random.choice(conditions)
                candidates.append(f"diagnosed with {condition}")
            else:
                med = random.choice(medications)
                candidates.append(f"prescribed {med}")
        
        return CandidateSet(
            name="medical_info",
            description="Medical information candidates for attack",
            candidates=candidates,
            category="health",
            expected_pattern=r"(diabetes|hypertension|asthma|depression|insulin|metformin)"
        )
    
    def generate_all_candidate_sets(self) -> Dict[str, CandidateSet]:
        """
        Generate all types of candidate sets.
        
        Returns:
            Dictionary mapping category names to CandidateSet objects
        """
        return {
            "emails": self.generate_email_candidates(),
            "passwords": self.generate_password_candidates(),
            "ssns": self.generate_ssn_candidates(),
            "credit_cards": self.generate_credit_card_candidates(),
            "phones": self.generate_phone_candidates(),
            "addresses": self.generate_address_candidates(),
            "api_keys": self.generate_api_key_candidates(),
            "company_info": self.generate_company_info_candidates(),
            "medical_info": self.generate_medical_info_candidates()
        }
    
    def save_candidate_sets(self, filename: str = "candidate_sets.json"):
        """
        Save all candidate sets to a JSON file.
        
        Args:
            filename: Output filename
        """
        candidate_sets = self.generate_all_candidate_sets()
        
        data = {}
        for name, candidate_set in candidate_sets.items():
            data[name] = {
                "name": candidate_set.name,
                "description": candidate_set.description,
                "candidates": candidate_set.candidates,
                "category": candidate_set.category,
                "expected_pattern": candidate_set.expected_pattern,
                "confidence_threshold": candidate_set.confidence_threshold
            }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved {len(candidate_sets)} candidate sets to {filename}")
    
    def load_candidate_sets(self, filename: str = "candidate_sets.json") -> Dict[str, CandidateSet]:
        """
        Load candidate sets from a JSON file.
        
        Args:
            filename: Input filename
            
        Returns:
            Dictionary mapping category names to CandidateSet objects
        """
        with open(filename, 'r') as f:
            data = json.load(f)
        
        candidate_sets = {}
        for name, candidate_data in data.items():
            candidate_sets[name] = CandidateSet(
                name=candidate_data["name"],
                description=candidate_data["description"],
                candidates=candidate_data["candidates"],
                category=candidate_data["category"],
                expected_pattern=candidate_data["expected_pattern"],
                confidence_threshold=candidate_data.get("confidence_threshold", 0.8)
            )
        
        return candidate_sets

def main():
    """Generate and save candidate sets"""
    generator = CandidateGenerator()
    generator.save_candidate_sets()
    
    # Print summary
    candidate_sets = generator.generate_all_candidate_sets()
    print(f"\nGenerated {len(candidate_sets)} candidate sets:")
    for name, candidate_set in candidate_sets.items():
        print(f"  {name}: {len(candidate_set.candidates)} candidates")

if __name__ == "__main__":
    main() 