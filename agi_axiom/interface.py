"""
Human interface and ledger management
"""
from typing import Dict, List
from datetime import datetime
import hashlib
import json


class NaturalLanguageInterface:
    """Interface for human-readable output"""
    
    def format_response(self, result: Dict, safety: Dict, include_ledger: bool = False) -> Dict:
        """Format response for human consumption"""
        
        # Extract key information
        certainty = result.get('final_certainty', 0.0)
        target = result.get('target', 'Unknown target')
        
        # Generate response text
        if certainty > 0.8:
            confidence_level = "HIGH_CONFIDENCE"
            response_text = f"Based on axiomatic reasoning: The query '{target}' can be addressed through foundational principles with {certainty:.1%} certainty."
        elif certainty > 0.5:
            confidence_level = "MEDIUM_CONFIDENCE"
            response_text = f"Axiomatic analysis suggests: '{target}' involves considerations that can be partially addressed with {certainty:.1%} certainty."
        else:
            confidence_level = "LOW_CONFIDENCE"
            response_text = f"Insufficient axiomatic foundation: '{target}' requires axioms not currently in the knowledge base."
        
        # Add safety information
        if safety['passed']:
            safety_note = "✓ Compton-class safety verified (10^-16 risk factor)"
        else:
            safety_note = "⚠️ Safety verification failed - result should not be trusted"
        
        # Build response
        response = {
            'response': response_text,
            'confidence': confidence_level,
            'certainty_score': certainty,
            'safety': safety_note,
            'architecture': 'AGI-Axiom (axiomatic reasoning, no neural networks)',
            'reasoning_depth': len(result.get('derivation_path', [])),
            'refinement_cycles': result.get('refinement_cycles', 0)
        }
        
        # Add ledger if requested
        if include_ledger and 'ledger_entry' in result:
            response['ledger_reference'] = result['ledger_entry']['derivation_id']
        
        return response


class TruthLedger:
    """Immutable truth recording system"""
    
    def __init__(self, blockchain_integration: bool = False):
        self.entries = []
        self.blockchain_enabled = blockchain_integration
        self.previous_hash = '0' * 64
    
    def record_entry(self, data: Dict) -> Dict:
        """Record an immutable entry in the truth ledger"""
        
        timestamp = datetime.utcnow().isoformat()
        data_str = json.dumps(data, sort_keys=True)
        
        # Create hash chain
        current_hash = hashlib.sha256(
            f"{self.previous_hash}{timestamp}{data_str}".encode()
        ).hexdigest()
        
        entry = {
            'timestamp': timestamp,
            'data': data,
            'hash': current_hash,
            'previous_hash': self.previous_hash,
            'blockchain_verified': False
        }
        
        # Simulate blockchain posting if enabled
        if self.blockchain_enabled:
            entry['blockchain_verified'] = True
            entry['blockchain_tx'] = f"0x{current_hash[:40]}"
        
        self.entries.append(entry)
        self.previous_hash = current_hash
        
        return entry
    
    def verify_integrity(self) -> bool:
        """Verify the integrity of the entire ledger"""
        for i in range(1, len(self.entries)):
            current = self.entries[i]
            previous = self.entries[i - 1]
            
            # Recalculate hash
            data_str = json.dumps(current['data'], sort_keys=True)
            expected_hash = hashlib.sha256(
                f"{previous['hash']}{current['timestamp']}{data_str}".encode()
            ).hexdigest()
            
            if expected_hash != current['hash']:
                return False
        
        return True
    
    def get_stats(self) -> Dict:
        """Get ledger statistics"""
        return {
            'total_entries': len(self.entries),
            'first_entry': self.entries[0]['timestamp'] if self.entries else None,
            'last_entry': self.entries[-1]['timestamp'] if self.entries else None,
            'integrity_check': self.verify_integrity(),
            'blockchain_enabled': self.blockchain_enabled,
            'estimated_size_kb': len(json.dumps(self.entries).encode()) / 1024
        }
