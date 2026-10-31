import math
from typing import Dict, List, Optional
import time

class FusionState:
    def __init__(self, decay_rate: float = 0.1, threshold_escalate: float = 5.0, threshold_deescalate: float = 3.0):
        self.decay_rate = decay_rate
        self.threshold_escalate = threshold_escalate
        self.threshold_deescalate = threshold_deescalate
        self.banked_audio_duration = 0.0
        
        self.last_update_time = time.time()
        self.llr_spoof = 0.0
        self.llr_target = 0.0
        self.risk_score = 0.0
        
        self.cusum_score = 0.0
        self.is_escalated = False

    def update(self, branch_scores: Dict[str, float], duration: float, current_time: float = None):
        if current_time is None:
            current_time = time.time()
            
        dt = current_time - self.last_update_time
        self.last_update_time = current_time
        
        # Exponential decay for staleness
        decay_factor = math.exp(-self.decay_rate * dt)
        
        # Update LLR with new evidence
        branch_llr = sum(branch_scores.values())
        
        self.llr_spoof = self.llr_spoof * decay_factor + branch_llr
        self.banked_audio_duration = min(10.0, self.banked_audio_duration + duration)
        
        self.risk_score = self.llr_spoof * self.banked_audio_duration
        
        # CUSUM change point detection
        self.cusum_score = max(0.0, self.cusum_score + branch_llr - 0.5)
        
        # Hysteresis
        if self.risk_score > self.threshold_escalate:
            self.is_escalated = True
        elif self.risk_score < self.threshold_deescalate:
            self.is_escalated = False
            
    def get_state(self) -> Dict:
        return {
            "llr_spoof": self.llr_spoof,
            "risk_score": self.risk_score,
            "cusum_score": self.cusum_score,
            "is_escalated": self.is_escalated
        }
