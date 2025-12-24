

import os
import csv
import cv2
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ViolationLogger:
    
    def __init__(self, log_file: str, snapshot_dir: str, cooldown_minutes: int = 5,
                 log_unknown: bool = False):
        self.log_file = log_file
        self.snapshot_dir = snapshot_dir
        self.cooldown_minutes = cooldown_minutes
        self.log_unknown = log_unknown
        
        # Track last violation time: {(person_name, violation_type): datetime}
        self.last_violations: Dict[Tuple[str, str], datetime] = {}
        
        # Create directories and files
        self._initialize()
    
    def _initialize(self):
        """Initialize log file and directories"""
        os.makedirs(self.snapshot_dir, exist_ok=True)
        
        # Create CSV file with headers if it doesn't exist
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'employee_name', 'violation_type', 
                               'confidence', 'snapshot_path'])
            logger.info(f"Created violation log file: {self.log_file}")
    
    def should_log_violation(self, person_name: str, violation_type: str) -> bool:
        """Check if violation should be logged based on cooldown"""
        
        # Skip unknown persons if configured
        if person_name == "Unknown" and not self.log_unknown:
            return False
        
        # Check cooldown
        key = (person_name, violation_type)
        if key in self.last_violations:
            time_diff = datetime.now() - self.last_violations[key]
            if time_diff < timedelta(minutes=self.cooldown_minutes):
                return False
        
        return True
    
    def log_violation(self, person_name: str, violation_type: str, 
                     confidence: float, frame) -> Optional[str]:
        
        if not self.should_log_violation(person_name, violation_type):
            return None
        
        try:
            # Generate snapshot filename
            timestamp = datetime.now()
            timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
            safe_name = person_name.replace(" ", "_").lower()
            snapshot_filename = f"{safe_name}_{violation_type.lower()}_{timestamp_str}.jpg"
            snapshot_path = os.path.join(self.snapshot_dir, snapshot_filename)
            
            # Save snapshot
            cv2.imwrite(snapshot_path, frame)
            
            # Log to CSV
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    person_name,
                    violation_type,
                    f"{confidence:.2f}",
                    snapshot_path
                ])
            
            # Update last violation time
            self.last_violations[(person_name, violation_type)] = timestamp
            
            logger.info(f"Logged violation: {person_name} - {violation_type}")
            return snapshot_path
            
        except Exception as e:
            logger.error(f"Error logging violation: {e}")
            return None
    
    def check_and_log_violations(self, ppe_status: Dict, face_results: list, 
                                 frame) -> int:
        """
        Check for violations and log them
        
        Returns: number of violations logged
        """
        violations_logged = 0
        
        # Only check if person detected
        if not ppe_status.get('person', False):
            return violations_logged
        
        # Determine violations
        violations = []
        if not ppe_status.get('hardhat', False):
            violations.append('NO-Hardhat')
        if not ppe_status.get('mask', False):
            violations.append('NO-Mask')
        if not ppe_status.get('safety_vest', False):
            violations.append('NO-Safety_Vest')
        
        # No violations found
        if not violations:
            return violations_logged
        
        # Get person name (use first recognized face, or "Unknown")
        person_name = "Unknown"
        confidence = 0.0
        
        if face_results:
            person_name = face_results[0]['identity']
            confidence = face_results[0]['confidence']
        
        # Combine all violations into one string
        violation_types = ', '.join(violations)
        
        logger.info(f"Violations detected: {violations} for {person_name}")
        
        # Log once with all violations (one snapshot for all)
        result = self.log_violation(person_name, violation_types, confidence, frame)
        if result:
            violations_logged = 1
            logger.info(f"Successfully logged: {violation_types}")
        else:
            logger.info(f"Skipped logging: {violation_types} (cooldown or unknown)")
        
        return violations_logged
