"""SWE-bench specific reward computation."""

import json
import re
from typing import Dict, Any


def evaluate_swe_bench(generated_response: str, verification_info: Dict[str, Any]) -> float:
    """
    Evaluate SWE-bench patch generation.
    
    For now, this is a placeholder that returns a simple reward based on
    whether the response looks like a valid patch.
    
    In a real implementation, this would:
    1. Apply the generated patch to the repository
    2. Run the fail_to_pass tests
    3. Ensure pass_to_pass tests still pass
    4. Return 1.0 if all tests pass, 0.0 otherwise
    """
    
    # Debug logging
    from zeroband.utils.logger import get_logger
    logger = get_logger("SWE_REWARD")
    logger.debug(f"Generated response length: {len(generated_response)}")
    logger.debug(f"Generated response preview: {generated_response[:200]}...")
    logger.debug(f"Verification info: {verification_info}")
    
    # For debugging: give varied rewards to create learning signal
    import random
    basic_reward = 0.05 + random.random() * 0.1  # Random reward between 0.05-0.15
    
    # Check if the response contains a diff/patch
    if "diff --git" in generated_response or "---" in generated_response and "+++" in generated_response:
        # Basic validation - contains patch-like content
        reward = 0.5
        
        # Bonus if it mentions the correct file paths from ground truth
        if "ground_truth" in verification_info:
            ground_truth = verification_info["ground_truth"]
            # Extract file paths from ground truth
            file_paths = re.findall(r'[ab]/(.+?)\s', ground_truth)
            
            # Check if generated response mentions these files
            for file_path in file_paths:
                if file_path in generated_response:
                    reward += 0.1
            
        # Cap at 1.0
        reward = min(reward, 1.0)
        logger.debug(f"Patch found, reward: {reward}")
        return reward
    
    # No patch found - return basic reward for debugging
    logger.debug(f"No patch found, returning basic reward: {basic_reward}")
    return basic_reward