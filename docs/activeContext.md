# Active Context

## Current Work
Working on improving entity detection confidence scoring and test reliability. Specifically:
1. Addressing test failures in `test_archetype_api.py`
2. Improving confidence scoring for location entities
3. Implementing a more robust approach to threshold management

## Recent Changes
1. Modified confidence calculation in `StreamingEntityDetector`
2. Adjusted threshold handling for multi-word phrases
3. Added context-aware bonuses for related terms
4. Enhanced test cases for real API testing

## Next Steps
1. Implement database-backed configuration for thresholds
2. Create background task for entity verification
3. Add metrics tracking for detection performance
4. Enhance confidence scoring with statistical analysis 