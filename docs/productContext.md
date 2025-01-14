# Product Context

## Project Purpose
This project implements an entity detection system that can identify and classify entities in text using semantic similarity and reference matching. The system is designed to work with streaming text input and supports progressive learning through reference entities.

## Problems Solved
1. Real-time entity detection in streaming text
2. Context-aware entity classification
3. Progressive improvement of detection accuracy
4. Handling of both exact and semantic matches
5. Support for multiple entity types/archetypes

## How It Works
1. **Entity Detection**
   - Uses embedding-based similarity matching
   - Supports both archetype and reference matching
   - Handles multi-word and single-word entities
   - Progressive confidence scoring system

2. **Reference System**
   - Maintains reference entities for improved matching
   - Learns from high-confidence matches
   - Supports context-aware detection

3. **Confidence Scoring**
   - Dynamic threshold adjustment
   - Reference-based confidence boosting
   - Context-aware scoring adjustments
   - Protection against false positives 