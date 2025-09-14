#!/usr/bin/env python3
"""
Graph Extractor Demo Script

This script demonstrates various use cases of the graph extractor test script
and shows how to analyze the results for prompt engineering and fine-tuning.

Usage:
    python scripts/demo_graph_extractor.py
"""

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any


def run_command(cmd: str) -> tuple[int, str, str]:
    """Run a shell command and return exit code, stdout, stderr."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, cwd=Path.cwd()
        )
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return 1, "", str(e)


def create_test_documents():
    """Create various test documents for demonstration."""
    
    # Simple document
    simple_doc = """# Simple Test Document

John Smith works at Microsoft Corporation in Seattle, Washington. 
He is a software engineer who develops cloud applications using Azure.
Microsoft is headquartered in Redmond and was founded by Bill Gates and Paul Allen.
"""
    
    # Complex document with multiple entity types
    complex_doc = """# Research Paper: AI in Healthcare

## Abstract

Dr. Sarah Johnson from Stanford University published groundbreaking research on artificial intelligence applications in medical diagnosis. The study, conducted at Stanford Medical Center, examined the effectiveness of machine learning algorithms in detecting cancer from medical imaging.

## Methodology

The research team, including Dr. Michael Chen (UCLA) and Dr. Emily Rodriguez (Mayo Clinic), analyzed 10,000 medical images using a convolutional neural network developed by Google DeepMind. The AI system, called MedNet-AI, was trained on data from three major hospitals:

- Johns Hopkins Hospital (Baltimore, Maryland)
- Massachusetts General Hospital (Boston, Massachusetts)  
- Cleveland Clinic (Cleveland, Ohio)

## Results

The AI system achieved 95% accuracy in detecting lung cancer, significantly outperforming traditional diagnostic methods. The study received funding from the National Institutes of Health (NIH) and was published in the New England Journal of Medicine.

## Impact

This breakthrough could revolutionize cancer diagnosis, potentially saving thousands of lives annually. The technology is being licensed to pharmaceutical companies including Pfizer and Johnson & Johnson for clinical trials.
"""
    
    # German document
    german_doc = """# Forschungsbericht: KI in der Medizin

## Einleitung

Prof. Dr. Hans Müller von der Universität München führte eine bahnbrechende Studie über künstliche Intelligenz in der Krebsdiagnose durch. Die Forschung wurde am Klinikum der Ludwig-Maximilians-Universität durchgeführt.

## Methodik

Das Forschungsteam, bestehend aus Dr. Anna Schmidt (Charité Berlin) und Dr. Thomas Weber (Universitätsklinikum Hamburg), untersuchte 5.000 Röntgenbilder mit einem neuronalen Netzwerk.

## Ergebnisse

Das KI-System erreichte eine Genauigkeit von 92% bei der Erkennung von Lungenkrebs. Die Studie wurde von der Deutschen Forschungsgemeinschaft (DFG) finanziert und im Deutschen Ärzteblatt veröffentlicht.
"""
    
    # Write test documents
    test_docs = {
        "simple_test.md": simple_doc,
        "complex_test.md": complex_doc,
        "german_test.md": german_doc
    }
    
    for filename, content in test_docs.items():
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
    
    return list(test_docs.keys())


def analyze_extraction_results(filename: str) -> Dict[str, Any]:
    """Analyze extraction results from a JSON file."""
    json_file = Path(filename).with_suffix('.graph.json')
    
    if not json_file.exists():
        return {"error": f"Results file not found: {json_file}"}
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        analysis = data.get('analysis', {})
        extraction_results = data.get('extraction_results', {})
        
        # Extract entity and relation details
        entities = extraction_results.get('entities', [])
        relations = extraction_results.get('relations', [])
        
        # Analyze entity types and confidence
        entity_analysis = {}
        for entity in entities:
            entity_type = entity.get('type', 'UNKNOWN')
            confidence = entity.get('confidence', 0.0)
            
            if entity_type not in entity_analysis:
                entity_analysis[entity_type] = {
                    'count': 0,
                    'confidences': [],
                    'examples': []
                }
            
            entity_analysis[entity_type]['count'] += 1
            entity_analysis[entity_type]['confidences'].append(confidence)
            entity_analysis[entity_type]['examples'].append(entity.get('name', ''))
        
        # Analyze relation types and confidence
        relation_analysis = {}
        for relation in relations:
            relation_type = relation.get('relation_type', 'UNKNOWN')
            confidence = relation.get('confidence', 0.0)
            
            if relation_type not in relation_analysis:
                relation_analysis[relation_type] = {
                    'count': 0,
                    'confidences': [],
                    'examples': []
                }
            
            relation_analysis[relation_type]['count'] += 1
            relation_analysis[relation_type]['confidences'].append(confidence)
            
            # Create relation example
            source_id = relation.get('source_entity_id', '')
            target_id = relation.get('target_entity_id', '')
            source_name = next((e['name'] for e in entities if e['id'] == source_id), source_id)
            target_name = next((e['name'] for e in entities if e['id'] == target_id), target_id)
            relation_analysis[relation_type]['examples'].append(f"{source_name} -> {target_name}")
        
        return {
            "filename": filename,
            "total_entities": len(entities),
            "total_relations": len(relations),
            "entity_analysis": entity_analysis,
            "relation_analysis": relation_analysis,
            "confidence_stats": analysis.get('confidence_stats', {}),
            "language": data.get('language'),
            "content_length": data.get('content_length', 0)
        }
        
    except Exception as e:
        return {"error": f"Error analyzing {json_file}: {e}"}


def print_analysis_report(analysis: Dict[str, Any]):
    """Print a formatted analysis report."""
    if "error" in analysis:
        print(f"[ERROR] {analysis['error']}")
        return

    print(f"\n[REPORT] Analysis Report: {analysis['filename']}")
    print("=" * 60)
    print(f"[OVERVIEW] Summary:")
    print(f"   - Content length: {analysis['content_length']} characters")
    print(f"   - Language: {analysis.get('language', 'auto-detect')}")
    print(f"   - Total entities: {analysis['total_entities']}")
    print(f"   - Total relations: {analysis['total_relations']}")

    print(f"\n[ENTITIES] Entity Analysis:")
    for entity_type, data in analysis['entity_analysis'].items():
        avg_conf = sum(data['confidences']) / len(data['confidences']) if data['confidences'] else 0
        examples = ', '.join(data['examples'][:3])
        if len(data['examples']) > 3:
            examples += f" (and {len(data['examples']) - 3} more)"
        print(f"   - {entity_type}: {data['count']} entities, avg confidence: {avg_conf:.2f}")
        print(f"     Examples: {examples}")

    print(f"\n[RELATIONS] Relation Analysis:")
    for relation_type, data in analysis['relation_analysis'].items():
        avg_conf = sum(data['confidences']) / len(data['confidences']) if data['confidences'] else 0
        examples = ', '.join(data['examples'][:2])
        if len(data['examples']) > 2:
            examples += f" (and {len(data['examples']) - 2} more)"
        print(f"   - {relation_type}: {data['count']} relations, avg confidence: {avg_conf:.2f}")
        print(f"     Examples: {examples}")

    # Confidence statistics
    conf_stats = analysis.get('confidence_stats', {})
    if conf_stats:
        print(f"\n[CONFIDENCE] Statistics:")
        if 'entity_confidence' in conf_stats:
            ec = conf_stats['entity_confidence']
            print(f"   - Entity confidence: {ec.get('avg', 0):.2f} (range: {ec.get('min', 0):.2f}-{ec.get('max', 0):.2f})")
        if 'relation_confidence' in conf_stats:
            rc = conf_stats['relation_confidence']
            print(f"   - Relation confidence: {rc.get('avg', 0):.2f} (range: {rc.get('min', 0):.2f}-{rc.get('max', 0):.2f})")


def main():
    """Main demo function."""
    print("[DEMO] Graph Extractor Demo")
    print("=" * 60)

    # Create test documents
    print("[SETUP] Creating test documents...")
    test_files = create_test_documents()
    print(f"[OK] Created {len(test_files)} test documents")

    # Check if we can run real extractions or just dry-run
    has_api_key = bool(os.getenv('GEMINI_API_KEY'))
    mode = "real extraction" if has_api_key else "dry-run mode"
    print(f"[MODE] Running in {mode}")
    
    # Run extractions on each test document
    for test_file in test_files:
        print(f"\n[PROCESS] Processing {test_file}...")

        # Determine language parameter
        language_param = ""
        if "german" in test_file:
            language_param = "--language de"
        elif "simple" in test_file or "complex" in test_file:
            language_param = "--language en"

        # Build command
        dry_run_param = "" if has_api_key else "--dry-run"
        cmd = f"python scripts/test_graph_extractor.py {test_file} {language_param} {dry_run_param}"

        # Run extraction
        exit_code, stdout, stderr = run_command(cmd)

        if exit_code == 0:
            print(f"[OK] Extraction completed for {test_file}")

            # Validate output
            json_file = Path(test_file).with_suffix('.graph.json')
            validate_cmd = f"python scripts/validate_graph_output.py {json_file}"
            val_exit_code, val_stdout, val_stderr = run_command(validate_cmd)

            if val_exit_code == 0:
                print(f"[OK] Validation passed for {test_file}")

                # Analyze results
                analysis = analyze_extraction_results(test_file)
                print_analysis_report(analysis)
            else:
                print(f"[ERROR] Validation failed for {test_file}: {val_stderr}")
        else:
            print(f"[ERROR] Extraction failed for {test_file}: {stderr}")
    
    print(f"\n[SUMMARY] Demo Summary:")
    print(f"   - Processed {len(test_files)} documents")
    print(f"   - Mode: {mode}")
    print(f"   - Output files: {', '.join([f'{f}.graph.json' for f in test_files])}")

    if not has_api_key:
        print(f"\n[INFO] To run with real API calls:")
        print(f"   export GEMINI_API_KEY='your-api-key'")
        print(f"   python scripts/demo_graph_extractor.py")

    print(f"\n[NEXT] Next steps:")
    print(f"   1. Review the generated JSON files")
    print(f"   2. Analyze entity and relation types")
    print(f"   3. Check confidence scores")
    print(f"   4. Fine-tune system prompts if needed")
    print(f"   5. Test with your own documents")


if __name__ == "__main__":
    main()
