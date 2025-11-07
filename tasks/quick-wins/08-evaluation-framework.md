# Quick Win 8: Basic Evaluation Framework

## Overview

**Priority**: 📋 **Planned** (2 weeks, Medium Impact, Medium ROI)
**Source**: GraphRAG adaptive benchmarking concepts
**Expected Impact**: Visibility into system performance and regression detection

## Problem Statement

MoRAG currently lacks systematic evaluation capabilities:
- No automated quality checks for entity extraction accuracy
- No query response relevance scoring
- No graph connectivity and completeness metrics
- No regression detection when changes are made
- No benchmarking against baseline performance
- No user satisfaction tracking

This makes it difficult to assess system performance, detect regressions, and measure improvements.

## Solution Overview

Implement a basic evaluation framework with automated quality checks, LLM-based response scoring, graph metrics, and regression detection to provide visibility into system performance and enable data-driven improvements.

## Technical Implementation

### 1. Evaluation Framework Core

Create `packages/morag-core/src/morag_core/evaluation/evaluation_framework.py`:

```python
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio
import json
from abc import ABC, abstractmethod

class EvaluationMetric(Enum):
    ENTITY_EXTRACTION_ACCURACY = "entity_extraction_accuracy"
    RESPONSE_RELEVANCE = "response_relevance"
    RESPONSE_COMPLETENESS = "response_completeness"
    GRAPH_CONNECTIVITY = "graph_connectivity"
    QUERY_PROCESSING_TIME = "query_processing_time"
    USER_SATISFACTION = "user_satisfaction"

@dataclass
class EvaluationResult:
    metric: EvaluationMetric
    score: float
    max_score: float
    details: Dict[str, Any]
    timestamp: datetime
    test_case_id: Optional[str] = None

@dataclass
class TestCase:
    id: str
    name: str
    description: str
    input_data: Dict[str, Any]
    expected_output: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EvaluationReport:
    test_suite_name: str
    timestamp: datetime
    results: List[EvaluationResult]
    overall_score: float
    passed_tests: int
    total_tests: int
    performance_metrics: Dict[str, float]
    recommendations: List[str]

class BaseEvaluator(ABC):
    @abstractmethod
    async def evaluate(self, test_case: TestCase, system_output: Any) -> EvaluationResult:
        pass

class EvaluationFramework:
    def __init__(self):
        self.evaluators = {}
        self.test_suites = {}
        self.baseline_scores = {}
        self.evaluation_history = []

        # Register default evaluators
        self._register_default_evaluators()

    def register_evaluator(self, metric: EvaluationMetric, evaluator: BaseEvaluator):
        """Register an evaluator for a specific metric."""
        self.evaluators[metric] = evaluator

    def register_test_suite(self, name: str, test_cases: List[TestCase]):
        """Register a test suite."""
        self.test_suites[name] = test_cases

    async def run_evaluation(self,
                           test_suite_name: str,
                           system_components: Dict[str, Any],
                           metrics: List[EvaluationMetric] = None) -> EvaluationReport:
        """Run evaluation on a test suite."""

        if test_suite_name not in self.test_suites:
            raise ValueError(f"Test suite '{test_suite_name}' not found")

        test_cases = self.test_suites[test_suite_name]
        metrics = metrics or list(self.evaluators.keys())

        all_results = []
        performance_metrics = {}

        for test_case in test_cases:
            # Execute system with test case input
            system_output = await self._execute_system(test_case, system_components)

            # Evaluate with each metric
            for metric in metrics:
                if metric in self.evaluators:
                    result = await self.evaluators[metric].evaluate(test_case, system_output)
                    result.test_case_id = test_case.id
                    all_results.append(result)

        # Calculate overall metrics
        overall_score = sum(r.score / r.max_score for r in all_results) / len(all_results) if all_results else 0
        passed_tests = sum(1 for r in all_results if r.score / r.max_score >= 0.7)  # 70% threshold

        # Calculate performance metrics
        processing_times = [r.details.get('processing_time', 0) for r in all_results if 'processing_time' in r.details]
        if processing_times:
            performance_metrics['avg_processing_time'] = sum(processing_times) / len(processing_times)
            performance_metrics['max_processing_time'] = max(processing_times)

        # Generate recommendations
        recommendations = self._generate_recommendations(all_results)

        report = EvaluationReport(
            test_suite_name=test_suite_name,
            timestamp=datetime.now(),
            results=all_results,
            overall_score=overall_score,
            passed_tests=passed_tests,
            total_tests=len(test_cases),
            performance_metrics=performance_metrics,
            recommendations=recommendations
        )

        # Store in history
        self.evaluation_history.append(report)

        return report

    async def _execute_system(self, test_case: TestCase, system_components: Dict[str, Any]) -> Dict[str, Any]:
        """Execute system with test case input."""

        input_data = test_case.input_data
        output = {}

        # Measure processing time
        start_time = datetime.now()

        try:
            if 'query' in input_data and 'graph_agent' in system_components:
                # Query processing test
                graph_agent = system_components['graph_agent']
                result = await graph_agent.process_query(
                    input_data['query'],
                    input_data.get('context', {})
                )
                output['query_result'] = result

            elif 'text' in input_data and 'entity_extractor' in system_components:
                # Entity extraction test
                entity_extractor = system_components['entity_extractor']
                entities = await entity_extractor.extract_entities(input_data['text'])
                output['extracted_entities'] = entities

            elif 'document' in input_data and 'document_processor' in system_components:
                # Document processing test
                doc_processor = system_components['document_processor']
                result = await doc_processor.process_document(
                    input_data['document'],
                    input_data.get('metadata', {})
                )
                output['processed_document'] = result

        except Exception as e:
            output['error'] = str(e)
            output['success'] = False
        else:
            output['success'] = True

        end_time = datetime.now()
        output['processing_time'] = (end_time - start_time).total_seconds()

        return output

    def _register_default_evaluators(self):
        """Register default evaluators."""
        self.evaluators[EvaluationMetric.ENTITY_EXTRACTION_ACCURACY] = EntityExtractionEvaluator()
        self.evaluators[EvaluationMetric.RESPONSE_RELEVANCE] = ResponseRelevanceEvaluator()
        self.evaluators[EvaluationMetric.RESPONSE_COMPLETENESS] = ResponseCompletenessEvaluator()
        self.evaluators[EvaluationMetric.GRAPH_CONNECTIVITY] = GraphConnectivityEvaluator()
        self.evaluators[EvaluationMetric.QUERY_PROCESSING_TIME] = ProcessingTimeEvaluator()

    def _generate_recommendations(self, results: List[EvaluationResult]) -> List[str]:
        """Generate recommendations based on evaluation results."""
        recommendations = []

        # Group results by metric
        metric_scores = {}
        for result in results:
            if result.metric not in metric_scores:
                metric_scores[result.metric] = []
            metric_scores[result.metric].append(result.score / result.max_score)

        # Analyze each metric
        for metric, scores in metric_scores.items():
            avg_score = sum(scores) / len(scores)

            if avg_score < 0.6:  # Poor performance
                if metric == EvaluationMetric.ENTITY_EXTRACTION_ACCURACY:
                    recommendations.append("Entity extraction accuracy is low - consider improving extraction models or rules")
                elif metric == EvaluationMetric.RESPONSE_RELEVANCE:
                    recommendations.append("Response relevance is low - review retrieval and ranking algorithms")
                elif metric == EvaluationMetric.GRAPH_CONNECTIVITY:
                    recommendations.append("Graph connectivity is poor - improve relationship extraction")
                elif metric == EvaluationMetric.QUERY_PROCESSING_TIME:
                    recommendations.append("Query processing is slow - optimize retrieval and caching")

            elif avg_score < 0.8:  # Room for improvement
                recommendations.append(f"{metric.value} has room for improvement (current: {avg_score:.2f})")

        if not recommendations:
            recommendations.append("System performance looks good across all metrics")

        return recommendations

    def compare_with_baseline(self, current_report: EvaluationReport) -> Dict[str, float]:
        """Compare current results with baseline."""
        if not self.baseline_scores:
            return {}

        comparisons = {}

        # Group current results by metric
        current_scores = {}
        for result in current_report.results:
            if result.metric not in current_scores:
                current_scores[result.metric] = []
            current_scores[result.metric].append(result.score / result.max_score)

        # Calculate average scores and compare
        for metric, scores in current_scores.items():
            current_avg = sum(scores) / len(scores)
            baseline_avg = self.baseline_scores.get(metric, current_avg)

            improvement = current_avg - baseline_avg
            comparisons[metric.value] = improvement

        return comparisons

    def set_baseline(self, report: EvaluationReport):
        """Set baseline scores from a report."""
        metric_scores = {}

        for result in report.results:
            if result.metric not in metric_scores:
                metric_scores[result.metric] = []
            metric_scores[result.metric].append(result.score / result.max_score)

        # Calculate average scores
        for metric, scores in metric_scores.items():
            self.baseline_scores[metric] = sum(scores) / len(scores)
```

### 2. Specific Evaluators

Create `packages/morag-core/src/morag_core/evaluation/evaluators.py`:

```python
from typing import Dict, List, Any
import re
from .evaluation_framework import BaseEvaluator, EvaluationResult, EvaluationMetric, TestCase
from datetime import datetime

class EntityExtractionEvaluator(BaseEvaluator):
    async def evaluate(self, test_case: TestCase, system_output: Any) -> EvaluationResult:
        """Evaluate entity extraction accuracy."""

        expected_entities = set(test_case.expected_output.get('entities', []))
        extracted_entities = set()

        if 'extracted_entities' in system_output:
            extracted_entities = set(entity['name'] for entity in system_output['extracted_entities'])

        # Calculate precision, recall, F1
        true_positives = len(expected_entities & extracted_entities)
        false_positives = len(extracted_entities - expected_entities)
        false_negatives = len(expected_entities - extracted_entities)

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return EvaluationResult(
            metric=EvaluationMetric.ENTITY_EXTRACTION_ACCURACY,
            score=f1_score,
            max_score=1.0,
            details={
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'true_positives': true_positives,
                'false_positives': false_positives,
                'false_negatives': false_negatives,
                'expected_entities': list(expected_entities),
                'extracted_entities': list(extracted_entities)
            },
            timestamp=datetime.now()
        )

class ResponseRelevanceEvaluator(BaseEvaluator):
    def __init__(self):
        # This would typically use an LLM for evaluation
        # For now, implement simple keyword-based relevance
        pass

    async def evaluate(self, test_case: TestCase, system_output: Any) -> EvaluationResult:
        """Evaluate response relevance using keyword matching."""

        expected_keywords = test_case.expected_output.get('keywords', [])
        response_text = ""

        if 'query_result' in system_output:
            response_text = system_output['query_result'].get('answer', '')

        # Simple keyword-based relevance scoring
        matched_keywords = 0
        for keyword in expected_keywords:
            if keyword.lower() in response_text.lower():
                matched_keywords += 1

        relevance_score = matched_keywords / len(expected_keywords) if expected_keywords else 0

        return EvaluationResult(
            metric=EvaluationMetric.RESPONSE_RELEVANCE,
            score=relevance_score,
            max_score=1.0,
            details={
                'matched_keywords': matched_keywords,
                'total_keywords': len(expected_keywords),
                'response_length': len(response_text),
                'expected_keywords': expected_keywords
            },
            timestamp=datetime.now()
        )

class ResponseCompletenessEvaluator(BaseEvaluator):
    async def evaluate(self, test_case: TestCase, system_output: Any) -> EvaluationResult:
        """Evaluate response completeness."""

        expected_aspects = test_case.expected_output.get('aspects', [])
        response_text = ""

        if 'query_result' in system_output:
            response_text = system_output['query_result'].get('answer', '')

        # Check if response addresses expected aspects
        covered_aspects = 0
        for aspect in expected_aspects:
            # Simple keyword-based check
            if any(keyword in response_text.lower() for keyword in aspect.lower().split()):
                covered_aspects += 1

        completeness_score = covered_aspects / len(expected_aspects) if expected_aspects else 1.0

        return EvaluationResult(
            metric=EvaluationMetric.RESPONSE_COMPLETENESS,
            score=completeness_score,
            max_score=1.0,
            details={
                'covered_aspects': covered_aspects,
                'total_aspects': len(expected_aspects),
                'response_length': len(response_text),
                'expected_aspects': expected_aspects
            },
            timestamp=datetime.now()
        )

class GraphConnectivityEvaluator(BaseEvaluator):
    async def evaluate(self, test_case: TestCase, system_output: Any) -> EvaluationResult:
        """Evaluate graph connectivity metrics."""

        # This would typically query the graph database
        # For now, return a placeholder score

        connectivity_score = 0.7  # Placeholder

        return EvaluationResult(
            metric=EvaluationMetric.GRAPH_CONNECTIVITY,
            score=connectivity_score,
            max_score=1.0,
            details={
                'connectivity_ratio': connectivity_score,
                'isolated_entities': 0,  # Placeholder
                'largest_component_size': 100  # Placeholder
            },
            timestamp=datetime.now()
        )

class ProcessingTimeEvaluator(BaseEvaluator):
    async def evaluate(self, test_case: TestCase, system_output: Any) -> EvaluationResult:
        """Evaluate query processing time."""

        processing_time = system_output.get('processing_time', 0)
        expected_max_time = test_case.expected_output.get('max_processing_time', 5.0)  # 5 seconds default

        # Score based on how much under the expected time
        if processing_time <= expected_max_time:
            score = 1.0
        else:
            # Penalty for exceeding expected time
            score = max(0, 1.0 - (processing_time - expected_max_time) / expected_max_time)

        return EvaluationResult(
            metric=EvaluationMetric.QUERY_PROCESSING_TIME,
            score=score,
            max_score=1.0,
            details={
                'processing_time': processing_time,
                'expected_max_time': expected_max_time,
                'within_limit': processing_time <= expected_max_time
            },
            timestamp=datetime.now()
        )
```

### 3. Test Case Management

Create `packages/morag-core/src/morag_core/evaluation/test_cases.py`:

```python
from typing import List
from .evaluation_framework import TestCase

def create_entity_extraction_test_cases() -> List[TestCase]:
    """Create test cases for entity extraction evaluation.

    Note: These are example test cases in English. In a multi-language system,
    test cases should be created for each supported language.
    """

    return [
        TestCase(
            id="entity_001",
            name="Basic Person and Organization Extraction",
            description="Test extraction of person and organization entities",
            input_data={
                'text': "Elon Musk is the CEO of Tesla and SpaceX. He founded both companies.",
                'language': 'en'  # Specify language for test case
            },
            expected_output={
                'entities': ['Elon Musk', 'Tesla', 'SpaceX']
            },
            metadata={
                'language': 'en',
                'entity_types': ['person', 'organization']
            }
        ),
        TestCase(
            id="entity_002",
            name="Technical Terms Extraction",
            description="Test extraction of technical terms and concepts",
            input_data={
                'text': "Machine learning algorithms use neural networks for pattern recognition.",
                'language': 'en'
            },
            expected_output={
                'entities': ['machine learning', 'neural networks', 'pattern recognition']
            },
            metadata={
                'language': 'en',
                'entity_types': ['concept', 'technology']
            }
        ),
        TestCase(
            id="entity_003",
            name="Location and Date Extraction",
            description="Test extraction of locations and dates",
            input_data={
                'text': "The conference was held in San Francisco on January 15, 2024."
            },
            expected_output={
                'entities': ['San Francisco', 'January 15, 2024']
            }
        )
    ]

def create_query_response_test_cases() -> List[TestCase]:
    """Create test cases for query response evaluation.

    Note: These are example test cases in English. In a multi-language system,
    test cases should be created for each supported language.
    """

    return [
        TestCase(
            id="query_001",
            name="Factual Query Test",
            description="Test response to factual question",
            input_data={
                'query': "Who is the CEO of Tesla?",
                'context': {'collection_name': 'test_collection'},
                'language': 'en'
            },
            expected_output={
                'keywords': ['Elon Musk', 'CEO', 'Tesla'],
                'aspects': ['person identification', 'role specification'],
                'max_processing_time': 3.0
            },
            metadata={
                'language': 'en',
                'query_type': 'factual'
            }
        ),
        TestCase(
            id="query_002",
            name="Analytical Query Test",
            description="Test response to analytical question",
            input_data={
                'query': "How does machine learning work?",
                'context': {'collection_name': 'test_collection'}
            },
            expected_output={
                'keywords': ['machine learning', 'algorithms', 'data', 'training'],
                'aspects': ['process explanation', 'technical details'],
                'max_processing_time': 5.0
            }
        ),
        TestCase(
            id="query_003",
            name="Summary Query Test",
            description="Test response to summary question",
            input_data={
                'query': "Summarize the key points about artificial intelligence",
                'context': {'collection_name': 'test_collection'}
            },
            expected_output={
                'keywords': ['artificial intelligence', 'key points', 'summary'],
                'aspects': ['overview', 'main concepts', 'applications'],
                'max_processing_time': 4.0
            }
        )
    ]
```

### 4. Integration with CLI

Create `cli/run-evaluation.py`:

```python
import asyncio
import argparse
import json
from morag_core.evaluation.evaluation_framework import EvaluationFramework, EvaluationMetric
from morag_core.evaluation.test_cases import create_entity_extraction_test_cases, create_query_response_test_cases

async def main():
    parser = argparse.ArgumentParser(description='Run MoRAG evaluation')
    parser.add_argument('--test-suite', choices=['entity_extraction', 'query_response', 'all'],
                       default='all', help='Test suite to run')
    parser.add_argument('--output', help='Output file for results')
    parser.add_argument('--baseline', action='store_true', help='Set results as baseline')

    args = parser.parse_args()

    # Initialize evaluation framework
    framework = EvaluationFramework()

    # Register test suites
    framework.register_test_suite('entity_extraction', create_entity_extraction_test_cases())
    framework.register_test_suite('query_response', create_query_response_test_cases())

    # Initialize system components (would be actual components in real implementation)
    system_components = {
        'entity_extractor': None,  # Would be actual EntityExtractor
        'graph_agent': None,       # Would be actual GraphTraversalAgent
        'document_processor': None # Would be actual DocumentProcessor
    }

    # Run evaluations
    test_suites = ['entity_extraction', 'query_response'] if args.test_suite == 'all' else [args.test_suite]

    for suite_name in test_suites:
        print(f"Running evaluation for {suite_name}...")

        report = await framework.run_evaluation(
            suite_name,
            system_components,
            metrics=[
                EvaluationMetric.ENTITY_EXTRACTION_ACCURACY,
                EvaluationMetric.RESPONSE_RELEVANCE,
                EvaluationMetric.QUERY_PROCESSING_TIME
            ]
        )

        print(f"Results for {suite_name}:")
        print(f"  Overall Score: {report.overall_score:.3f}")
        print(f"  Passed Tests: {report.passed_tests}/{report.total_tests}")
        print(f"  Recommendations: {', '.join(report.recommendations)}")

        if args.baseline:
            framework.set_baseline(report)
            print(f"  Set as baseline for future comparisons")

        # Save results if output file specified
        if args.output:
            with open(args.output, 'w') as f:
                json.dump({
                    'test_suite': suite_name,
                    'timestamp': report.timestamp.isoformat(),
                    'overall_score': report.overall_score,
                    'passed_tests': report.passed_tests,
                    'total_tests': report.total_tests,
                    'recommendations': report.recommendations
                }, f, indent=2)

if __name__ == "__main__":
    asyncio.run(main())
```

## Configuration

```yaml
# evaluation.yml
evaluation:
  enabled: true

  test_suites:
    entity_extraction:
      enabled: true
      test_cases_file: "test_cases/entity_extraction.json"

    query_response:
      enabled: true
      test_cases_file: "test_cases/query_response.json"

    graph_quality:
      enabled: true

  thresholds:
    passing_score: 0.7
    warning_score: 0.5

  performance:
    max_query_time: 5.0
    max_extraction_time: 2.0

  reporting:
    save_detailed_results: true
    results_directory: "evaluation_results"
    compare_with_baseline: true
```

## Testing Strategy

```python
# tests/unit/test_evaluation_framework.py
import pytest
from morag_core.evaluation.evaluation_framework import EvaluationFramework, TestCase

class TestEvaluationFramework:
    def setup_method(self):
        self.framework = EvaluationFramework()

    @pytest.mark.asyncio
    async def test_entity_extraction_evaluation(self):
        # Test entity extraction evaluation
        pass

    @pytest.mark.asyncio
    async def test_response_relevance_evaluation(self):
        # Test response relevance evaluation
        pass
```

## Success Metrics

- **Evaluation Coverage**: >90% of core functionality covered by tests
- **Regression Detection**: Catch >95% of performance regressions
- **Baseline Tracking**: Maintain historical performance baselines
- **Automated Testing**: Daily automated evaluation runs

## Future Enhancements

1. **LLM-based Evaluation**: Use LLMs for more sophisticated response quality assessment
2. **User Feedback Integration**: Incorporate user ratings into evaluation
3. **A/B Testing Framework**: Compare different system configurations
4. **Continuous Monitoring**: Real-time performance tracking in production
