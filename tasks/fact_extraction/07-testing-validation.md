# Task 7: Testing and Validation

## Objective
Create comprehensive tests for the fact extraction pipeline and validate that it produces more focused, actionable knowledge graphs compared to the current approach.

## Testing Strategy

### 7.1 Unit Testing Framework

**File**: `tests/fact_extraction/test_fact_extractor.py`

```python
class TestFactExtractor:
    """Unit tests for fact extraction functionality."""
    
    @pytest.fixture
    def fact_extractor(self):
        """Create fact extractor instance for testing."""
        return FactExtractor(
            model_id="gemini-2.0-flash",
            min_confidence=0.7,
            domain="test"
        )
    
    async def test_extract_facts_from_simple_text(self, fact_extractor):
        """Test fact extraction from simple, clear text."""
        
    async def test_extract_facts_from_complex_text(self, fact_extractor):
        """Test fact extraction from complex, multi-topic text."""
        
    async def test_fact_validation(self, fact_extractor):
        """Test fact validation and quality checking."""
        
    async def test_fact_deduplication(self, fact_extractor):
        """Test that duplicate facts are properly handled."""
        
    async def test_confidence_scoring(self, fact_extractor):
        """Test confidence score calculation and filtering."""
```

### 7.2 Integration Testing

**File**: `tests/fact_extraction/test_fact_integration.py`

```python
class TestFactIntegration:
    """Integration tests for fact extraction pipeline."""
    
    async def test_end_to_end_processing(self):
        """Test complete pipeline from document to stored facts."""
        
    async def test_fact_graph_building(self):
        """Test fact graph construction and relationship detection."""
        
    async def test_fact_storage_retrieval(self):
        """Test storing facts and retrieving them with queries."""
        
    async def test_hybrid_processing(self):
        """Test processing with both entity and fact extraction."""
        
    async def test_migration_process(self):
        """Test migration from entity to fact representation."""
```

### 7.3 Quality Validation Framework

**File**: `tests/fact_extraction/test_fact_quality.py`

```python
class FactQualityValidator:
    """Validate quality of extracted facts."""
    
    def __init__(self):
        self.quality_metrics = [
            SpecificityMetric(),
            ActionabilityMetric(),
            CompletenessMetric(),
            VerifiabilityMetric(),
            RelevanceMetric()
        ]
    
    def validate_fact_quality(self, fact: Fact, source_text: str) -> QualityScore:
        """Validate quality of a single fact."""
        
    def validate_fact_set_quality(self, facts: List[Fact], source_text: str) -> QualityReport:
        """Validate quality of a set of facts from the same source."""
        
    def compare_extraction_quality(self, entity_result: EntityResult, fact_result: FactResult) -> ComparisonReport:
        """Compare quality between entity and fact extraction."""

class QualityMetric(ABC):
    """Abstract base for quality metrics."""
    
    @abstractmethod
    def calculate_score(self, fact: Fact, source_text: str) -> float:
        """Calculate quality score for a fact."""
        
    @abstractmethod
    def get_feedback(self, fact: Fact, source_text: str) -> str:
        """Get human-readable feedback on quality."""
```

### 7.4 Performance Testing

**File**: `tests/fact_extraction/test_fact_performance.py`

```python
class TestFactPerformance:
    """Performance tests for fact extraction."""
    
    async def test_extraction_speed(self):
        """Test fact extraction speed vs entity extraction."""
        
    async def test_graph_size_comparison(self):
        """Compare graph size between entity and fact approaches."""
        
    async def test_query_performance(self):
        """Test query performance on fact-based graphs."""
        
    async def test_memory_usage(self):
        """Test memory usage during fact extraction."""
        
    async def test_scalability(self):
        """Test performance with large document sets."""
```

## Validation Datasets

### 7.5 Test Data Creation

**File**: `tests/data/fact_extraction_datasets.py`

```python
class FactExtractionDatasets:
    """Curated datasets for testing fact extraction."""
    
    @staticmethod
    def get_research_papers() -> List[TestDocument]:
        """Get research papers with known facts for validation."""
        
    @staticmethod
    def get_technical_documentation() -> List[TestDocument]:
        """Get technical docs with procedural facts."""
        
    @staticmethod
    def get_news_articles() -> List[TestDocument]:
        """Get news articles with factual claims."""
        
    @staticmethod
    def get_multilingual_content() -> List[TestDocument]:
        """Get content in multiple languages."""

class TestDocument(BaseModel):
    """Test document with expected facts."""
    
    content: str = Field(description="Document content")
    expected_facts: List[ExpectedFact] = Field(description="Facts that should be extracted")
    domain: str = Field(description="Document domain")
    language: str = Field(description="Document language")
    difficulty: str = Field(description="easy|medium|hard")

class ExpectedFact(BaseModel):
    """Expected fact for validation."""
    
    subject: str
    object: str
    approach: Optional[str] = None
    solution: Optional[str] = None
    remarks: Optional[str] = None
    fact_type: str
    min_confidence: float = 0.7
```

### 7.6 Evaluation Metrics

**File**: `tests/fact_extraction/evaluation_metrics.py`

```python
class FactExtractionMetrics:
    """Metrics for evaluating fact extraction quality."""
    
    def calculate_precision_recall(self, extracted_facts: List[Fact], expected_facts: List[ExpectedFact]) -> PrecisionRecall:
        """Calculate precision and recall for fact extraction."""
        
    def calculate_f1_score(self, precision: float, recall: float) -> float:
        """Calculate F1 score from precision and recall."""
        
    def calculate_semantic_similarity(self, extracted_fact: Fact, expected_fact: ExpectedFact) -> float:
        """Calculate semantic similarity between extracted and expected facts."""
        
    def evaluate_fact_completeness(self, fact: Fact) -> float:
        """Evaluate how complete a fact is (has required fields)."""
        
    def evaluate_fact_actionability(self, fact: Fact) -> float:
        """Evaluate how actionable a fact is."""

class GraphQualityMetrics:
    """Metrics for evaluating graph quality."""
    
    def calculate_graph_density(self, facts: List[Fact], relationships: List[FactRelation]) -> float:
        """Calculate graph density (relationships per fact)."""
        
    def calculate_clustering_coefficient(self, graph_data: GraphData) -> float:
        """Calculate clustering coefficient of fact graph."""
        
    def evaluate_relationship_quality(self, relationships: List[FactRelation]) -> float:
        """Evaluate quality of detected relationships."""
        
    def compare_graph_sizes(self, entity_graph: EntityGraph, fact_graph: FactGraph) -> SizeComparison:
        """Compare sizes of entity vs fact graphs."""
```

## Automated Testing Pipeline

### 7.7 Continuous Testing

**File**: `.github/workflows/fact_extraction_tests.yml`

```yaml
name: Fact Extraction Tests

on:
  push:
    paths:
      - 'packages/morag-graph/src/morag_graph/extraction/**'
      - 'tests/fact_extraction/**'
  pull_request:
    paths:
      - 'packages/morag-graph/src/morag_graph/extraction/**'

jobs:
  test-fact-extraction:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install -r requirements-test.txt
        
    - name: Run unit tests
      run: |
        pytest tests/fact_extraction/test_fact_extractor.py -v
        
    - name: Run integration tests
      run: |
        pytest tests/fact_extraction/test_fact_integration.py -v
        
    - name: Run quality validation
      run: |
        pytest tests/fact_extraction/test_fact_quality.py -v
        
    - name: Run performance tests
      run: |
        pytest tests/fact_extraction/test_fact_performance.py -v --benchmark-only
        
    - name: Generate test report
      run: |
        pytest --html=reports/fact_extraction_report.html --self-contained-html
```

### 7.8 Quality Benchmarks

**File**: `tests/fact_extraction/benchmarks.py`

```python
class FactExtractionBenchmarks:
    """Benchmark tests for fact extraction quality."""
    
    def __init__(self):
        self.baseline_scores = {
            "precision": 0.85,
            "recall": 0.80,
            "f1_score": 0.82,
            "graph_size_reduction": 0.60,
            "query_speed_improvement": 0.30
        }
    
    async def run_quality_benchmark(self) -> BenchmarkResult:
        """Run comprehensive quality benchmark."""
        
    async def run_performance_benchmark(self) -> PerformanceBenchmark:
        """Run performance comparison benchmark."""
        
    async def run_scalability_benchmark(self) -> ScalabilityBenchmark:
        """Run scalability benchmark with large datasets."""
        
    def validate_against_baseline(self, results: BenchmarkResult) -> ValidationResult:
        """Validate results against established baselines."""
```

## Manual Testing Procedures

### 7.9 Human Evaluation

**File**: `tests/fact_extraction/manual_evaluation.py`

```python
class ManualEvaluationFramework:
    """Framework for human evaluation of fact extraction."""
    
    def generate_evaluation_tasks(self, documents: List[Document]) -> List[EvaluationTask]:
        """Generate tasks for human evaluators."""
        
    def collect_human_feedback(self, task: EvaluationTask, feedback: HumanFeedback) -> None:
        """Collect and store human evaluation feedback."""
        
    def analyze_human_feedback(self) -> FeedbackAnalysis:
        """Analyze collected human feedback for insights."""
        
    def compare_human_vs_automated_scores(self) -> ComparisonAnalysis:
        """Compare human evaluation with automated metrics."""

class EvaluationTask(BaseModel):
    """Task for human evaluation."""
    
    document_id: str
    extracted_facts: List[Fact]
    questions: List[EvaluationQuestion]
    
class EvaluationQuestion(BaseModel):
    """Question for human evaluator."""
    
    question_type: str  # "relevance", "accuracy", "completeness", "actionability"
    question_text: str
    scale: str  # "1-5", "yes/no", "free_text"
```

## Validation Reports

### 7.10 Automated Reporting

**File**: `tests/fact_extraction/reporting.py`

```python
class FactExtractionReporter:
    """Generate comprehensive reports on fact extraction quality."""
    
    def generate_quality_report(self, test_results: TestResults) -> QualityReport:
        """Generate detailed quality assessment report."""
        
    def generate_performance_report(self, benchmark_results: BenchmarkResults) -> PerformanceReport:
        """Generate performance comparison report."""
        
    def generate_comparison_report(self, entity_results: EntityResults, fact_results: FactResults) -> ComparisonReport:
        """Generate comparison between entity and fact approaches."""
        
    def export_report(self, report: Report, format: str = "html") -> str:
        """Export report in specified format (html, pdf, json)."""
```

## Success Criteria

### Quality Metrics
- **Precision**: ≥85% of extracted facts are correct and relevant
- **Recall**: ≥80% of important facts are successfully extracted
- **F1 Score**: ≥82% overall extraction quality
- **Actionability**: ≥90% of facts provide actionable information

### Performance Metrics
- **Graph Size Reduction**: 50-70% fewer nodes while maintaining information quality
- **Query Speed**: 30%+ improvement in query response times
- **Processing Speed**: No more than 20% slower than entity extraction
- **Memory Usage**: Comparable or better memory efficiency

### Comparison Metrics
- **Retrieval Quality**: Improved relevance of retrieved information
- **User Satisfaction**: Better answers to domain-specific questions
- **Maintenance**: Easier to understand and maintain graph structure
- **Scalability**: Better performance with large document collections

## Implementation Tasks

### Task 7.1: Test Infrastructure
- [ ] Set up comprehensive testing framework
- [ ] Create test datasets with expected facts
- [ ] Implement automated quality metrics
- [ ] Set up continuous testing pipeline

### Task 7.2: Quality Validation
- [ ] Implement fact quality validation framework
- [ ] Create comparison tools for entity vs fact approaches
- [ ] Add human evaluation framework
- [ ] Implement automated quality scoring

### Task 7.3: Performance Testing
- [ ] Create performance benchmarks
- [ ] Implement scalability tests
- [ ] Add memory usage monitoring
- [ ] Create performance comparison reports

### Task 7.4: Validation and Reporting
- [ ] Generate comprehensive test reports
- [ ] Create quality dashboards
- [ ] Implement automated validation against baselines
- [ ] Add regression testing for quality metrics
