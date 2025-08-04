### Knowledge Base Pipeline: Neo4j + Qdrant

#### **Overview**
This pipeline ingests documents in various formats, processes and analyzes their content, extracts structured facts, and builds a hybrid knowledge base using Neo4j (graph) and Qdrant (vector search). The process is modular, allowing for easy extension and maintenance.

---

#### **Pipeline Stages**

##### 1. Input Formatter
- **Tool:** `markitdown`
- **Purpose:** Convert incoming documents (PDF, HTML, DOCX, etc.) to clean Markdown for consistent downstream processing.
- **Implementation:**
  - Ingest files via upload or API.
  - Use `markitdown` to convert each file to Markdown.
  - Optionally extract and store metadata (title, author, date).

##### 2. Analyzer Agent
- **Tools:** `google langextract`, custom NLP scripts
- **Purpose:** Analyze the Markdown text to extract:
  - **Summary:** Short description of the document.
  - **Domain:** Classify the subject area.
  - **Keywords:** Identify key terms for indexing and linking.
  - **Language:** Detect document language using `google langextract`.
- **Implementation:**
  - Run language detection on the Markdown text.
  - Use NLP models or APIs for summarization, keyword extraction, and domain classification.

##### 3. Chunker
- **Purpose:** Split the Markdown text into manageable, semantically meaningful chunks (e.g., paragraphs, sections).
- **Implementation:**
  - Use rule-based or model-based chunking.
  - Optionally create overlapping chunks for better context retention.

##### 4. Fact Extractor
- **Purpose:** Extract structured facts from each chunk, such as:
  - Subject
  - Object
  - Approach
  - Solution
  - Remarks
- **Implementation:**
  - Use LLMs or custom extractors to identify and structure facts.
  - Track provenance (source chunk/document for each fact).

##### 5. Graph Builder
- **Tools:** Neo4j, Qdrant
- **Purpose:** Build a hybrid knowledge base:
  - **Neo4j:** Store facts and their relationships as a graph.
    - Example relationships: `Keyword → Fact ← DocumentChunk ← Document`, `Subject → Fact`, `Object → Fact`
  - **Qdrant:** Store vector embeddings of chunks/facts for semantic search.
- **Implementation:**
  - For each fact/chunk, generate embeddings (using a model like Sentence Transformers).
  - Store embeddings in Qdrant, with references to graph nodes in Neo4j.
  - Create and link nodes/edges in Neo4j according to the extracted structure.

---

#### **Example Workflow**

1. **Ingest**: User uploads a PDF.
2. **Format**: `markitdown` converts PDF to Markdown.
3. **Analyze**: `google langextract` detects language; NLP scripts extract summary, domain, keywords.
4. **Chunk**: Markdown is split into sections/paragraphs.
5. **Extract Facts**: Each chunk is analyzed for structured facts.
6. **Build KB**: Facts and relationships are stored in Neo4j; embeddings are stored in Qdrant for semantic search.

---

#### **Extensibility & Enhancements**
- Add feedback mechanisms for users to correct or enrich facts.
- Integrate additional NLP tools for entity recognition, sentiment, or topic modeling.
- Support more document formats or languages as needed.

---

#### **References**
- [markitdown documentation](https://github.com/markitdown/markitdown)
- [google langextract documentation](https://pypi.org/project/langextract/)
- [Neo4j](https://neo4j.com/)
- [Qdrant](https://qdrant.tech/)