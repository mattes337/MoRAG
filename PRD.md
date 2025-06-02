## **Product Requirements Document: Multimodal RAG Ingestion Pipeline**

Version: 1.0  
Date: June 2, 2025  
Author: Gemini AI  
Based on concept by: User

### **1\. Introduction**

This document outlines the requirements for a Multimodal Retrieval Augmented Generation (RAG) Ingestion Pipeline. The system is designed to process various data sources, transform them into a queryable format, and store them in a vector database to be used by downstream RAG applications. The pipeline will handle documents, voice, video, websites, and YouTube content, ensuring that information is accurately extracted, chunked, enriched with metadata and summaries, embedded, and indexed.

The core of the ingestion process will be handled by a Python-based microservice, designed for asynchronous processing and scalability. Orchestration will be managed by a workflow automation tool like n8n.

### **2\. Goals & Objectives**

* **Primary Goal:** To create a robust, scalable, and automated pipeline for ingesting diverse data types into a vector database (Qdrant) for use in RAG applications.  
* **Key Objectives:**  
  * Support ingestion from multiple sources: PDF, Word documents, Markdown files, voice recordings, videos, website URLs, and YouTube links.  
  * Standardize content into a processable format (primarily Markdown with associated metadata).  
  * Implement semantic chunking for optimal retrieval.  
  * Enrich chunks with generated summaries (CRAG-inspired approach) and relevant metadata.  
  * Convert speech to text and extract information from audio and video content.  
  * Process images within documents and videos, generating textual descriptions.  
  * Enable efficient embedding generation for all processed content.  
  * Provide asynchronous processing capabilities with progress tracking and webhook notifications.  
  * Ensure the pipeline is maintainable and extensible for future data sources or processing steps.

### **3\. Target Users**

* **Data Engineers/MLOps Engineers:** Responsible for deploying, maintaining, and scaling the ingestion pipeline.  
* **AI/ML Developers:** Who will build RAG applications that consume the data from the vector database.  
* **Knowledge Management Teams:** Who need to make diverse internal and external information sources searchable and accessible.

### **4\. Scope**

#### **4.1. In-Scope Features (Version 1.0)**

* **Content Source Ingestion:**  
  * File Uploads: PDF, DOCX (Word), Markdown.  
  * Voice: Common audio file formats (e.g., MP3, WAV, M4A).  
  * Video: Common video file formats (e.g., MP4, MOV, AVI) and YouTube URLs.  
  * Website: Publicly accessible website URLs.  
* **Processing Pipelines (as per diagram):**  
  * **Document Path:**  
    * Conversion to Markdown (including handling of images and tables).  
    * Semantic splitting/chunking.  
    * Prepending chunk summary.  
    * Adding metadata (source, file type, etc.).  
  * **Voice Path:**  
    * Speech-to-Text (STT).  
    * Topic identification (basic).  
    * Speaker diarization (basic).  
    * Prepending contextual summary.  
    * Adding metadata (source, timestamp, speaker if available).  
  * **Video Path:**  
    * Extraction of audio and keyframes/images.  
    * Audio processing via Voice Path.  
    * Image processing: Image to Markdown (captioning/OCR).  
    * Prepending contextual summary for image-derived text.  
    * Adding metadata (source, timestamp, image reference).  
  * **Website Path:**  
    * Web crawling and content extraction (e.g., using Firecrawl).  
    * Website/page summarization.  
    * Splitting content by page or semantic sections.  
    * Prepending contextual summary.  
    * Adding metadata (source URL, page title, crawl date).  
  * **YouTube Path:**  
    * Video download.  
    * Processing via Video Path.  
* **Embedding Generation:**  
  * Generate vector embeddings for all processed text chunks.  
* **Vector Storage:**  
  * Store text chunks and their embeddings in Qdrant.  
  * Store associated metadata with each vector.  
* **Python Processing Service:**  
  * Expose API endpoints to trigger ingestion for different sources.  
  * Support asynchronous processing.  
  * Provide a GET endpoint to check processing status/progress.  
  * Support webhook callback upon completion/failure.  
* **Orchestration:**  
  * Use n8n (or a similar workflow automation tool) to manage the overall flow, trigger the Python service, and handle basic routing based on initial input type.

#### **4.2. Out-of-Scope Features (Version 1.0)**

* Advanced real-time continuous ingestion (focus on batch/on-demand).  
* Complex topic modeling beyond basic identification.  
* Advanced speaker diarization for overlapping speech.  
* User interface for managing the pipeline (orchestration tool's UI will be used).  
* Detailed analytics and monitoring dashboards beyond basic logging.  
* Direct RAG query interface (this system focuses on ingestion only).  
* Support for proprietary or highly secured/DRM-protected content sources without proper connectors.  
* Automated quality control and re-processing loops beyond basic error handling.

### **5\. Functional Requirements**

#### **5.1. Data Ingestion Module (Python Service)**

* **FR1.1 Document Ingestion:**  
  * Accept PDF, DOCX, MD files.
  * Utilize unstructured.io as the primary library for parsing, converting to Markdown, and extracting elements (text, tables, image references).
  * Consider docling as an alternative for complex PDF layouts or performance optimization.
  * Handle images within documents by extracting them and queueing for image processing.
  * Handle tables by converting them to a Markdown representation or a structured textual description.
* **FR1.2 Voice Ingestion:**  
  * Accept audio files (MP3, WAV, M4A).  
  * Utilize OpenAI Whisper for Speech-to-Text (STT).  
  * Perform basic speaker diarization if multiple speakers are detected by Whisper.  
  * Segment transcript based on speaker turns or semantic breaks.  
* **FR1.3 Video Ingestion:**  
  * Accept video files (MP4, MOV, etc.) and YouTube URLs.  
  * Download YouTube videos.  
  * Extract audio track and process via Voice Ingestion path (FR1.2).  
  * Extract keyframes/images at regular intervals or based on scene changes.  
  * Queue extracted images for Image Processing (FR1.4).  
* **FR1.4 Image Processing (from Documents/Videos):**  
  * Accept image files (PNG, JPG).  
  * Generate descriptive text/captions for images (e.g., using a multimodal LLM or dedicated image captioning model).  
  * Alternatively, perform OCR if images contain significant text.  
  * Output: Markdown representation (e.g., \!\[Image Alt Text\](image\_ref)\\nCaption: Generated caption.).  
* **FR1.5 Website Ingestion:**  
  * Accept website URLs.  
  * Utilize Firecrawl (or similar library like BeautifulSoup \+ requests for simpler cases) to scrape main content.  
  * Convert scraped HTML to clean Markdown.  
* **FR1.6 Content Chunking:**  
  * Implement semantic chunking using spaCy for sentence/paragraph boundary detection and NLP-based grouping.  
  * Chunk size should be configurable but aim for meaningful semantic units.  
* **FR1.7 Summary Generation (CRAG-inspired):**
  * For each text chunk, generate a concise summary.
  * Prepend this summary to the chunk content before embedding (e.g., "Summary: \[Generated Summary\]. Original Content: \[Chunk Content\]").
  * This will be done using Google Gemini API for consistent, high-quality summarization.
* **FR1.8 Metadata Association:**  
  * Extract/generate and associate relevant metadata with each chunk:  
    * Original source (filename, URL).  
    * Content type (PDF, voice, video frame, webpage).  
    * Timestamps (creation, processing, relevant content timestamp for A/V).  
    * Speaker ID (for voice/audio).  
    * Page number/section ID (for documents/websites).  
    * Image reference/URL (for image-derived text).  
    * Generated topics.  
* **FR1.9 Embedding Generation:**
  * Utilize Google Gemini API with text-embedding-004 model to generate vector embeddings for each (summary \+ chunk) combination.
  * Fallback to sentence transformer models (e.g., from Hugging Face) should be configurable for offline scenarios or cost optimization.
* **FR1.10 Vector Storage (Qdrant):**  
  * Store the generated embeddings along with their corresponding text chunks and metadata in a Qdrant collection.  
  * Ensure Qdrant is configured with appropriate indexing for efficient similarity search.

#### **5.2. Asynchronous Processing & API (Python Service)**

* **FR2.1 Asynchronous Task Handling:**  
  * All ingestion tasks must be processed asynchronously to prevent blocking API calls.  
  * Use a task queue (e.g., Celery with Redis/RabbitMQ, or Python's asyncio with aiohttp for the service itself if simpler).  
* **FR2.2 Ingestion API Endpoint:**  
  * Provide a RESTful API endpoint (e.g., POST /ingest) that accepts:  
    * Source type (document, voice, video, website, youtube).  
    * Data (file upload, URL).  
    * Optional webhook URL for notifications.  
  * The API should return a task\_id immediately.  
* **FR2.3 Progress Tracking API Endpoint:**  
  * Provide a GET /ingest/status/{task\_id} endpoint to check the status of an ingestion task (e.g., pending, processing, completed, failed, progress percentage).  
* **FR2.4 Webhook Notification:**  
  * If a webhook URL is provided in the ingestion request, send a POST request to this URL upon task completion or failure, including task\_id, status, and any relevant output or error message.

#### **5.3. Orchestration (n8n or similar)**

* **FR3.1 Workflow Definition:**  
  * Define n8n workflows to handle initial requests, route them to the Python service, and manage basic error handling or retries at the orchestration level.  
  * The "Mime?" decision diamond from the diagram will be implemented here, determining which parameters to pass to the Python service.  
* **FR3.2 Python Service Integration:**  
  * n8n workflows will make HTTP requests to the Python processing service's API endpoints.  
* **FR3.3 Monitoring (Basic):**  
  * Leverage n8n's execution logs for basic monitoring of workflow success/failure.

### **6\. Non-Functional Requirements**

* **NFR1.1 Performance:**  
  * STT for a 10-minute audio file should complete within a reasonable timeframe (target TBD, e.g., \< 5 minutes).  
  * Document parsing and chunking should be efficient.  
  * Embedding generation will depend on the chosen model and batch size.  
  * The system should handle concurrent ingestion requests effectively.  
* **NFR1.2 Scalability:**  
  * The Python processing service should be designed to be scalable (e.g., containerized and deployable with multiple replicas).  
  * The task queue and vector database (Qdrant) should also support scaling.  
* **NFR1.3 Reliability:**  
  * The pipeline should be resilient to transient errors with retry mechanisms (at orchestration and/or service level).  
  * Robust error logging and reporting for troubleshooting.  
* **NFR1.4 Maintainability:**  
  * Code should be well-documented, modular, and follow best practices.  
  * Configuration (e.g., model names, chunk sizes, API keys) should be externalized.  
* **NFR1.5 Security:**  
  * API endpoints for the Python service should be secured (e.g., API key authentication).  
  * Handle sensitive data (if any) appropriately.  
  * Secure credentials for external services (OpenAI, Qdrant, etc.).  
* **NFR1.6 Extensibility:**  
  * The architecture should allow for adding new data source types or processing steps with relative ease.

### **7\. Technical Stack & Design Considerations**

* **7.1. Orchestration:**  
  * **Tool:** n8n (or similar like Apache Airflow if complexity grows significantly).  
  * **Reasoning:** n8n offers a visual workflow builder, suitable for managing the sequence of operations and integrating with the Python service via HTTP.  
* **7.2. Python Processing Service:**  
  * **Framework:** FastAPI or Flask (FastAPI preferred for async support and performance).  
  * **Task Queue (Optional but Recommended for heavy tasks):** Celery with Redis or RabbitMQ. For simpler async, FastAPI's background tasks or asyncio might suffice initially.  
  * **Document Parsing:**
    * **Primary Library:** unstructured.io
    * **Alternative Library:** docling
    * **Reasoning:** unstructured.io provides comprehensive support for various file formats (PDF, DOCX, MD, HTML, etc.), robust parsing and cleaning capabilities, and options for different parsing strategies. It's well-suited for converting diverse documents into a clean textual format, including handling elements like tables and images (by referencing them). Docling is considered as an alternative that may offer better performance for certain document types, particularly PDFs with complex layouts. The implementation should allow for easy switching between these libraries based on document type or performance requirements.
  * **NLP / Semantic Chunking:**  
    * **Library:** spaCy  
    * **Reasoning:** spaCy is efficient for NLP tasks like sentence segmentation, tokenization, and can be used to inform semantic chunking logic. Its pre-trained models are robust.  
  * **Speech-to-Text (STT):**  
    * **Library/Service:** OpenAI Whisper (self-hosted models or API).  
    * **Reasoning:** Whisper provides state-of-the-art accuracy for STT and supports speaker diarization to some extent.  
  * **Image Processing (Captioning/OCR):**  
    * Models like BLIP, GIT (via Hugging Face Transformers) for captioning.  
    * Tesseract (via pytesseract) or unstructured.io's image capabilities for OCR.  
    * Consider multimodal LLMs if available and suitable.  
  * **Web Scraping:**  
    * **Library:** Firecrawl.io (as suggested, if it meets requirements for scraping and Markdown conversion) or BeautifulSoup \+ requests \+ markdownify for a custom solution.  
  * **Embedding Models:**
    * **Primary:** Google Gemini API with text-embedding-004 model.
    * **Alternative:** Sentence Transformers (e.g., all-MiniLM-L6-v2, multi-qa-mpnet-base-dot-v1) via Hugging Face.
    * **Reasoning:** Gemini's text-embedding-004 provides state-of-the-art embedding quality with competitive pricing. Sentence Transformers offer a self-hosted alternative for cost optimization or offline scenarios.
  * **YouTube Download:**  
    * **Library:** yt-dlp (a maintained fork of youtube-dl).  
* **7.3. Vector Database:**  
  * **Tool:** Qdrant  
  * **Reasoning:** Qdrant is a performant vector database with filtering capabilities, suitable for RAG.  
* **7.4. API Design (Python Service):**  
  * RESTful principles.  
  * Clear request/response schemas (Pydantic models with FastAPI).  
  * Stateless design for scalability.

### **8\. Success Metrics**

* **SM1. Throughput:** Number of documents/items processed per hour/day.  
* **SM2. Error Rate:** Percentage of ingestion tasks failing.  
* **SM3. Processing Latency:** Average time taken for end-to-end processing of different content types.  
* **SM4. Coverage:** Percentage of supported data types successfully ingested.  
* **SM5. Retrieval Relevance (Indirect):** While not directly measured by the ingestion pipeline, the quality of ingested data will be evaluated by the performance of downstream RAG applications (e.g., precision/recall of answers).  
* **SM6. System Uptime:** Availability of the ingestion service.

### **9\. Future Considerations**

* Support for more data sources (e.g., Confluence, Jira, Slack, databases).  
* Advanced OCR and table extraction techniques.  
* More sophisticated topic modeling and knowledge graph creation.  
* Integration with active learning loops to improve chunking or summarization based on RAG feedback.  
* Multi-tenancy if the system needs to serve different RAG applications or user groups with isolated data.  
* Enhanced monitoring and alerting dashboard.  
* Support for incremental updates and deletion of content from the vector store.