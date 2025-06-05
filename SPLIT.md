# MoRAG Modularization Plan

## Table of Contents
1. [Overview](#overview)
2. [Package Structure](#package-structure)
3. [Implementation Steps](#implementation-steps)
4. [Dependency Management](#dependency-management)
5. [Cross-Package Communication](#cross-package-communication)
6. [Migration Strategy](#migration-strategy)

## Overview

This document outlines the plan to modularize the MoRAG project into separate packages to address dependency issues and improve maintainability. The modularization will be implemented in seven phases:

1. **Preparation**: Create backups, analyze dependencies, define interfaces
2. **Core Package Implementation**: Create and implement `morag-core`
3. **Processor Packages Implementation**: Create and implement processor-specific packages
4. **Services Package Implementation**: Create and implement `morag-services`
5. **Integration Layer Implementation**: Create and implement the integration layer (`morag`)
6. **Cleanup**: Remove obsolete code, update documentation
7. **Testing and Deployment**: Test the modularized system, deploy to production

## Package Structure

The modularized MoRAG project will consist of the following packages:

1. **morag-core**: Core interfaces, models, and utilities
2. **Processor Packages**:
   - morag-audio: Audio processing
   - morag-video: Video processing
   - morag-document: Document processing
   - morag-image: Image processing
   - morag-web: Web scraping and processing
   - morag-youtube: YouTube video processing
3. **morag-services**: Vector storage, AI services
4. **morag**: Integration layer, CLI, API

## Implementation Steps

### Phase 1: Preparation

- [x] Create a backup of the current codebase
- [x] Analyze dependencies between components
- [x] Define interfaces for cross-package communication
- [x] Set up development environment for modular development

### Phase 2: Core Package Implementation

- [x] Create package structure for `morag-core`
- [x] Implement base interfaces
- [x] Implement utility functions
- [x] Implement core exceptions
- [x] Test and publish `morag-core` package

### Phase 3: Processor Packages Implementation

- [x] Create and implement `morag-audio` package
   - [x] Basic audio processing
   - [x] Speech-to-text conversion
   - [x] Audio metadata extraction
   - [x] Speaker diarization services
   - [x] Topic segmentation services
   - [x] Package structure and dependencies

- [x] Create and implement `morag-video` package
   - [x] Basic video processing
   - [x] Video metadata extraction
   - [x] Frame extraction
   - [x] Video transcription
   - [x] Audio extraction integration
   - [x] Package structure and dependencies

- [x] Create and implement `morag-document` package
   - [x] Basic document processing
   - [x] Text extraction
   - [x] Document metadata extraction
   - [x] Document conversion
   - [x] Multiple format support (PDF, Office, etc.)
   - [x] Package structure and dependencies

- [x] Create and implement `morag-image` package
   - [x] Basic image processing
   - [x] Image captioning
   - [x] OCR (Optical Character Recognition)
   - [x] Image metadata extraction
   - [x] Comprehensive documentation and examples
   - [x] Package structure and dependencies

- [x] Create and implement `morag-embedding` package
   - [x] Text embedding generation
   - [x] Embedding storage and retrieval
   - [x] Similarity search
   - [x] Package structure and dependencies

- [ ] Complete `morag-web` package separation
   - [x] Web scraping functionality
   - [x] Content extraction
   - [x] Content cleaning
   - [x] HTML to Markdown conversion
   - [x] Package structure created
   - [ ] Move implementation from main codebase
   - [ ] Update dependencies and imports

- [ ] Complete `morag-youtube` package separation
   - [x] YouTube video downloading
   - [x] Metadata extraction
   - [x] Caption extraction
   - [x] Package structure created
   - [ ] Move implementation from main codebase
   - [ ] Update dependencies and imports

### Phase 4: Services Package Implementation

- [x] Create package structure for `morag-services`
- [x] Implement embedding services
- [x] Implement unified service layer
- [x] Implement pipeline framework
- [ ] Complete vector storage services integration
- [ ] Complete AI services integration
- [ ] Test and publish `morag-services` package

### Phase 5: Integration Layer Implementation

- [ ] Create package structure for `morag`
- [ ] Implement integration layer
- [ ] Implement CLI
- [ ] Implement API
- [ ] Test and publish `morag` package

### Phase 6: Cleanup

- [ ] Remove obsolete code from the original codebase
- [ ] Update documentation
- [ ] Update CI/CD pipelines

### Phase 7: Testing and Deployment

- [ ] Test the modularized system
- [ ] Deploy to production
- [ ] Monitor for issues

## Dependency Management

Each package will have its own `pyproject.toml` file with the following dependencies:

1. **morag-core**: No dependencies on other MoRAG packages
2. **Processor Packages**: Depend on `morag-core`
3. **morag-services**: Depends on `morag-core`
4. **morag**: Depends on all other packages

## Cross-Package Communication

Communication between packages will be done through the interfaces defined in `morag-core`. This ensures loose coupling between packages and allows for easier testing and maintenance.

## Migration Strategy

The migration will be done incrementally, with each package being implemented and tested separately. The original codebase will be kept functional until all packages are implemented and tested.

## Implementation Status

### Completed

#### morag-core
- Base interfaces for processors and services
- Core data models
- Utility functions
- Exception handling

#### morag-document
- Document processing
- Text extraction from various document formats
- Document metadata extraction
- Document conversion

#### morag-audio
- Audio processing
- Speech-to-text conversion
- Audio metadata extraction

#### morag-video
- Video processing
- Video metadata extraction
- Frame extraction
- Video transcription

#### morag-embedding
- Text embedding generation
- Embedding storage and retrieval
- Similarity search

#### morag-image
- Basic image processing
- Image captioning using Gemini
- OCR using Tesseract and EasyOCR
- Image metadata extraction
- Service interface for high-level operations
- CLI for command-line usage

### In Progress

#### morag-web
- Web scraping functionality implemented
- Content extraction implemented
- Content cleaning implemented
- HTML to Markdown conversion implemented
- Package structure created
- **Need to**: Move implementation from main codebase to package

#### morag-youtube
- YouTube video downloading implemented
- Metadata extraction implemented
- Caption extraction implemented
- Package structure created
- **Need to**: Move implementation from main codebase to package

#### morag-services
- Unified service layer implemented
- Pipeline framework implemented
- **Need to**: Complete vector storage and AI services integration

### Pending

- Complete morag-web package separation (move from src/morag to packages/morag-web)
- Complete morag-youtube package separation (move from src/morag to packages/morag-youtube)
- Complete morag-services package (vector storage, AI services)
- Implement integration layer (morag package)
- Create Docker containers for each package
- Clean up obsolete code from main codebase
- Update documentation and deployment guides

## Next Steps

1. **Complete morag-web package separation** - Move web processing implementation from main codebase
2. **Complete morag-youtube package separation** - Move YouTube processing implementation from main codebase
3. **Complete morag-services package** - Integrate vector storage and AI services
4. **Create integration layer (morag package)** - Unified API and CLI interface
5. **Create Docker containers** - Individual containers for each package component
6. **Clean up main codebase** - Remove obsolete code after successful migration
7. **Update deployment documentation** - Docker Compose orchestration for modular system

## Architecture Benefits

The modular architecture provides several key benefits:

- **Isolated Dependencies**: Each converter has its own package and container, eliminating dependency conflicts
- **Scalability**: Scale individual converters based on demand (e.g., more document workers, fewer video workers)
- **Resource Optimization**: Allocate resources appropriately per converter type
- **Simplified Maintenance**: Update converters independently without affecting the entire system
- **Resilience**: If one converter fails, others continue working
- **Microservices Ready**: Each package can be deployed as a separate microservice