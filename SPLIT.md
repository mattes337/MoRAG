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

- [x] Create and implement `morag-video` package
   - [x] Basic video processing
   - [x] Video metadata extraction
   - [x] Frame extraction
   - [x] Video transcription

- [x] Create and implement `morag-document` package
   - [x] Basic document processing
   - [x] Text extraction
   - [x] Document metadata extraction
   - [x] Document conversion

- [x] Create and implement `morag-image` package
   - [x] Basic image processing
   - [x] Image captioning
   - [x] OCR (Optical Character Recognition)
   - [x] Image metadata extraction

- [ ] Create and implement `morag-web` package
   - [x] Web scraping
   - [x] Content extraction
   - [x] Content cleaning
   - [x] HTML to Markdown conversion
   - [ ] Move to separate package

- [ ] Create and implement `morag-youtube` package
   - [x] YouTube video downloading
   - [x] Metadata extraction
   - [x] Caption extraction
   - [ ] Move to separate package

### Phase 4: Services Package Implementation

- [ ] Create package structure for `morag-services`
- [x] Implement embedding services
- [ ] Implement vector storage services
- [ ] Implement AI services
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
- Web scraping
- Content extraction
- Content cleaning
- HTML to Markdown conversion
- Need to move to separate package

### Pending

- Complete remaining processor packages (morag-web, morag-youtube)
- Implement morag-services (vector storage, AI services)
- Implement integration layer (morag)
- Clean up obsolete code
- Update documentation

## Next Steps

1. Create and implement the `morag-web` package
2. Create and implement the `morag-youtube` package
3. Create and implement the `morag-services` package
4. Create and implement the integration layer (`morag`)
5. Clean up obsolete code and update documentation