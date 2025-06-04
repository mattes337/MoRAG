# MoRAG Project Modularization Implementation Guide

This document outlines the comprehensive plan for splitting the MoRAG project into separate packages to address the "resolution-too-deep" pip dependency issues and improve overall project maintainability.

## Table of Contents

- [Overview](#overview)
- [Package Structure](#package-structure)
- [Implementation Steps](#implementation-steps)
  - [Phase 1: Preparation](#phase-1-preparation)
  - [Phase 2: Core Package Implementation](#phase-2-core-package-implementation)
  - [Phase 3: Processor Packages Implementation](#phase-3-processor-packages-implementation)
  - [Phase 4: Services Package Implementation](#phase-4-services-package-implementation)
  - [Phase 5: Integration Layer Implementation](#phase-5-integration-layer-implementation)
  - [Phase 6: Testing and Validation](#phase-6-testing-and-validation)
  - [Phase 7: Documentation and Deployment](#phase-7-documentation-and-deployment)
- [Dependency Management](#dependency-management)
- [Cross-Package Communication](#cross-package-communication)
- [Migration Strategy](#migration-strategy)

## Overview

The MoRAG project currently faces "resolution-too-deep" issues with pip dependencies due to the complex dependency tree across different processors (audio, video, document, image, web). By splitting the project into separate packages, we can isolate dependencies and improve maintainability while preserving functionality.

## Package Structure

We will reorganize the project into the following packages:

1. **morag-core**: Essential components and interfaces
   - Base classes and interfaces
   - Common utilities
   - Configuration management
   - Core exceptions

2. **Processor-specific packages**:
   - **morag-audio**: Audio processing capabilities
   - **morag-video**: Video processing capabilities
   - **morag-document**: Document processing capabilities
   - **morag-image**: Image processing capabilities
   - **morag-web**: Web scraping capabilities
   - **morag-youtube**: YouTube integration

3. **morag-services**: Shared services
   - Chunking service
   - Embedding services
   - Vector storage services
   - AI services (Gemini, etc.)

4. **morag**: Integration layer
   - API endpoints
   - Task definitions
   - Service orchestration
   - Universal converter

## Implementation Steps

### Phase 1: Preparation

- [ ] **1.1 Create a backup of the current project**
  - [ ] Create a full backup of the existing codebase
  - [ ] Set up a version control branch for the modularization work

- [ ] **1.2 Analyze dependencies in detail**
  - [ ] Map all dependencies for each processor
  - [ ] Identify shared dependencies across processors
  - [ ] Document dependency conflicts and resolution issues

- [ ] **1.3 Define package interfaces**
  - [ ] Design interface contracts for each package
  - [ ] Define communication protocols between packages
  - [ ] Create interface documentation

- [ ] **1.4 Set up development environment**
  - [ ] Create directory structure for new packages
  - [ ] Set up virtual environments for testing
  - [ ] Configure development tools for multi-package development

### Phase 2: Core Package Implementation

- [ ] **2.1 Create morag-core package structure**
  - [ ] Set up package directory structure
  - [ ] Create pyproject.toml with minimal dependencies
  - [ ] Implement package initialization

- [ ] **2.2 Implement base interfaces**
  - [ ] Define processor base classes
  - [ ] Create service interface definitions
  - [ ] Implement common data models
  - [ ] Define result type interfaces

- [ ] **2.3 Implement utility functions**
  - [ ] Move common utilities to core package
  - [ ] Implement configuration management
  - [ ] Create logging infrastructure

- [ ] **2.4 Implement core exceptions**
  - [ ] Define exception hierarchy
  - [ ] Implement error handling utilities
  - [ ] Create exception documentation

- [ ] **2.5 Test and publish core package**
  - [ ] Write unit tests for core functionality
  - [ ] Set up CI/CD for the core package
  - [ ] Publish initial version to package repository

### Phase 3: Processor Packages Implementation

- [ ] **3.1 Implement morag-audio package**
  - [ ] Create package structure with pyproject.toml
  - [ ] Move audio processor code
  - [ ] Define dependencies (Whisper, etc.)
  - [ ] Implement interface adapters to core
  - [ ] Write unit tests

- [x] **3.2 Implement morag-video package**
  - [x] Create package structure with pyproject.toml
  - [x] Move video processor code
  - [x] Define dependencies (FFmpeg, etc.)
  - [x] Implement interface adapters to core
  - [x] Write unit tests

- [x] **3.3 Implement morag-document package**
  - [x] Create package structure with pyproject.toml
  - [x] Move document processor code
  - [x] Define dependencies (pypdf, python-docx, etc.)
  - [x] Implement interface adapters to core
  - [x] Write unit tests

- [ ] **3.4 Implement morag-image package**
  - [ ] Create package structure with pyproject.toml
  - [ ] Move image processor code
  - [ ] Define dependencies (vision libraries, etc.)
  - [ ] Implement interface adapters to core
  - [ ] Write unit tests

- [ ] **3.5 Implement morag-web package**
  - [ ] Create package structure with pyproject.toml
  - [ ] Move web scraping code
  - [ ] Define dependencies (BeautifulSoup, etc.)
  - [ ] Implement interface adapters to core
  - [ ] Write unit tests

- [ ] **3.6 Implement morag-youtube package**
  - [ ] Create package structure with pyproject.toml
  - [ ] Move YouTube integration code
  - [ ] Define dependencies (yt-dlp, etc.)
  - [ ] Implement interface adapters to core
  - [ ] Write unit tests

### Phase 4: Services Package Implementation

- [ ] **4.1 Create morag-services package structure**
  - [ ] Set up package directory structure
  - [ ] Create pyproject.toml with service dependencies
  - [ ] Implement package initialization

- [ ] **4.2 Implement chunking service**
  - [ ] Move chunking service code
  - [ ] Adapt to use core interfaces
  - [ ] Write unit tests

- [ ] **4.3 Implement embedding services**
  - [ ] Move embedding service code
  - [ ] Adapt to use core interfaces
  - [ ] Write unit tests

- [ ] **4.4 Implement vector storage services**
  - [ ] Move Qdrant service code
  - [ ] Adapt to use core interfaces
  - [ ] Write unit tests

- [ ] **4.5 Implement AI services**
  - [ ] Move Gemini service code
  - [ ] Implement provider abstraction
  - [ ] Adapt to use core interfaces
  - [ ] Write unit tests

- [ ] **4.6 Test and publish services package**
  - [ ] Write integration tests
  - [ ] Set up CI/CD for the services package
  - [ ] Publish initial version to package repository

### Phase 5: Integration Layer Implementation

- [ ] **5.1 Create morag integration package**
  - [ ] Set up package directory structure
  - [ ] Create pyproject.toml with dependencies on all subpackages
  - [ ] Implement package initialization

- [ ] **5.2 Implement service discovery**
  - [ ] Create service registry
  - [ ] Implement dynamic loading of processors
  - [ ] Add configuration for service selection

- [ ] **5.3 Implement API endpoints**
  - [ ] Move API code to integration package
  - [ ] Adapt to use modular services
  - [ ] Implement health checks for all services

- [ ] **5.4 Implement task definitions**
  - [ ] Move Celery task definitions
  - [ ] Adapt to use modular processors and services
  - [ ] Implement task routing based on available processors

- [ ] **5.5 Implement universal converter**
  - [ ] Adapt universal converter to use modular processors
  - [ ] Implement lazy loading of converters
  - [ ] Add fallback mechanisms

- [ ] **5.6 Test integration layer**
  - [ ] Write integration tests
  - [ ] Test with various processor combinations
  - [ ] Verify backward compatibility

### Phase 6: Testing and Validation

- [ ] **6.1 Create test suite for full system**
  - [ ] Implement end-to-end tests
  - [ ] Create performance benchmarks
  - [ ] Develop compatibility tests

- [ ] **6.2 Test with different dependency combinations**
  - [ ] Test with minimal installations
  - [ ] Test with full installations
  - [ ] Test with partial processor availability

- [ ] **6.3 Validate Docker configurations**
  - [ ] Update Dockerfiles for modular structure
  - [ ] Test Docker builds with different configurations
  - [ ] Update docker-compose files

- [ ] **6.4 Perform regression testing**
  - [ ] Run existing test suite against new structure
  - [ ] Compare performance metrics
  - [ ] Verify all functionality works as expected

### Phase 7: Documentation and Deployment

- [ ] **7.1 Update documentation**
  - [ ] Create installation guides for different configurations
  - [ ] Update API documentation
  - [ ] Document package interfaces

- [ ] **7.2 Create migration guides**
  - [ ] Document upgrade path for existing installations
  - [ ] Provide scripts for migration
  - [ ] Create troubleshooting guide

- [ ] **7.3 Update deployment configurations**
  - [ ] Update Kubernetes configurations
  - [ ] Update CI/CD pipelines
  - [ ] Create deployment templates

- [ ] **7.4 Release and publish**
  - [ ] Finalize version numbers
  - [ ] Publish all packages
  - [ ] Create release notes

## Dependency Management

### Strategies for Resolving Dependencies

1. **Minimal Core Dependencies**
   - Keep core package dependencies to an absolute minimum
   - Use abstract interfaces where possible
   - Avoid direct dependencies on processor-specific libraries

2. **Optional Dependencies**
   - Use optional dependency groups in each package
   - Allow users to install only what they need
   - Provide clear documentation on dependency requirements

3. **Version Constraints**
   - Use loose version constraints where possible
   - Pin only critical dependencies
   - Document compatibility matrix

4. **Dependency Isolation**
   - Ensure processor packages don't depend on each other
   - Use core interfaces for communication
   - Implement adapter pattern for cross-processor functionality

## Cross-Package Communication

### Communication Patterns

1. **Interface-based Communication**
   - Define clear interfaces in core package
   - Implement interfaces in processor packages
   - Use dependency injection for service composition

2. **Service Discovery**
   - Implement a service registry
   - Allow dynamic discovery of available processors
   - Support fallback mechanisms

3. **Event-based Communication**
   - Use events for loose coupling
   - Implement pub/sub patterns for notifications
   - Support asynchronous processing

4. **Lazy Loading**
   - Load processor implementations only when needed
   - Handle missing processors gracefully
   - Provide clear error messages for missing dependencies

## Migration Strategy

### Gradual Migration Approach

1. **Parallel Development**
   - Maintain existing codebase during migration
   - Develop modular packages in parallel
   - Gradually replace components

2. **Compatibility Layer**
   - Implement adapters for backward compatibility
   - Support both old and new interfaces during transition
   - Deprecate old interfaces gradually

3. **Testing Strategy**
   - Test each package independently
   - Test integration points thoroughly
   - Maintain comprehensive test coverage

4. **Deployment Strategy**
   - Support side-by-side installation
   - Provide clear upgrade paths
   - Document breaking changes

### Rollback Plan

1. **Version Control**
   - Maintain clear version history
   - Tag stable versions
   - Document changes between versions

2. **Backup Strategy**
   - Create backups before major changes
   - Test restoration procedures
   - Document recovery process

3. **Phased Rollout**
   - Deploy to test environments first
   - Gradually roll out to production
   - Monitor for issues and be prepared to rollback

## Implementation Progress

### Completed Packages

- **morag-core**: Base interfaces and models implemented
  - Base processor, converter, and service interfaces
  - Document models and chunking strategies
  - Common utilities and exceptions

- **morag-document**: Complete implementation
  - Document processor with format detection and validation
  - Multiple format converters:
    - PDF converter using pypdf
    - Word converter using python-docx
    - Excel converter using openpyxl
    - PowerPoint converter using python-pptx
    - Text converter with support for plain text, markdown, and HTML
  - Document service with embedding and summarization capabilities
  - CLI for document processing
  - Comprehensive test suite
  - Example scripts for demonstration

### Further Tasks

- [ ] **Complete remaining processor packages**
  - [ ] Implement morag-audio package
  - [x] Implement morag-video package
  - [ ] Implement morag-image package
  - [ ] Implement morag-web package
  - [ ] Implement morag-youtube package

- [ ] **Implement morag-services package**
  - [ ] Move embedding services
  - [ ] Move vector storage services
  - [ ] Move AI services

- [ ] **Implement integration layer**
  - [ ] Create unified API
  - [ ] Implement service discovery
  - [ ] Create universal converter

- [ ] **Cleanup obsolete code**
  - [ ] Remove old document processor implementation from main codebase
  - [ ] Clean up duplicate converter implementations
  - [ ] Remove deprecated utilities
  - [ ] Update imports in existing code to use new package structure
  - [ ] Remove unused dependencies

- [ ] **Documentation updates**
  - [ ] Create comprehensive API documentation
  - [ ] Update installation guides
  - [ ] Create migration guides for existing users
  - [ ] Document new package structure and dependencies