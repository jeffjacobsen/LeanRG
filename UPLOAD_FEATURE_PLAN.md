# User Document Upload Feature - Implementation Plan

## Overview

This plan outlines the implementation of a user document upload feature that allows users to upload one or more documents, process them into chunks, and create queryable knowledge graphs using a `project_name` as the database identifier.

## Feature Goals

- **User-Friendly Upload**: Allow users to upload multiple documents in various formats
- **Project Organization**: Use `project_name` as database prefix for multi-project support
- **Automatic Processing**: Seamless integration with existing LeanRAG pipeline
- **Content Management**: Track uploaded documents, processing status, and metadata

## Architecture Overview

### Directory Structure
```
data/
├── {project_name}/           # User project directory
│   ├── documents/            # Original uploaded documents
│   │   ├── doc1.pdf
│   │   ├── doc2.txt
│   │   └── metadata.json     # Document metadata and upload info
│   ├── processed/            # Extracted and processed content
│   │   ├── extracted_text.json
│   │   └── unique_contexts.json
│   ├── chunk.json            # Final chunked content
│   └── [standard LeanRAG outputs...]
```

### Database Organization
- **MySQL Database**: `{project_name}` (following existing pattern)
- **Qdrant Collection**: `{project_name}_entities`
- **Project Isolation**: Each project has separate database and vector storage

## Implementation Plan

### Phase 1: Core Upload Infrastructure

#### 1.1 Document Upload Handler (`upload_documents.py`)
**Purpose**: Handle file uploads and initial document management

**Features**:
- Support multiple file formats:
  - **Text**: `.txt`, `.md`
  - **PDF**: `.pdf` (using PyPDF2 or pdfplumber)
  - **Word**: `.docx` (using python-docx)
  - **HTML**: `.html` (using BeautifulSoup)
- Validate file types and sizes
- Create project directory structure
- Generate document metadata
- Handle duplicate detection

**Key Functions**:
```python
def upload_documents(project_name: str, file_paths: list, overwrite: bool = False)
def validate_file_format(file_path: str) -> bool
def extract_document_metadata(file_path: str) -> dict
def check_project_exists(project_name: str) -> bool
```

#### 1.2 Document Processing Pipeline (`process_documents.py`)
**Purpose**: Extract text from uploaded documents and prepare for chunking

**Features**:
- Text extraction from various formats
- Content normalization and cleaning
- Duplicate content detection across documents
- Integration with existing chunking pipeline

**Key Functions**:
```python
def extract_text_from_document(file_path: str, file_type: str) -> str
def clean_and_normalize_text(text: str) -> str
def create_unique_contexts(project_name: str) -> dict
def generate_chunk_file(project_name: str, max_tokens: int = 512, overlap: int = 64)
```

#### 1.3 Project Management (`project_manager.py`)
**Purpose**: Manage project lifecycle and metadata

**Features**:
- Create/delete projects
- List existing projects
- Track processing status
- Manage project configuration

**Key Functions**:
```python
def create_project(project_name: str, description: str = "") -> bool
def delete_project(project_name: str, confirm: bool = False) -> bool
def list_projects() -> list
def get_project_status(project_name: str) -> dict
def update_project_metadata(project_name: str, metadata: dict)
```

### Phase 2: Web Interface (Optional)

#### 2.1 Upload Web Interface
**Purpose**: Provide user-friendly web interface for document upload

**Technology**: Streamlit or FastAPI + HTML
**Features**:
- Drag-and-drop file upload
- Project name input and validation
- Upload progress tracking
- Processing status display
- Error handling and user feedback

#### 2.2 Project Dashboard
**Purpose**: Manage existing projects and view processing status

**Features**:
- List all projects with metadata
- View document lists and processing status
- Delete projects and documents
- Initiate reprocessing
- Basic project statistics

### Phase 3: Enhanced Processing Integration

#### 3.1 Automated Pipeline Execution
**Purpose**: Automatically trigger LeanRAG pipeline after upload

**Features**:
- Auto-trigger after successful upload
- Configurable processing options
- Progress tracking and notifications
- Error handling and retry logic

**Integration Points**:
```bash
# Automated sequence after upload
python upload_documents.py --project project_name --files file1.pdf file2.txt
python process_documents.py --project project_name
python process_chunk.py project_name -c config.yaml
python deal_triple.py --input data/project_name --output data/project_name -c config.yaml
python build_graph.py -p data/project_name -c config.yaml
```

#### 3.2 Configuration Management
**Purpose**: Project-specific configuration and settings

**Features**:
- Per-project configuration files
- Override global settings
- Processing parameter customization
- API provider selection per project

## Technical Implementation Details

### File Format Support

#### Text Extraction Libraries
```python
# Add to requirements.txt
PyPDF2==3.0.1              # PDF processing
python-docx==0.8.11         # Word document processing
pdfplumber==0.9.0           # Alternative PDF processing
beautifulsoup4==4.12.2      # HTML processing
chardet==5.2.0              # Character encoding detection
```

#### Document Metadata Schema
```json
{
  "project_name": "my_project",
  "created_at": "2024-08-26T10:00:00Z",
  "documents": [
    {
      "filename": "document1.pdf",
      "original_path": "/path/to/document1.pdf",
      "size_bytes": 1024000,
      "upload_date": "2024-08-26T10:00:00Z",
      "file_type": "pdf",
      "pages": 10,
      "word_count": 5000,
      "processed": true,
      "hash": "md5_hash_of_content"
    }
  ],
  "processing_status": {
    "upload_complete": true,
    "text_extracted": true,
    "chunked": true,
    "entities_extracted": false,
    "graph_built": false,
    "last_updated": "2024-08-26T10:05:00Z"
  },
  "settings": {
    "chunk_size": 512,
    "chunk_overlap": 64,
    "llm_provider": "groq"
  }
}
```

### Command Line Interface

#### Upload Command
```bash
python upload_documents.py \
  --project "company_docs" \
  --files report1.pdf manual.docx notes.txt \
  --description "Company documentation project" \
  --auto-process
```

#### Project Management Commands
```bash
# List projects
python project_manager.py --list

# Project status
python project_manager.py --status company_docs

# Delete project
python project_manager.py --delete company_docs --confirm

# Reprocess project
python project_manager.py --reprocess company_docs
```

### Database Schema Extensions

#### Project Metadata Table
```sql
CREATE TABLE IF NOT EXISTS projects (
    project_name VARCHAR(255) PRIMARY KEY,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    document_count INT DEFAULT 0,
    total_chunks INT DEFAULT 0,
    processing_status JSON,
    settings JSON
);
```

## Integration with Existing System

### Configuration Updates

#### config.yaml Extensions
```yaml
# User Upload Configuration
upload_conf:
  max_file_size_mb: 100
  allowed_extensions: [".pdf", ".txt", ".docx", ".md", ".html"]
  auto_process: true
  notification_email: null
  
# Project-specific overrides
project_defaults:
  chunk_size: 512
  chunk_overlap: 64
  max_entities_per_chunk: 50
```

### Error Handling and Validation

#### File Validation
- File size limits (configurable)
- File type validation
- Virus scanning (optional)
- Content validation (readable text)
- Duplicate detection

#### Project Validation
- Name format validation (alphanumeric + underscores)
- Existing project checks
- Database name conflicts
- Storage space checks

## Security Considerations

### File Upload Security
- File type validation beyond extensions
- Content scanning for malicious code
- Size limits and rate limiting
- Sandboxed text extraction
- Input sanitization

### Data Privacy
- Project isolation (database and files)
- Optional encryption at rest
- Access control (future feature)
- Audit logging
- Data retention policies

## Testing Strategy

### Unit Tests
- File upload validation
- Text extraction accuracy
- Project management operations
- Database operations
- Error handling scenarios

### Integration Tests
- End-to-end upload process
- Pipeline integration
- Multiple file format handling
- Project lifecycle management

### Performance Tests
- Large file handling
- Multiple concurrent uploads
- Memory usage optimization
- Processing speed benchmarks

## Future Enhancements

### Advanced Features
- **Incremental Updates**: Add/remove documents from existing projects
- **Document Versioning**: Track document changes over time
- **Batch Processing**: Handle large document collections efficiently
- **Content Preview**: Quick document preview before processing
- **Export/Import**: Project backup and migration tools

### API Integration
- **REST API**: Programmatic access to upload functionality
- **Webhooks**: Processing completion notifications
- **SDK**: Python client library for integration
- **CLI Tools**: Advanced command-line interface

### Monitoring and Analytics
- **Processing Metrics**: Track performance and resource usage
- **Quality Metrics**: Document processing success rates
- **Usage Analytics**: Project and feature usage statistics
- **Alert System**: Notifications for failures or issues

## Implementation Timeline

### Week 1: Core Infrastructure
- Document upload handler
- Basic text extraction
- Project directory management

### Week 2: Processing Integration
- Integration with existing chunking
- Automated pipeline triggers
- Error handling and validation

### Week 3: Project Management
- Project lifecycle management
- Configuration and metadata handling
- Command-line interface

### Week 4: Testing and Polish
- Comprehensive testing suite
- Documentation updates
- Performance optimization
- Bug fixes and refinements

## Success Criteria

1. **Functionality**: Users can upload documents and generate queryable knowledge graphs
2. **Reliability**: Robust error handling with clear user feedback
3. **Performance**: Handle documents up to 100MB with reasonable processing times
4. **Integration**: Seamless integration with existing LeanRAG pipeline
5. **Usability**: Clear documentation and intuitive command-line interface
6. **Scalability**: Support for multiple concurrent projects

This plan provides a comprehensive roadmap for implementing user document upload functionality while maintaining compatibility with the existing LeanRAG system architecture.