"""
SQLAlchemy Database Models
Based on the DATABASE.md specification and Prisma schema.
"""

from sqlalchemy import (
    create_engine, Column, String, Integer, Float, Boolean, DateTime, 
    ForeignKey, Text, Enum as SQLEnum, UniqueConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.mysql import CHAR
from datetime import datetime
import enum
import uuid

# Base class for all models
Base = declarative_base()

# Enums
class UserRole(enum.Enum):
    ADMIN = "ADMIN"
    USER = "USER"
    VIEWER = "VIEWER"

class Theme(enum.Enum):
    LIGHT = "LIGHT"
    DARK = "DARK"
    SYSTEM = "SYSTEM"

class DocumentState(enum.Enum):
    PENDING = "PENDING"
    INGESTING = "INGESTING"
    INGESTED = "INGESTED"
    DEPRECATED = "DEPRECATED"
    DELETED = "DELETED"

class DatabaseType(enum.Enum):
    QDRANT = "QDRANT"
    NEO4J = "NEO4J"
    PINECONE = "PINECONE"
    WEAVIATE = "WEAVIATE"
    CHROMA = "CHROMA"

class JobStatus(enum.Enum):
    PENDING = "PENDING"
    WAITING_FOR_REMOTE_WORKER = "WAITING_FOR_REMOTE_WORKER"
    PROCESSING = "PROCESSING"
    FINISHED = "FINISHED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

# Helper function to generate UUID
def generate_uuid():
    return str(uuid.uuid4())

# Models
class User(Base):
    __tablename__ = 'users'
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    avatar = Column(String(500), nullable=True)
    role = Column(SQLEnum(UserRole), default=UserRole.USER, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    user_settings = relationship("UserSettings", back_populates="user", uselist=False, cascade="all, delete-orphan")
    databases = relationship("Database", back_populates="user", cascade="all, delete-orphan")
    documents = relationship("Document", back_populates="user", cascade="all, delete-orphan")
    api_keys = relationship("ApiKey", back_populates="user", cascade="all, delete-orphan")
    jobs = relationship("Job", back_populates="user", cascade="all, delete-orphan")
    database_servers = relationship("DatabaseServer", back_populates="user", cascade="all, delete-orphan")

class UserSettings(Base):
    __tablename__ = 'user_settings'
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey('users.id', ondelete='CASCADE'), unique=True, nullable=False)
    theme = Column(SQLEnum(Theme), default=Theme.LIGHT, nullable=False)
    language = Column(String(10), default="en", nullable=False)
    notifications = Column(Boolean, default=True, nullable=False)
    auto_save = Column(Boolean, default=True, nullable=False)
    default_database = Column(String(36), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="user_settings")

class DatabaseServer(Base):
    __tablename__ = 'database_servers'
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    name = Column(String(255), nullable=False)
    type = Column(SQLEnum(DatabaseType), nullable=False)
    host = Column(String(255), nullable=False)
    port = Column(Integer, nullable=False)
    username = Column(String(255), nullable=True)
    password = Column(String(255), nullable=True)
    api_key = Column(String(500), nullable=True)
    database = Column(String(255), nullable=True)
    collection = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_connected = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    user_id = Column(String(36), ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="database_servers")
    databases = relationship("Database", back_populates="server", cascade="all, delete-orphan")

class Database(Base):
    __tablename__ = 'databases'
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    document_count = Column(Integer, default=0, nullable=False)
    last_updated = Column(DateTime, default=datetime.utcnow, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    user_id = Column(String(36), ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    server_id = Column(String(36), ForeignKey('database_servers.id', ondelete='CASCADE'), nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="databases")
    server = relationship("DatabaseServer", back_populates="databases")
    documents = relationship("Document", back_populates="database")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('name', 'user_id', name='unique_database_name_per_user'),
    )

class Document(Base):
    __tablename__ = 'documents'
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    name = Column(String(255), nullable=False)
    type = Column(String(50), nullable=False)
    state = Column(SQLEnum(DocumentState), default=DocumentState.PENDING, nullable=False)
    version = Column(Integer, default=1, nullable=False)
    chunks = Column(Integer, default=0, nullable=False)
    quality = Column(Float, default=0.0, nullable=False)
    upload_date = Column(DateTime, default=datetime.utcnow, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    user_id = Column(String(36), ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    database_id = Column(String(36), ForeignKey('databases.id', ondelete='SET NULL'), nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="documents")
    database = relationship("Database", back_populates="documents")
    jobs = relationship("Job", back_populates="document", cascade="all, delete-orphan")

class ApiKey(Base):
    __tablename__ = 'api_keys'
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    name = Column(String(255), nullable=False)
    key = Column(String(500), unique=True, nullable=False)
    created = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_used = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    user_id = Column(String(36), ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="api_keys")

class Job(Base):
    __tablename__ = 'jobs'
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    document_name = Column(String(255), nullable=False)
    document_type = Column(String(50), nullable=False)
    start_date = Column(DateTime, default=datetime.utcnow, nullable=False)
    end_date = Column(DateTime, nullable=True)
    status = Column(SQLEnum(JobStatus), default=JobStatus.PENDING, nullable=False)
    percentage = Column(Integer, default=0, nullable=False)
    summary = Column(Text, default="", nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    document_id = Column(String(36), ForeignKey('documents.id', ondelete='CASCADE'), nullable=False)
    user_id = Column(String(36), ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    
    # Relationships
    document = relationship("Document", back_populates="jobs")
    user = relationship("User", back_populates="jobs")

# Database configuration and session management
class DatabaseManager:
    """Database manager for handling connections and sessions."""
    
    def __init__(self, database_url: str):
        self.engine = create_engine(database_url, echo=False)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
    
    def create_tables(self):
        """Create all tables in the database."""
        Base.metadata.create_all(bind=self.engine)
    
    def drop_tables(self):
        """Drop all tables in the database."""
        Base.metadata.drop_all(bind=self.engine)
    
    def get_session(self):
        """Get a database session."""
        return self.SessionLocal()
    
    def close_session(self, session):
        """Close a database session."""
        session.close()

# Utility functions
def create_user(session, name: str, email: str, role: UserRole = UserRole.USER, avatar: str = None) -> User:
    """Create a new user with default settings."""
    user = User(
        name=name,
        email=email,
        role=role,
        avatar=avatar
    )
    session.add(user)
    session.flush()  # Get the user ID
    
    # Create default user settings
    user_settings = UserSettings(
        user_id=user.id,
        theme=Theme.LIGHT,
        language="en",
        notifications=True,
        auto_save=True
    )
    session.add(user_settings)
    session.commit()
    
    return user

def create_database_server(session, user_id: str, name: str, db_type: DatabaseType, 
                          host: str, port: int, **kwargs) -> DatabaseServer:
    """Create a new database server configuration."""
    server = DatabaseServer(
        user_id=user_id,
        name=name,
        type=db_type,
        host=host,
        port=port,
        username=kwargs.get('username'),
        password=kwargs.get('password'),
        api_key=kwargs.get('api_key'),
        database=kwargs.get('database'),
        collection=kwargs.get('collection'),
        is_active=kwargs.get('is_active', False)
    )
    session.add(server)
    session.commit()
    
    return server

def create_database(session, user_id: str, server_id: str, name: str, description: str) -> Database:
    """Create a new database."""
    database = Database(
        user_id=user_id,
        server_id=server_id,
        name=name,
        description=description
    )
    session.add(database)
    session.commit()
    
    return database

def create_document(session, user_id: str, name: str, doc_type: str, 
                   database_id: str = None) -> Document:
    """Create a new document."""
    document = Document(
        user_id=user_id,
        database_id=database_id,
        name=name,
        type=doc_type
    )
    session.add(document)
    session.commit()
    
    return document

def create_api_key(session, user_id: str, name: str, key: str) -> ApiKey:
    """Create a new API key."""
    api_key = ApiKey(
        user_id=user_id,
        name=name,
        key=key
    )
    session.add(api_key)
    session.commit()
    
    return api_key

def create_job(session, user_id: str, document_id: str, document_name: str, 
               document_type: str) -> Job:
    """Create a new processing job."""
    job = Job(
        user_id=user_id,
        document_id=document_id,
        document_name=document_name,
        document_type=document_type
    )
    session.add(job)
    session.commit()
    
    return job

def update_document_count(session, database_id: str):
    """Update the document count for a database."""
    database = session.query(Database).filter(Database.id == database_id).first()
    if database:
        count = session.query(Document).filter(
            Document.database_id == database_id,
            Document.state != DocumentState.DELETED
        ).count()
        database.document_count = count
        database.last_updated = datetime.utcnow()
        session.commit()

def update_job_progress(session, job_id: str, status: JobStatus, percentage: int, 
                       summary: str = ""):
    """Update job progress and status."""
    job = session.query(Job).filter(Job.id == job_id).first()
    if job:
        job.status = status
        job.percentage = percentage
        job.summary = summary
        
        # Set end date if job is finished
        if status in [JobStatus.FINISHED, JobStatus.FAILED, JobStatus.CANCELLED]:
            job.end_date = datetime.utcnow()
        
        session.commit()

# Example usage and initialization
if __name__ == "__main__":
    # Example database URL (adjust for your database)
    DATABASE_URL = "mysql+pymysql://username:password@localhost:3306/database_name"
    
    # Initialize database manager
    db_manager = DatabaseManager(DATABASE_URL)
    
    # Create tables
    db_manager.create_tables()
    
    # Example usage
    session = db_manager.get_session()
    try:
        # Create a user
        user = create_user(session, "John Doe", "john@example.com", UserRole.USER)
        print(f"Created user: {user.id}")
        
        # Create a database server
        server = create_database_server(
            session, user.id, "Local Qdrant", DatabaseType.QDRANT, 
            "localhost", 6333, is_active=True
        )
        print(f"Created server: {server.id}")
        
        # Create a database
        database = create_database(
            session, user.id, server.id, "My Documents", 
            "Personal document collection"
        )
        print(f"Created database: {database.id}")
        
    except Exception as e:
        session.rollback()
        print(f"Error: {e}")
    finally:
        db_manager.close_session(session)