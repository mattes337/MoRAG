#!/usr/bin/env python3
"""
Demo script for MoRAG Database Integration
Showcases the new SQL Alchemy database functionality.
"""

import os
import tempfile
from morag_core.database import (
    DatabaseInitializer,
    get_database_manager,
    create_user,
    create_database_server,
    create_database,
    create_document,
    create_job,
    UserRole,
    DatabaseType,
    DocumentState,
    JobStatus,
    get_session_context
)

def main():
    """Demonstrate database integration features."""
    print("🗄️ MoRAG Database Integration Demo")
    print("=" * 50)
    
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    
    db_url = f"sqlite:///{db_path}"
    print(f"📍 Using database: {db_url}")
    
    try:
        # Step 1: Initialize database
        print("\n1️⃣ Initializing database...")
        initializer = DatabaseInitializer(db_url)
        success = initializer.initialize_database()
        
        if not success:
            print("❌ Database initialization failed")
            return
        
        print("✅ Database initialized successfully")
        
        # Step 2: Create users and all related data in single session
        print("\n2️⃣ Creating users and demo data...")
        db_manager = get_database_manager(db_url)

        with get_session_context(db_manager) as session:
            # Create admin user
            admin_user = create_user(
                session,
                name="Admin User",
                email="admin@morag.demo",
                role=UserRole.ADMIN
            )
            print(f"✅ Admin user created: {admin_user.email} (ID: {admin_user.id})")

            # Create regular user
            regular_user = create_user(
                session,
                name="John Doe",
                email="john@morag.demo",
                role=UserRole.USER
            )
            print(f"✅ Regular user created: {regular_user.email} (ID: {regular_user.id})")

            # Step 3: Create database server configuration
            print("\n3️⃣ Setting up database server...")
            server = create_database_server(
                session,
                user_id=admin_user.id,
                name="Local Qdrant Server",
                db_type=DatabaseType.QDRANT,
                host="localhost",
                port=6333,
                is_active=True
            )
            print(f"✅ Database server created: {server.name} (ID: {server.id})")

            # Step 4: Create logical database
            print("\n4️⃣ Creating logical database...")
            database = create_database(
                session,
                user_id=admin_user.id,
                server_id=server.id,
                name="Demo Documents",
                description="Demo database for document storage"
            )
            print(f"✅ Database created: {database.name} (ID: {database.id})")

            # Step 5: Create documents
            print("\n5️⃣ Adding documents...")
            doc1 = create_document(
                session,
                user_id=regular_user.id,
                name="sample.pdf",
                doc_type="document",
                database_id=database.id
            )
            print(f"✅ Document created: {doc1.name} (ID: {doc1.id})")

            doc2 = create_document(
                session,
                user_id=regular_user.id,
                name="presentation.pptx",
                doc_type="document",
                database_id=database.id
            )
            print(f"✅ Document created: {doc2.name} (ID: {doc2.id})")

            # Step 6: Create processing jobs
            print("\n6️⃣ Creating processing jobs...")
            job1 = create_job(
                session,
                user_id=regular_user.id,
                document_id=doc1.id,
                document_name=doc1.name,
                document_type=doc1.type
            )
            print(f"✅ Job created: {job1.document_name} (ID: {job1.id})")

            job2 = create_job(
                session,
                user_id=regular_user.id,
                document_id=doc2.id,
                document_name=doc2.name,
                document_type=doc2.type
            )
            print(f"✅ Job created: {job2.document_name} (ID: {job2.id})")

            # Step 8: Show relationships (while objects are still in session)
            print("\n8️⃣ Demonstrating relationships...")
            print(f"  👤 User '{regular_user.name}' has:")
            print(f"    📄 {len(regular_user.documents)} documents")
            print(f"    ⚙️ User settings: {regular_user.user_settings.theme.value} theme")
            print(f"    🔧 Jobs: {len(regular_user.jobs)}")
        
        # Step 7: Display database statistics
        print("\n7️⃣ Database Statistics:")
        info = initializer.get_database_info()
        print(f"  📊 Total tables: {len(info['tables'])}")
        print(f"  👥 Total users: {info['user_count']}")
        print(f"  📄 Total documents: {info['document_count']}")
        print(f"  🔗 Connection status: {'✅ OK' if info['connection_ok'] else '❌ Failed'}")
        print(f"  📋 Schema status: {'✅ Valid' if info['schema_valid'] else '❌ Invalid'}")

        print("\n🎉 Demo completed successfully!")
        print(f"📁 Database file: {db_path}")
        print("💡 You can inspect the database using any SQLite browser")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        try:
            os.unlink(db_path)
            print(f"🧹 Cleaned up database file: {db_path}")
        except OSError:
            pass


if __name__ == "__main__":
    main()
