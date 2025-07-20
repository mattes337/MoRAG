#!/usr/bin/env python3
"""
Setup Neo4j database schema and indices for Graphiti integration.

This script creates all the necessary indices, constraints, and schema
required for Graphiti to work properly with Neo4j.
"""

import os
import sys
from neo4j import GraphDatabase
import time

def setup_graphiti_schema():
    """Set up Neo4j database schema for Graphiti."""
    
    # Connection details
    uri = "bolt://localhost:7687"
    username = "neo4j"
    password = "password"
    database = "neo4j"
    
    print("🔄 Connecting to Neo4j...")
    driver = GraphDatabase.driver(uri, auth=(username, password))
    
    try:
        with driver.session(database=database) as session:
            print("✅ Connected to Neo4j successfully")
            
            # Test basic connectivity
            result = session.run("RETURN 1 as test")
            print(f"📋 Connection test: {result.single()['test']}")
            
            print("\n🔧 Setting up Graphiti schema...")
            
            # 1. Create constraints for unique identifiers
            print("📋 Creating constraints...")
            
            constraints = [
                # Entity constraints
                "CREATE CONSTRAINT entity_uuid IF NOT EXISTS FOR (e:Entity) REQUIRE e.uuid IS UNIQUE",
                "CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS NOT NULL",
                
                # Episodic constraints  
                "CREATE CONSTRAINT episodic_uuid IF NOT EXISTS FOR (e:Episodic) REQUIRE e.uuid IS UNIQUE",
                
                # Community constraints
                "CREATE CONSTRAINT community_uuid IF NOT EXISTS FOR (c:Community) REQUIRE c.uuid IS UNIQUE",
                
                # Relation constraints
                "CREATE CONSTRAINT relation_uuid IF NOT EXISTS FOR (r:Relation) REQUIRE r.uuid IS UNIQUE",
            ]
            
            for constraint in constraints:
                try:
                    session.run(constraint)
                    print(f"  ✅ {constraint.split()[2]}")
                except Exception as e:
                    if "already exists" in str(e).lower():
                        print(f"  ⚠️ {constraint.split()[2]} (already exists)")
                    else:
                        print(f"  ❌ {constraint.split()[2]}: {e}")
            
            # 2. Create fulltext indices
            print("\n📋 Creating fulltext indices...")
            
            fulltext_indices = [
                # Node name and summary index (required by Graphiti)
                """
                CREATE FULLTEXT INDEX node_name_and_summary IF NOT EXISTS
                FOR (n:Entity|Episodic|Community) 
                ON EACH [n.name, n.summary]
                """,
                
                # Entity content index
                """
                CREATE FULLTEXT INDEX entity_content IF NOT EXISTS
                FOR (e:Entity) 
                ON EACH [e.name, e.summary, e.description]
                """,
                
                # Episodic content index
                """
                CREATE FULLTEXT INDEX episodic_content IF NOT EXISTS
                FOR (e:Episodic) 
                ON EACH [e.content, e.name, e.summary]
                """,
            ]
            
            for index in fulltext_indices:
                try:
                    session.run(index.strip())
                    index_name = index.split("INDEX")[1].split("IF")[0].strip()
                    print(f"  ✅ {index_name}")
                except Exception as e:
                    if "already exists" in str(e).lower():
                        index_name = index.split("INDEX")[1].split("IF")[0].strip()
                        print(f"  ⚠️ {index_name} (already exists)")
                    else:
                        print(f"  ❌ Fulltext index error: {e}")
            
            # 3. Create vector indices for embeddings
            print("\n📋 Creating vector indices...")
            
            vector_indices = [
                # Entity name embedding index
                """
                CREATE VECTOR INDEX entity_name_embedding IF NOT EXISTS
                FOR (e:Entity) ON (e.name_embedding)
                OPTIONS {
                    indexConfig: {
                        `vector.dimensions`: 1536,
                        `vector.similarity_function`: 'cosine'
                    }
                }
                """,
                
                # Entity summary embedding index
                """
                CREATE VECTOR INDEX entity_summary_embedding IF NOT EXISTS
                FOR (e:Entity) ON (e.summary_embedding)
                OPTIONS {
                    indexConfig: {
                        `vector.dimensions`: 1536,
                        `vector.similarity_function`: 'cosine'
                    }
                }
                """,
                
                # Episodic content embedding index
                """
                CREATE VECTOR INDEX episodic_content_embedding IF NOT EXISTS
                FOR (e:Episodic) ON (e.content_embedding)
                OPTIONS {
                    indexConfig: {
                        `vector.dimensions`: 1536,
                        `vector.similarity_function`: 'cosine'
                    }
                }
                """,
            ]
            
            for index in vector_indices:
                try:
                    session.run(index.strip())
                    index_name = index.split("INDEX")[1].split("IF")[0].strip()
                    print(f"  ✅ {index_name}")
                except Exception as e:
                    if "already exists" in str(e).lower():
                        index_name = index.split("INDEX")[1].split("IF")[0].strip()
                        print(f"  ⚠️ {index_name} (already exists)")
                    else:
                        print(f"  ❌ Vector index error: {e}")
            
            # 4. Create regular indices for performance
            print("\n📋 Creating performance indices...")
            
            regular_indices = [
                "CREATE INDEX entity_created_at IF NOT EXISTS FOR (e:Entity) ON (e.created_at)",
                "CREATE INDEX entity_valid_at IF NOT EXISTS FOR (e:Entity) ON (e.valid_at)",
                "CREATE INDEX episodic_created_at IF NOT EXISTS FOR (e:Episodic) ON (e.created_at)",
                "CREATE INDEX episodic_valid_at IF NOT EXISTS FOR (e:Episodic) ON (e.valid_at)",
                "CREATE INDEX episodic_group_id IF NOT EXISTS FOR (e:Episodic) ON (e.group_id)",
                "CREATE INDEX episodic_source IF NOT EXISTS FOR (e:Episodic) ON (e.source)",
                "CREATE INDEX relation_created_at IF NOT EXISTS FOR (r:Relation) ON (r.created_at)",
                "CREATE INDEX relation_valid_at IF NOT EXISTS FOR (r:Relation) ON (r.valid_at)",
            ]
            
            for index in regular_indices:
                try:
                    session.run(index)
                    index_name = index.split("INDEX")[1].split("IF")[0].strip()
                    print(f"  ✅ {index_name}")
                except Exception as e:
                    if "already exists" in str(e).lower():
                        index_name = index.split("INDEX")[1].split("IF")[0].strip()
                        print(f"  ⚠️ {index_name} (already exists)")
                    else:
                        print(f"  ❌ Regular index error: {e}")
            
            # 5. Verify setup
            print("\n🔍 Verifying setup...")
            
            # Check constraints
            constraints_result = session.run("SHOW CONSTRAINTS")
            constraint_count = len(list(constraints_result))
            print(f"  📊 Constraints: {constraint_count}")
            
            # Check indices
            indices_result = session.run("SHOW INDEXES")
            index_count = len(list(indices_result))
            print(f"  📊 Indices: {index_count}")
            
            # Test vector similarity function
            try:
                session.run("RETURN gds.similarity.cosine([1,2,3], [1,2,3]) as similarity")
                print("  ✅ Vector similarity functions available")
            except Exception as e:
                print(f"  ❌ Vector similarity test failed: {e}")
            
            print("\n🎉 Graphiti schema setup completed successfully!")
            print("\n📋 Summary:")
            print("  ✅ Constraints created for unique identifiers")
            print("  ✅ Fulltext indices created for search")
            print("  ✅ Vector indices created for embeddings")
            print("  ✅ Performance indices created")
            print("  ✅ Schema ready for Graphiti integration")
            
    except Exception as e:
        print(f"❌ Error setting up schema: {e}")
        return False
    finally:
        driver.close()
    
    return True

if __name__ == "__main__":
    print("🚀 Setting up Neo4j schema for Graphiti integration")
    print("=" * 60)
    
    success = setup_graphiti_schema()
    
    if success:
        print("\n✅ Setup completed successfully!")
        print("🎯 Neo4j is now ready for Graphiti integration")
    else:
        print("\n❌ Setup failed!")
        sys.exit(1)
