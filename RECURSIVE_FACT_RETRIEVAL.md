# **Graph-Based RAG System Implementation Plan**

This document outlines a detailed implementation plan for a Retrieval Augmented Generation (RAG) system that leverages knowledge graphs (Neo4j), vector stores (Qdrant), and multiple Large Language Models (LLMs) for intelligent graph traversal, fact extraction, and response generation. This plan is designed to be consumed by an LLM-powered coding agent or a human developer.

## **1. System Overview**

The system is designed to answer complex user queries by intelligently traversing a knowledge graph, extracting relevant facts, critically evaluating them, and then synthesizing a comprehensive response. It comprises three main logical components:

1. **GraphTraversalAgent (GTA):** Responsible for navigating the Neo4j graph, identifying relevant nodes and relationships, and extracting *raw* facts.  
2. **FactCriticAgent (FCA):** Evaluates the relevance and quality of the raw facts extracted by the GTA, assigning a score and generating a user-friendly source description.  
3. **Orchestration Logic:** The central control flow that manages the overall process, including initial query processing, iterative graph traversal, fact collection, scoring, relevance decay, and final answer generation using a powerful LLM.

## **2. External Dependencies**

* **Neo4j Database:** For storing and querying the knowledge graph (nodes and relationships).  
* **Qdrant Vector Database:** For storing vector embeddings of text chunks associated with graph nodes and retrieving content.  
* **LLM APIs:** Access to at least two LLMs:  
  * A smaller, faster LLM for GTA and FCA (e.g., gemini-2.0-flash).  
  * A larger, more powerful reasoning LLM for final synthesis.

## **3. Data Structures**

### **3.1. RawFact (Output from GTA)**

{  
  "fact\_text": "A concise, extracted piece of information.",  
  "source\_node\_id": "ID\_of\_the\_node\_where\_this\_fact\_was\_found",  
  "source\_property": "Optional\_property\_name\_if\_fact\_from\_property",  
  "source\_qdrant\_chunk\_id": "Optional\_ID\_of\_Qdrant\_chunk\_if\_fact\_from\_chunk\_content",  
  "extracted\_from\_depth": "Integer representing the traversal depth when this fact was extracted (0 for initial entities)."  
}

### **3.2. ScoredFact (Output from FCA)**

{  
  "fact\_text": "A concise, extracted piece of information.",  
  "source\_node\_id": "ID\_of\_the\_node\_where\_this\_fact\_was\_found",  
  "source\_property": "Optional\_property\_name\_if\_fact\_from\_property",  
  "source\_qdrant\_chunk\_id": "Optional\_ID\_of\_Qdrant\_chunk\_if\_fact\_from\_chunk\_content",  
  "extracted\_from\_depth": "Integer representing the traversal depth when this fact was extracted.",  
  "score": 0.0 to 1.0,  // Relevance score assigned by FCA.  
  "source\_description": "A user-friendly description of the source (e.g., 'From the Wikipedia page for Neo4j', 'From a research paper on graph algorithms')."  
}

### **3.3. FinalFact (After Relevance Decay)**

This is the ScoredFact object with an additional final\_decayed\_score field.

{  
  "fact\_text": "...",  
  "source\_node\_id": "...",  
  "source\_property": "...",  
  "source\_qdrant\_chunk\_id": "...",  
  "extracted\_from\_depth": ...,  
  "score": ...,  
  "source\_description": "...",  
  "final\_decayed\_score": 0.0 to 1.0 // Score after applying depth-based decay.  
}

## **4. Component Details**

### **4.1. GraphTraversalAgent (GTA)**

* **Role:** To perform intelligent graph traversal based on the user query, identify potential next nodes to explore, and extract *raw* facts from the current node's context.  
* **Inputs:**  
  * user\_query (string)  
  * current\_node\_id (string)  
  * traversal\_depth (integer)  
  * max\_depth (integer)  
  * visited\_nodes (set of strings \- node IDs)  
  * graph\_schema (string \- optional, e.g., "Node labels: Person, Company. Relationship types: WORKS\_FOR, LOCATED\_IN.")  
* **Outputs:**  
  * raw\_extracted\_facts (JSON array of RawFact objects)  
  * next\_nodes\_to\_explore (string \- "STOP\_TRAVERSAL", "NONE", or comma-separated (node\_id, relationship\_type) tuples)  
* **Key Logic:**  
  1. **Node Information Retrieval:**  
     * Query Neo4j for properties of current\_node\_id.  
     * Query Neo4j for immediate neighbors and their relationship types connected to current\_node\_id.  
     * Query Qdrant (using current\_node\_id as a key or a semantic search on node properties) to retrieve associated text chunks/content.  
  2. **LLM Prompting:** Construct a prompt for the GTA LLM (smaller, faster model) as described in the previous turn, instructing it to:  
     * Extract all *potentially* relevant facts from the provided context (current\_node\_properties, current\_node\_qdrant\_content, neighbors\_and\_relations).  
     * Format these facts as the RawFact JSON array.  
     * Decide on the next\_nodes\_to\_explore based on the user\_query, current\_depth, max\_depth, and visited\_nodes.

### **4.2. FactCriticAgent (FCA)**

* **Role:** To evaluate the relevance of a single RawFact to the original user\_query, assign a numerical score, and generate a user-friendly source description.  
* **Inputs:**  
  * user\_query (string)  
  * raw\_fact\_json (single RawFact object as a JSON string)  
* **Outputs:**  
  * scored\_fact\_json (single ScoredFact object as a JSON string)  
* **Key Logic:**  
  1. **LLM Prompting:** Construct a prompt for the FCA LLM (smaller, faster model) as described in the previous turn, instructing it to:  
     * Read the user\_query and the raw\_fact\_json.  
     * Assign a score (0.0 to 1.0) indicating the relevance of fact\_text to user\_query.  
     * Generate a source\_description based on the source\_node\_id, source\_property, and source\_qdrant\_chunk\_id.  
     * Return the original raw\_fact\_json with the score and source\_description fields added, forming a ScoredFact object.

## **5. Orchestration Logic**

This is the main Python (or similar language) script that coordinates the GTA, FCA, and the final LLM.

import json  
import math  
from collections import deque

\# Assume these are initialized elsewhere with Neo4j and Qdrant connections  
\# and LLM API clients.  
\# neo4j\_driver \= ...  
\# qdrant\_client \= ...  
\# call\_gta\_llm \= lambda user\_q, node\_id, props, q\_content, neighbors, depth, max\_d, visited: ...  
\# call\_fca\_llm \= lambda user\_q, raw\_fact\_str: ...  
\# call\_stronger\_llm \= lambda user\_q, context\_str: ...  
\# extract\_entities\_from\_query \= lambda query: \["entity1", "entity2"\]  
\# get\_initial\_graph\_nodes \= lambda entities, neo4j\_driver, qdrant\_client: \["node\_id\_A", "node\_id\_B"\]  
\# parse\_json\_from\_llm\_response \= lambda llm\_resp, key: json.loads(llm\_resp.get(key, "\[\]"))  
\# parse\_text\_from\_llm\_response \= lambda llm\_resp, key: llm\_resp.get(key, "")  
\# parse\_next\_nodes \= lambda next\_nodes\_str: \[tuple(s.strip().split(', ')) for s in next\_nodes\_str.strip('()').split('), (')\] if next\_nodes\_str not in \["NONE", "STOP\_TRAVERSAL"\] else \[\]

def graph\_rag\_system(user\_query: str, max\_depth: int, decay\_rate: float \= 0.2) \-\> str:  
    """  
    Orchestrates the graph traversal, fact criticism, and final response generation.

    Args:  
        user\_query: The original query from the user.  
        max\_depth: The maximum traversal depth for the GraphTraversalAgent.  
        decay\_rate: The rate at which fact scores decay per depth level (e.g., 0.2 means 20% decay per level).

    Returns:  
        The final synthesized answer from the stronger LLM.  
    """

    \# 1. Initial Entity Extraction from User Query  
    \# This can be a simple NER model or another LLM call.  
    initial\_entities \= extract\_entities\_from\_query(user\_query)  
    if not initial\_entities:  
        return "Could not identify key entities in your query to start graph traversal."

    \# 2. Map Initial Entities to Graph Nodes  
    \# Find corresponding nodes in Neo4j and retrieve their associated Qdrant content.  
    initial\_node\_ids \= get\_initial\_graph\_nodes(initial\_entities, neo4j\_driver, qdrant\_client)  
    if not initial\_node\_ids:  
        return "No relevant starting nodes found in the knowledge graph for your query."

    all\_raw\_facts \= \[\]  
    \# Queue for breadth-first traversal: (node\_id, current\_depth)  
    nodes\_to\_explore\_queue \= deque(\[(node\_id, 0\) for node\_id in initial\_node\_ids\])  
    visited\_nodes \= set(initial\_node\_ids) \# Track visited nodes to prevent cycles and redundant work

    print(f"Starting graph traversal from initial nodes: {initial\_node\_ids}")

    while nodes\_to\_explore\_queue:  
        current\_node\_id, current\_depth \= nodes\_to\_explore\_queue.popleft()

        \# Stop if max depth reached for this path  
        if current\_depth \>= max\_depth:  
            print(f"Reached max depth ({max\_depth}) at node {current\_node\_id}. Stopping traversal for this path.")  
            continue

        print(f"Exploring node: {current\_node\_id} at depth {current\_depth}")

        \# Prepare context for GTA LLM  
        \# These functions would interact with Neo4j and Qdrant  
        current\_node\_properties \= neo4j\_driver.get\_node\_properties(current\_node\_id)  
        current\_node\_qdrant\_content \= qdrant\_client.get\_chunks\_for\_node(current\_node\_id)  
        neighbors\_and\_relations \= neo4j\_driver.get\_neighbors\_and\_relations(current\_node\_id)

        \# Call the GraphTraversalAgent LLM  
        try:  
            gta\_llm\_response \= call\_gta\_llm(  
                user\_query,  
                current\_node\_id,  
                json.dumps(current\_node\_properties), \# Pass as JSON string  
                json.dumps(current\_node\_qdrant\_content), \# Pass as JSON string  
                json.dumps(neighbors\_and\_relations), \# Pass as JSON string  
                current\_depth,  
                max\_depth,  
                json.dumps(list(visited\_nodes)) \# Pass as JSON string  
            )  
            raw\_facts\_from\_level \= parse\_json\_from\_llm\_response(gta\_llm\_response, "Extracted Facts")  
            next\_steps\_str \= parse\_text\_from\_llm\_response(gta\_llm\_response, "Next Nodes to Explore")

            \# Accumulate raw facts for later criticism  
            if raw\_facts\_from\_level:  
                all\_raw\_facts.extend(raw\_facts\_from\_level)  
                print(f"Extracted {len(raw\_facts\_from\_level)} raw facts from node {current\_node\_id}.")

            \# Plan next traversal  
            if next\_steps\_str not in \["STOP\_TRAVERSAL", "NONE"\]:  
                parsed\_next\_nodes \= parse\_next\_nodes(next\_steps\_str)  
                for next\_node\_id, \_ in parsed\_next\_nodes:  
                    if next\_node\_id not in visited\_nodes:  
                        nodes\_to\_explore\_queue.append((next\_node\_id, current\_depth \+ 1))  
                        visited\_nodes.add(next\_node\_id)  
                        print(f"Added {next\_node\_id} to queue for depth {current\_depth \+ 1}.")  
            else:  
                print(f"GTA decided to {next\_steps\_str} further traversal from node {current\_node\_id}.")

        except Exception as e:  
            print(f"Error calling GTA for node {current\_node\_id}: {e}")  
            \# Fallback: Continue traversal but skip fact extraction for this node if LLM call fails  
            \# Or, add a specific error fact to all\_raw\_facts to be handled by FCA/final LLM

    print(f"\\nFinished graph traversal. Total raw facts collected: {len(all\_raw\_facts)}")

    \# 3. Fact Filtering and Scoring by FactCriticAgent  
    all\_scored\_facts \= \[\]  
    for i, raw\_fact in enumerate(all\_raw\_facts):  
        print(f"Criticizing fact {i+1}/{len(all\_raw\_facts)}: {raw\_fact.get('fact\_text', 'N/A')}")  
        try:  
            fca\_llm\_response \= call\_fca\_llm(user\_query, json.dumps(raw\_fact))  
            scored\_fact \= parse\_json\_from\_llm\_response(fca\_llm\_response, "Output JSON")  
            if scored\_fact and "score" in scored\_fact: \# Ensure fact was successfully scored  
                all\_scored\_facts.append(scored\_fact)  
            else:  
                print(f"FCA did not return a valid scored fact for: {raw\_fact.get('fact\_text', 'N/A')}")  
        except Exception as e:  
            print(f"Error calling FCA for fact: {raw\_fact.get('fact\_text', 'N/A')}. Error: {e}")  
            \# Fallback: Optionally add the raw fact with a default low score if FCA fails

    print(f"Total facts after criticism: {len(all\_scored\_facts)}")

    \# 4. Apply Relevance Decay  
    final\_relevant\_facts \= \[\]  
    for fact in all\_scored\_facts:  
        initial\_score \= fact.get("score", 0.0)  
        depth \= fact.get("extracted\_from\_depth", 0\)  
        \# Decay formula: score\_at\_depth \= initial\_score \* (1 \- decay\_rate \* depth)  
        \# Ensure score doesn't go below 0.  
        decay\_factor \= (1 \- decay\_rate \* depth)  
        decayed\_score \= max(0.0, initial\_score \* decay\_factor)

        fact\['final\_decayed\_score'\] \= decayed\_score  
        final\_relevant\_facts.append(fact)

    \# Sort facts by their final decayed score, highest first  
    final\_relevant\_facts.sort(key=lambda x: x.get('final\_decayed\_score', 0.0), reverse=True)

    \# Filter out facts below a very low threshold if desired, to save context window  
    \# final\_relevant\_facts \= \[f for f in final\_relevant\_facts if f\['final\_decayed\_score'\] \> 0.05\]  
    print(f"Total facts after decay and sorting: {len(final\_relevant\_facts)}")

    \# 5. Final Synthesis (Stronger LLM)  
    if not final\_relevant\_facts:  
        return "I could not find enough relevant information in the knowledge graph to answer your query."

    \# Prepare context for the final LLM, prioritizing higher-scored facts.  
    \# This part needs careful token management.  
    formatted\_context\_for\_final\_llm \= \[\]  
    for fact in final\_relevant\_facts:  
        \# Include source\_description for better attribution in final answer  
        formatted\_context\_for\_final\_llm.append(  
            f"Fact (Score: {fact\['final\_decayed\_score'\]:.2f}, Source: {fact\['source\_description'\]}): {fact\['fact\_text'\]}"  
        )

    \# Join facts into a single string. Consider chunking this if it exceeds LLM context window.  
    final\_context\_string \= "\\n\\n".join(formatted\_context\_for\_final\_llm)

    print(f"\\nSending {len(formatted\_context\_for\_final\_llm)} facts to final LLM. Context length: {len(final\_context\_string)} characters.")

    try:  
        final\_answer \= call\_stronger\_llm(user\_query, final\_context\_string)  
        return final\_answer  
    except Exception as e:  
        print(f"Error calling final LLM: {e}")  
        return "An error occurred while generating the final answer. Please try again."

## **6. Key Parameters**

* max\_depth: Integer. Defines the maximum number of hops the GraphTraversalAgent will explore from the initial entities. A higher value means broader exploration but increased processing time and potential for irrelevant facts.  
* decay\_rate: Float (0.0 to 1.0). Controls how quickly the relevance score of a fact diminishes with each increasing traversal depth. A higher decay\_rate means facts from deeper levels are penalized more heavily.  
* **LLM Token Limits:** Crucial to monitor, especially for the final synthesis step. Implement truncation or further summarization of final\_relevant\_facts if the combined context exceeds the stronger LLM's capacity.

## **7. Error Handling and Fallbacks**

* **LLM Call Failures:** Implement try-except blocks around LLM calls. If an LLM call fails, log the error and decide on a fallback (e.g., skip processing that node/fact, return a default error message, or retry).  
* **Parsing Errors:** Ensure robust JSON parsing for LLM outputs. Invalid JSON should be handled gracefully, potentially leading to a retry or skipping that particular LLM output.  
* **Graph/Vector DB Errors:** Implement error handling for database queries (e.g., node not found, connection issues).  
* **Empty Results:** Handle cases where no initial entities are found, no graph nodes are retrieved, or no relevant facts are extracted.

This plan provides a solid foundation for implementing your advanced graph-based RAG system.