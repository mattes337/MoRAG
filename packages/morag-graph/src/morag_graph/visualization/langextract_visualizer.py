"""LangExtract-based visualization for MoRAG graphs."""

import os
import tempfile
import webbrowser
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import langextract as lx

    LANGEXTRACT_AVAILABLE = True
except ImportError:
    LANGEXTRACT_AVAILABLE = False
    lx = None

from ..models import Entity, Relation


class LangExtractVisualizer:
    """Visualizer that uses LangExtract's HTML visualization capabilities."""

    def __init__(
        self,
        model_id: str = "gemini-2.0-flash",
        api_key: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the LangExtract visualizer.

        Args:
            model_id: LangExtract model ID
            api_key: API key for LangExtract
            **kwargs: Additional arguments
        """
        if not LANGEXTRACT_AVAILABLE:
            raise ImportError(
                "LangExtract is not available. Please install it with: pip install langextract"
            )

        self.model_id = model_id
        self.api_key = api_key or self._get_api_key()

        if not self.api_key:
            raise ValueError(
                "No API key found for LangExtract. Set LANGEXTRACT_API_KEY or GOOGLE_API_KEY."
            )

    def _get_api_key(self) -> Optional[str]:
        """Get API key from environment variables."""
        return os.getenv("LANGEXTRACT_API_KEY") or os.getenv("GOOGLE_API_KEY")

    def visualize_extraction(
        self,
        text: str,
        entities: Optional[List[Entity]] = None,
        relations: Optional[List[Relation]] = None,
        output_file: Optional[str] = None,
        open_browser: bool = True,
        **extraction_kwargs,
    ) -> str:
        """Visualize entity and relation extraction using LangExtract.

        Args:
            text: Text to visualize extraction for
            entities: Optional pre-extracted entities (for reference)
            relations: Optional pre-extracted relations (for reference)
            output_file: Optional output HTML file path
            open_browser: Whether to open the visualization in browser
            **extraction_kwargs: Additional arguments for LangExtract

        Returns:
            Path to the generated HTML file
        """
        # Create extraction prompt
        prompt = "Extract entities and relationships from the text. Focus on meaningful connections between people, organizations, locations, and concepts."

        # Create examples if we have pre-extracted data
        examples = []
        if entities or relations:
            examples = self._create_examples_from_data(text, entities, relations)

        # Run LangExtract with visualization
        result = lx.extract(
            text_or_documents=text,
            prompt_description=prompt,
            examples=examples,
            model_id=self.model_id,
            api_key=self.api_key,
            **extraction_kwargs,
        )

        # Generate HTML visualization
        html_content = self._generate_html_visualization(
            result, text, entities, relations
        )

        # Save to file
        if output_file is None:
            output_file = tempfile.mktemp(suffix=".html")

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        # Open in browser if requested
        if open_browser:
            webbrowser.open(f"file://{os.path.abspath(output_file)}")

        return output_file

    def _create_examples_from_data(
        self,
        text: str,
        entities: Optional[List[Entity]],
        relations: Optional[List[Relation]],
    ) -> List[Any]:
        """Create LangExtract examples from existing entity/relation data."""
        examples: List[Dict[str, Any]] = []

        if not entities and not relations:
            return examples

        try:
            # Create a simple example based on the data
            extractions = []

            # Add entity extractions
            if entities:
                for entity in entities[:5]:  # Limit to first 5 entities
                    extractions.append(
                        lx.data.Extraction(
                            extraction_class=entity.type.lower(),
                            extraction_text=entity.name,
                            attributes={
                                "entity_type": entity.type,
                                "confidence": entity.confidence,
                            },
                        )
                    )

            # Add relation extractions
            if relations:
                for relation in relations[:5]:  # Limit to first 5 relations
                    extractions.append(
                        lx.data.Extraction(
                            extraction_class="relationship",
                            extraction_text=relation.context
                            or f"{relation.source_entity_id} {relation.type} {relation.target_entity_id}",
                            attributes={
                                "source_entity": relation.source_entity_id,
                                "target_entity": relation.target_entity_id,
                                "relationship_type": relation.type,
                                "confidence": relation.confidence,
                            },
                        )
                    )

            if extractions:
                examples.append(
                    lx.data.ExampleData(
                        text=text[:500] + "..."
                        if len(text) > 500
                        else text,  # Truncate for example
                        extractions=extractions,
                    )
                )

        except Exception:
            # If example creation fails, return empty list
            pass

        return examples

    def _generate_html_visualization(
        self,
        langextract_result: Any,
        original_text: str,
        entities: Optional[List[Entity]] = None,
        relations: Optional[List[Relation]] = None,
    ) -> str:
        """Generate HTML visualization combining LangExtract results with MoRAG data."""

        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>MoRAG Graph Visualization</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .section { margin-bottom: 30px; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
                .section h2 { color: #333; border-bottom: 2px solid #007acc; padding-bottom: 10px; }
                .text-content { background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin: 10px 0; }
                .extraction-item { margin: 10px 0; padding: 10px; background-color: #f0f8ff; border-left: 4px solid #007acc; }
                .entity { background-color: #e6f3ff; border-left-color: #0066cc; }
                .relation { background-color: #fff0e6; border-left-color: #ff6600; }
                .langextract-item { background-color: #f0fff0; border-left-color: #00cc66; }
                .confidence { font-weight: bold; color: #666; }
                .metadata { font-size: 0.9em; color: #888; }
                .stats { display: flex; gap: 20px; }
                .stat-box { padding: 10px; background-color: #f5f5f5; border-radius: 5px; text-align: center; }
            </style>
        </head>
        <body>
            <h1>MoRAG Graph Visualization</h1>

            <div class="section">
                <h2>Original Text</h2>
                <div class="text-content">{original_text}</div>
            </div>

            <div class="section">
                <h2>Extraction Statistics</h2>
                <div class="stats">
                    <div class="stat-box">
                        <strong>{morag_entities_count}</strong><br>
                        MoRAG Entities
                    </div>
                    <div class="stat-box">
                        <strong>{morag_relations_count}</strong><br>
                        MoRAG Relations
                    </div>
                    <div class="stat-box">
                        <strong>{langextract_extractions_count}</strong><br>
                        LangExtract Extractions
                    </div>
                </div>
            </div>

            {morag_entities_section}

            {morag_relations_section}

            {langextract_section}

        </body>
        </html>
        """

        # Format original text (truncate if too long)
        display_text = original_text
        if len(display_text) > 2000:
            display_text = display_text[:2000] + "... (truncated)"

        # Generate MoRAG entities section
        morag_entities_section = ""
        if entities:
            morag_entities_section = "<div class='section'><h2>MoRAG Entities</h2>"
            for entity in entities:
                morag_entities_section += f"""
                <div class="extraction-item entity">
                    <strong>{entity.name}</strong> ({entity.type})
                    <div class="confidence">Confidence: {entity.confidence:.2f}</div>
                    <div class="metadata">Source: {entity.source_doc_id or 'Unknown'}</div>
                </div>
                """
            morag_entities_section += "</div>"

        # Generate MoRAG relations section
        morag_relations_section = ""
        if relations:
            morag_relations_section = "<div class='section'><h2>MoRAG Relations</h2>"
            for relation in relations:
                morag_relations_section += f"""
                <div class="extraction-item relation">
                    <strong>{relation.source_entity_id}</strong> → <strong>{relation.type}</strong> → <strong>{relation.target_entity_id}</strong>
                    <div class="confidence">Confidence: {relation.confidence:.2f}</div>
                    <div class="metadata">Context: {relation.context or 'N/A'}</div>
                </div>
                """
            morag_relations_section += "</div>"

        # Generate LangExtract section
        langextract_section = ""
        if langextract_result and hasattr(langextract_result, "extractions"):
            langextract_section = (
                "<div class='section'><h2>LangExtract Extractions</h2>"
            )
            for extraction in langextract_result.extractions:
                langextract_section += f"""
                <div class="extraction-item langextract-item">
                    <strong>{extraction.extraction_class}</strong>: {extraction.extraction_text}
                    <div class="metadata">Attributes: {extraction.attributes or {}}</div>
                </div>
                """
            langextract_section += "</div>"

        # Fill in the template
        html_content = html_template.format(
            original_text=display_text,
            morag_entities_count=len(entities) if entities else 0,
            morag_relations_count=len(relations) if relations else 0,
            langextract_extractions_count=len(langextract_result.extractions)
            if langextract_result and hasattr(langextract_result, "extractions")
            else 0,
            morag_entities_section=morag_entities_section,
            morag_relations_section=morag_relations_section,
            langextract_section=langextract_section,
        )

        return html_content

    def visualize_graph(
        self,
        entities: List[Entity],
        relations: List[Relation],
        output_file: Optional[str] = None,
        open_browser: bool = True,
    ) -> str:
        """Visualize a complete graph of entities and relations.

        Args:
            entities: List of entities to visualize
            relations: List of relations to visualize
            output_file: Optional output HTML file path
            open_browser: Whether to open the visualization in browser

        Returns:
            Path to the generated HTML file
        """
        # Create a simple graph visualization
        html_content = self._generate_graph_html(entities, relations)

        # Save to file
        if output_file is None:
            output_file = tempfile.mktemp(suffix=".html")

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        # Open in browser if requested
        if open_browser:
            webbrowser.open(f"file://{os.path.abspath(output_file)}")

        return output_file

    def _generate_graph_html(
        self, entities: List[Entity], relations: List[Relation]
    ) -> str:
        """Generate HTML for graph visualization."""

        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>MoRAG Graph Network</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .graph-container { width: 100%; height: 600px; border: 1px solid #ddd; margin: 20px 0; }
                .entity-list, .relation-list { margin: 20px 0; }
                .entity-item, .relation-item { margin: 5px 0; padding: 10px; background-color: #f9f9f9; border-radius: 5px; }
                .entity-item { border-left: 4px solid #007acc; }
                .relation-item { border-left: 4px solid #ff6600; }
                .stats { display: flex; gap: 20px; margin: 20px 0; }
                .stat-box { padding: 15px; background-color: #f5f5f5; border-radius: 5px; text-align: center; }
            </style>
        </head>
        <body>
            <h1>MoRAG Graph Network</h1>

            <div class="stats">
                <div class="stat-box">
                    <strong>{entity_count}</strong><br>
                    Entities
                </div>
                <div class="stat-box">
                    <strong>{relation_count}</strong><br>
                    Relations
                </div>
                <div class="stat-box">
                    <strong>{avg_confidence:.2f}</strong><br>
                    Avg Confidence
                </div>
            </div>

            <div class="graph-container">
                <p style="text-align: center; margin-top: 250px; color: #666;">
                    Interactive graph visualization would be rendered here.<br>
                    Consider integrating with D3.js, vis.js, or similar libraries for interactive graphs.
                </p>
            </div>

            <h2>Entities ({entity_count})</h2>
            <div class="entity-list">
                {entity_list}
            </div>

            <h2>Relations ({relation_count})</h2>
            <div class="relation-list">
                {relation_list}
            </div>

        </body>
        </html>
        """

        # Generate entity list
        entity_list = ""
        for entity in entities:
            entity_list += f"""
            <div class="entity-item">
                <strong>{entity.name}</strong> ({entity.type})
                <span style="float: right;">Confidence: {entity.confidence:.2f}</span>
            </div>
            """

        # Generate relation list
        relation_list = ""
        for relation in relations:
            relation_list += f"""
            <div class="relation-item">
                <strong>{relation.source_entity_id}</strong> → {relation.type} → <strong>{relation.target_entity_id}</strong>
                <span style="float: right;">Confidence: {relation.confidence:.2f}</span>
            </div>
            """

        # Calculate average confidence
        all_confidences = [e.confidence for e in entities] + [
            r.confidence for r in relations
        ]
        avg_confidence = (
            sum(all_confidences) / len(all_confidences) if all_confidences else 0
        )

        # Fill in the template
        html_content = html_template.format(
            entity_count=len(entities),
            relation_count=len(relations),
            avg_confidence=avg_confidence,
            entity_list=entity_list,
            relation_list=relation_list,
        )

        return html_content
