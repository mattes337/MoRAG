"""Web content formatter for LLM documentation compliance."""

import re
from pathlib import Path
from typing import Dict, Any, Optional, List
from urllib.parse import urlparse, urljoin
import structlog

logger = structlog.get_logger(__name__)


class WebFormatter:
    """Formats web content according to LLM documentation specifications."""

    def format_web_content(
        self,
        raw_content: str,
        url: str,
        metadata: Dict[str, Any]
    ) -> str:
        """Format web content according to LLM documentation format.

        Args:
            raw_content: Raw extracted web content
            url: Original URL
            metadata: Web page metadata

        Returns:
            Formatted markdown content following LLM specifications
        """
        # Extract page title from metadata or content
        page_title = metadata.get('title', self._extract_title_from_content(raw_content))
        if not page_title:
            parsed_url = urlparse(url)
            page_title = parsed_url.netloc or 'Unknown Page'

        # Build formatted content
        formatted_parts = []

        # Header Section: "Web Page: page-title"
        formatted_parts.append(f"# Web Page: {page_title}")
        formatted_parts.append("")

        # Page Information Section
        formatted_parts.append("## Page Information")

        # Extract metadata for information section
        author = metadata.get('author', 'Unknown')
        publication_date = metadata.get('publication_date', metadata.get('date', 'Unknown'))
        word_count = metadata.get('word_count', self._count_words(raw_content))
        language = metadata.get('language', 'Unknown')
        last_modified = metadata.get('last_modified', 'Unknown')

        formatted_parts.extend([
            f"- **URL**: {url}",
            f"- **Title**: {page_title}",
            f"- **Author**: {author}",
            f"- **Publication Date**: {publication_date}",
            f"- **Word Count**: {word_count}",
            f"- **Language**: {language}",
            f"- **Last Modified**: {last_modified}",
            ""
        ])

        # Parse content sections
        sections = self._parse_web_content(raw_content)

        # Main Content Section
        main_content = sections.get('main_content', '')
        if main_content:
            formatted_parts.append("## Main Content")
            formatted_parts.append("")
            formatted_parts.append(main_content.strip())
            formatted_parts.append("")

        # Subsections (if any)
        subsections = sections.get('subsections', [])
        for subsection in subsections:
            if subsection.get('title') and subsection.get('content'):
                # Ensure proper heading level (### for subsections)
                title = subsection['title'].strip()
                if not title.startswith('#'):
                    title = f"### {title}"
                elif title.startswith('#'):
                    # Adjust heading level to be at least ###
                    level = len(title) - len(title.lstrip('#'))
                    if level < 3:
                        title = f"### {title.lstrip('#').strip()}"

                formatted_parts.append(title)
                formatted_parts.append("")
                formatted_parts.append(subsection['content'].strip())
                formatted_parts.append("")

        # Links Section
        links = sections.get('links', [])
        if links:
            formatted_parts.append("## Links")
            formatted_parts.append("")
            for link in links:
                if isinstance(link, dict):
                    text = link.get('text', 'Link')
                    href = link.get('href', '#')
                    formatted_parts.append(f"- [{text}]({href})")
                else:
                    formatted_parts.append(f"- {link}")
            formatted_parts.append("")

        # Additional Metadata Section (if available)
        additional_metadata = self._extract_additional_metadata(metadata)
        if additional_metadata:
            formatted_parts.append("## Metadata")
            formatted_parts.append("")
            for key, value in additional_metadata.items():
                formatted_parts.append(f"- **{key}**: {value}")
            formatted_parts.append("")

        return "\n".join(formatted_parts).rstrip() + "\n"

    def _extract_title_from_content(self, content: str) -> Optional[str]:
        """Extract title from content if not in metadata.

        Args:
            content: Raw content

        Returns:
            Extracted title or None
        """
        # Look for h1 headers
        h1_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        if h1_match:
            return h1_match.group(1).strip()

        # Look for title-like patterns
        title_match = re.search(r'^(.{10,100})$', content, re.MULTILINE)
        if title_match:
            title = title_match.group(1).strip()
            # Check if it looks like a title (not too long, no special chars)
            if len(title) < 100 and not re.search(r'[.!?]{2,}', title):
                return title

        return None

    def _count_words(self, content: str) -> int:
        """Count words in content.

        Args:
            content: Text content

        Returns:
            Word count
        """
        # Remove markdown formatting and count words
        clean_content = re.sub(r'[#*_`\[\]()]', '', content)
        words = re.findall(r'\b\w+\b', clean_content)
        return len(words)

    def _parse_web_content(self, content: str) -> Dict[str, Any]:
        """Parse web content into structured sections.

        Args:
            content: Raw web content

        Returns:
            Dictionary with parsed sections
        """
        sections = {
            'main_content': '',
            'subsections': [],
            'links': []
        }

        if not content.strip():
            return sections

        # Split content by headers to identify sections
        lines = content.split('\n')
        current_section: Dict[str, Any] = {'title': '', 'content': '', 'level': 0}
        main_content_lines: List[str] = []
        subsections: List[Dict[str, str]] = []

        for line in lines:
            # Check for headers
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if header_match:
                level = len(header_match.group(1))
                title = header_match.group(2).strip()

                # Save previous section
                if current_section['content'].strip():
                    if current_section['level'] <= 2:
                        # Top-level content goes to main
                        main_content_lines.extend(current_section['content'].split('\n'))
                    else:
                        # Subsection
                        subsections.append({
                            'title': current_section['title'],
                            'content': current_section['content'].strip()
                        })

                # Start new section
                current_section = {'title': title, 'content': '', 'level': level}
            else:
                # Add line to current section
                if current_section['content']:
                    current_section['content'] += f"\n{line}"
                else:
                    current_section['content'] = line

        # Save final section
        if current_section['content'].strip():
            if current_section['level'] <= 2:
                main_content_lines.extend(current_section['content'].split('\n'))
            else:
                subsections.append({
                    'title': current_section['title'],
                    'content': current_section['content'].strip()
                })

        # Clean up main content
        sections['main_content'] = '\n'.join(main_content_lines).strip()
        sections['subsections'] = subsections

        # Extract links
        sections['links'] = self._extract_links(content)

        return sections

    def _extract_links(self, content: str) -> List[Dict[str, str]]:
        """Extract links from content.

        Args:
            content: Content to extract links from

        Returns:
            List of link dictionaries
        """
        links = []

        # Find markdown links
        link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        matches = re.findall(link_pattern, content)

        for text, href in matches:
            # Skip internal anchors and common non-content links
            if not href.startswith('#') and not any(skip in href.lower() for skip in ['javascript:', 'mailto:', 'tel:']):
                links.append({'text': text.strip(), 'href': href.strip()})

        # Limit to most important links (avoid overwhelming output)
        return links[:10]

    def _extract_additional_metadata(self, metadata: Dict[str, Any]) -> Dict[str, str]:
        """Extract additional metadata for the metadata section.

        Args:
            metadata: Full metadata dictionary

        Returns:
            Filtered metadata for display
        """
        # Define which metadata fields to include in the additional section
        additional_fields = {
            'keywords': 'Keywords',
            'description': 'Description',
            'site_name': 'Site Name',
            'article_section': 'Section',
            'article_tag': 'Tags',
            'og_type': 'Content Type',
            'twitter_card': 'Twitter Card',
            'canonical_url': 'Canonical URL'
        }

        additional_metadata = {}
        for key, display_name in additional_fields.items():
            if key in metadata and metadata[key]:
                value = metadata[key]
                if isinstance(value, list):
                    value = ', '.join(str(v) for v in value)
                additional_metadata[display_name] = str(value)

        return additional_metadata

    def clean_web_content(self, content: str) -> str:
        """Clean up web content by removing navigation and ads.

        Args:
            content: Raw web content

        Returns:
            Cleaned content
        """
        # Remove common navigation and advertisement patterns
        content = re.sub(r'(?i)(advertisement|sponsored|related articles?|you may also like)', '', content)

        # Remove excessive whitespace
        content = re.sub(r'\n{3,}', '\n\n', content)

        # Remove leading/trailing whitespace
        content = content.strip()

        return content

    def extract_web_metadata(self, url: str, content: str) -> Dict[str, Any]:
        """Extract metadata from web content and URL.

        Args:
            url: Web page URL
            content: Page content

        Returns:
            Dictionary containing web metadata
        """
        metadata = {}

        # Basic URL info
        parsed_url = urlparse(url)
        metadata['domain'] = parsed_url.netloc
        metadata['path'] = parsed_url.path

        # Extract title
        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        if title_match:
            metadata['title'] = title_match.group(1).strip()

        # Count words
        metadata['word_count'] = self._count_words(content)

        # Extract language (basic detection)
        try:
            from langdetect import detect
            detected_lang = detect(content[:1000])
            metadata['language'] = detected_lang.upper()
        except:
            metadata['language'] = 'Unknown'

        return metadata
