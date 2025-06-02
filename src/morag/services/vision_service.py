"""Vision service for image analysis using Google Gemini Vision."""

import asyncio
from typing import Dict, Any, List, Optional
from pathlib import Path
import structlog

import google.generativeai as genai
from PIL import Image

from morag.core.config import settings
from morag.core.exceptions import ExternalServiceError

logger = structlog.get_logger()

class VisionService:
    """Service for vision-based image analysis using Gemini Vision."""
    
    def __init__(self):
        if settings.gemini_api_key:
            genai.configure(api_key=settings.gemini_api_key)
        else:
            logger.warning("Gemini API key not configured - vision features will be disabled")
    
    async def generate_caption(self, image_path: Path, custom_prompt: Optional[str] = None) -> str:
        """Generate descriptive caption for an image."""
        try:
            if not settings.gemini_api_key:
                raise ExternalServiceError("Gemini API key not configured", "gemini")
            
            logger.debug("Generating image caption", image_path=str(image_path))
            
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != "RGB":
                    img = img.convert("RGB")
                
                model = genai.GenerativeModel('gemini-pro-vision')
                
                prompt = custom_prompt or """
                Analyze this image and provide a detailed, descriptive caption.
                Focus on:
                - Main subjects and objects in the image
                - Setting, environment, and background
                - Actions, activities, or interactions
                - Visual style, composition, and mood
                - Any visible text or signage
                - Colors, lighting, and atmosphere
                
                Provide a comprehensive but concise description in 2-3 sentences.
                """
                
                response = await asyncio.to_thread(
                    model.generate_content,
                    [prompt.strip(), img]
                )
                
                caption = response.text.strip() if response.text else ""
                
                logger.debug("Caption generated successfully",
                           caption_length=len(caption))
                
                return caption
                
        except Exception as e:
            logger.error("Caption generation failed",
                        image_path=str(image_path),
                        error=str(e))
            raise ExternalServiceError(f"Caption generation failed: {str(e)}", "gemini")
    
    async def analyze_image_content(self, image_path: Path) -> Dict[str, Any]:
        """Analyze image content for detailed information."""
        try:
            if not settings.gemini_api_key:
                raise ExternalServiceError("Gemini API key not configured", "gemini")
            
            logger.debug("Analyzing image content", image_path=str(image_path))
            
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != "RGB":
                    img = img.convert("RGB")
                
                model = genai.GenerativeModel('gemini-pro-vision')
                
                prompt = """
                Analyze this image in detail and provide structured information about:
                
                1. OBJECTS: List the main objects, people, animals, or items visible
                2. SETTING: Describe the location, environment, or scene type
                3. ACTIVITIES: What actions or activities are taking place
                4. STYLE: Visual style, artistic elements, photography type
                5. TEXT: Any visible text, signs, or written content
                6. EMOTIONS: Mood, atmosphere, or emotional content
                7. TECHNICAL: Image quality, lighting, composition notes
                
                Format your response as clear, structured information for each category.
                If a category doesn't apply, indicate "None" or "Not applicable".
                """
                
                response = await asyncio.to_thread(
                    model.generate_content,
                    [prompt.strip(), img]
                )
                
                analysis = response.text.strip() if response.text else ""
                
                # Parse the structured response into categories
                content_analysis = self._parse_content_analysis(analysis)
                
                logger.debug("Image content analysis completed",
                           analysis_length=len(analysis))
                
                return content_analysis
                
        except Exception as e:
            logger.error("Image content analysis failed",
                        image_path=str(image_path),
                        error=str(e))
            raise ExternalServiceError(f"Image content analysis failed: {str(e)}", "gemini")
    
    async def classify_image_type(self, image_path: Path) -> str:
        """Classify the type/category of the image."""
        try:
            if not settings.gemini_api_key:
                raise ExternalServiceError("Gemini API key not configured", "gemini")
            
            logger.debug("Classifying image type", image_path=str(image_path))
            
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != "RGB":
                    img = img.convert("RGB")
                
                model = genai.GenerativeModel('gemini-pro-vision')
                
                prompt = """
                Classify this image into one of the following categories:
                
                - photograph: Real-world photography (people, places, objects)
                - artwork: Paintings, drawings, digital art, illustrations
                - document: Text documents, forms, papers, books
                - screenshot: Computer/phone screenshots, UI elements
                - diagram: Charts, graphs, technical diagrams, flowcharts
                - meme: Internet memes, humorous images with text
                - logo: Company logos, brand symbols, icons
                - nature: Landscapes, animals, plants, natural scenes
                - food: Food items, meals, cooking, restaurants
                - product: Product photos, advertisements, catalogs
                - other: Anything that doesn't fit the above categories
                
                Respond with just the category name (one word) that best fits this image.
                """
                
                response = await asyncio.to_thread(
                    model.generate_content,
                    [prompt.strip(), img]
                )
                
                classification = response.text.strip().lower() if response.text else "other"
                
                # Validate classification
                valid_types = {
                    "photograph", "artwork", "document", "screenshot", "diagram",
                    "meme", "logo", "nature", "food", "product", "other"
                }
                
                if classification not in valid_types:
                    classification = "other"
                
                logger.debug("Image classified",
                           classification=classification)
                
                return classification
                
        except Exception as e:
            logger.error("Image classification failed",
                        image_path=str(image_path),
                        error=str(e))
            return "other"
    
    async def extract_text_content(self, image_path: Path) -> str:
        """Extract and interpret text content from image using vision model."""
        try:
            if not settings.gemini_api_key:
                raise ExternalServiceError("Gemini API key not configured", "gemini")
            
            logger.debug("Extracting text content with vision model", image_path=str(image_path))
            
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != "RGB":
                    img = img.convert("RGB")
                
                model = genai.GenerativeModel('gemini-pro-vision')
                
                prompt = """
                Extract and transcribe ALL visible text from this image.
                Include:
                - All readable text, signs, labels, captions
                - Text in any language or script
                - Numbers, dates, and symbols
                - Handwritten text if legible
                
                Organize the text logically (top to bottom, left to right).
                If there's no visible text, respond with "No text detected".
                """
                
                response = await asyncio.to_thread(
                    model.generate_content,
                    [prompt.strip(), img]
                )
                
                extracted_text = response.text.strip() if response.text else ""
                
                if extracted_text.lower() == "no text detected":
                    extracted_text = ""
                
                logger.debug("Text extraction completed",
                           text_length=len(extracted_text))
                
                return extracted_text
                
        except Exception as e:
            logger.error("Vision-based text extraction failed",
                        image_path=str(image_path),
                        error=str(e))
            return ""
    
    def _parse_content_analysis(self, analysis_text: str) -> Dict[str, Any]:
        """Parse structured content analysis response."""
        try:
            content = {
                "objects": "",
                "setting": "",
                "activities": "",
                "style": "",
                "text": "",
                "emotions": "",
                "technical": "",
                "raw_analysis": analysis_text
            }
            
            # Simple parsing - look for numbered sections or keywords
            lines = analysis_text.split('\n')
            current_section = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check for section headers
                line_lower = line.lower()
                if any(keyword in line_lower for keyword in ['objects:', '1.', 'objects']):
                    current_section = "objects"
                elif any(keyword in line_lower for keyword in ['setting:', '2.', 'setting']):
                    current_section = "setting"
                elif any(keyword in line_lower for keyword in ['activities:', '3.', 'activities']):
                    current_section = "activities"
                elif any(keyword in line_lower for keyword in ['style:', '4.', 'style']):
                    current_section = "style"
                elif any(keyword in line_lower for keyword in ['text:', '5.', 'text']):
                    current_section = "text"
                elif any(keyword in line_lower for keyword in ['emotions:', '6.', 'emotions']):
                    current_section = "emotions"
                elif any(keyword in line_lower for keyword in ['technical:', '7.', 'technical']):
                    current_section = "technical"
                elif current_section and line:
                    # Add content to current section
                    if content[current_section]:
                        content[current_section] += " " + line
                    else:
                        content[current_section] = line
            
            return content
            
        except Exception as e:
            logger.warning("Failed to parse content analysis", error=str(e))
            return {
                "objects": "",
                "setting": "",
                "activities": "",
                "style": "",
                "text": "",
                "emotions": "",
                "technical": "",
                "raw_analysis": analysis_text
            }

# Global instance
vision_service = VisionService()
