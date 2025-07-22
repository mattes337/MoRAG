#!/usr/bin/env python3
"""Comprehensive end-to-end testing for markitdown integration."""

import asyncio
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple
from morag_document.processor import DocumentProcessor

class ComprehensiveMarkitdownTest:
    """Comprehensive test suite for markitdown integration."""
    
    def __init__(self):
        self.processor = DocumentProcessor()
        self.test_results = []
        
    async def test_format_support(self) -> Dict[str, bool]:
        """Test format support across all converters."""
        print("🔍 Testing format support...")
        
        # Test formats by category
        test_formats = {
            'Document Formats': ['pdf', 'docx', 'doc', 'xlsx', 'xls', 'pptx', 'ppt'],
            'Text Formats': ['txt', 'md', 'html', 'htm'],
            'Image Formats': ['jpg', 'png', 'gif', 'bmp', 'tiff', 'webp', 'svg'],
            'Audio Formats': ['mp3', 'wav', 'm4a', 'flac', 'aac', 'ogg'],
            'Video Formats': ['mp4', 'avi', 'mov', 'mkv', 'webm', 'flv', 'wmv'],
            'Archive Formats': ['zip', 'epub', 'tar', 'gz', 'rar', '7z']
        }
        
        results = {}
        total_formats = 0
        supported_formats = 0
        
        for category, formats in test_formats.items():
            print(f"\n  📁 {category}:")
            for fmt in formats:
                supported = await self.processor.supports_format(fmt)
                converter = self.processor.converters.get(fmt)
                converter_name = converter.__class__.__name__ if converter else 'None'
                
                status = "✅" if supported else "❌"
                print(f"    {status} {fmt}: {supported} ({converter_name})")
                
                results[fmt] = supported
                total_formats += 1
                if supported:
                    supported_formats += 1
        
        print(f"\n📊 Format Support Summary: {supported_formats}/{total_formats} formats supported")
        return results
    
    async def test_text_conversion_quality(self) -> Dict[str, float]:
        """Test text conversion quality with sample content."""
        print("\n🎯 Testing text conversion quality...")
        
        # Create test files with different content types
        test_cases = {
            'md': {
                'content': '# Test Document\n\nThis is a **bold** text with *italic* and `code`.\n\n## Section 2\n\n- Item 1\n- Item 2\n\n| Column 1 | Column 2 |\n|----------|----------|\n| Data 1   | Data 2   |',
                'expected_elements': ['# Test Document', '**bold**', '*italic*', '`code`', '## Section 2', '- Item', '| Column']
            },
            'html': {
                'content': '<html><body><h1>Test HTML</h1><p>This is <strong>bold</strong> and <em>italic</em> text.</p><ul><li>Item 1</li><li>Item 2</li></ul></body></html>',
                'expected_elements': ['Test HTML', 'bold', 'italic', 'Item 1', 'Item 2']
            },
            'txt': {
                'content': 'Simple text file\nWith multiple lines\nAnd some content for testing.',
                'expected_elements': ['Simple text file', 'multiple lines', 'testing']
            }
        }
        
        quality_scores = {}
        
        for file_type, test_data in test_cases.items():
            print(f"  📝 Testing {file_type} conversion...")
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{file_type}', delete=False) as f:
                f.write(test_data['content'])
                test_file = f.name
            
            try:
                # Process the file
                start_time = time.time()
                result = await self.processor.process_file(test_file)
                processing_time = time.time() - start_time
                
                if result.success and result.document:
                    # Check quality
                    quality_score = result.metadata.get('quality_score', 0.0)
                    content_length = len(result.document.raw_text)
                    chunk_count = len(result.document.chunks)
                    
                    # Check for expected elements
                    elements_found = sum(1 for element in test_data['expected_elements'] 
                                       if element.lower() in result.document.raw_text.lower())
                    element_coverage = elements_found / len(test_data['expected_elements'])
                    
                    print(f"    ✅ {file_type}: Quality={quality_score:.2f}, Time={processing_time:.3f}s, "
                          f"Length={content_length}, Chunks={chunk_count}, Coverage={element_coverage:.2f}")
                    
                    quality_scores[file_type] = {
                        'quality_score': quality_score,
                        'processing_time': processing_time,
                        'content_length': content_length,
                        'chunk_count': chunk_count,
                        'element_coverage': element_coverage
                    }
                else:
                    print(f"    ❌ {file_type}: Conversion failed")
                    quality_scores[file_type] = {'quality_score': 0.0, 'error': 'Conversion failed'}
                    
            finally:
                # Clean up
                Path(test_file).unlink()
        
        return quality_scores
    
    async def test_performance_benchmarks(self) -> Dict[str, float]:
        """Test performance benchmarks for different file sizes."""
        print("\n⚡ Testing performance benchmarks...")
        
        # Create test files of different sizes
        test_sizes = {
            'small': 'Small test content for performance testing.',
            'medium': 'Medium test content. ' * 100,  # ~2KB
            'large': 'Large test content for performance testing. ' * 1000,  # ~40KB
        }
        
        performance_results = {}
        
        for size_name, content in test_sizes.items():
            print(f"  📏 Testing {size_name} file ({len(content)} chars)...")
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(content)
                test_file = f.name
            
            try:
                # Run multiple iterations for average
                times = []
                for i in range(3):
                    start_time = time.time()
                    result = await self.processor.process_file(test_file)
                    processing_time = time.time() - start_time
                    times.append(processing_time)
                
                avg_time = sum(times) / len(times)
                chars_per_second = len(content) / avg_time if avg_time > 0 else 0
                
                print(f"    ⏱️  {size_name}: Avg={avg_time:.3f}s, Speed={chars_per_second:.0f} chars/s")
                performance_results[size_name] = {
                    'avg_time': avg_time,
                    'chars_per_second': chars_per_second,
                    'content_size': len(content)
                }
                
            finally:
                # Clean up
                Path(test_file).unlink()
        
        return performance_results
    
    async def test_error_handling(self) -> Dict[str, bool]:
        """Test error handling for various scenarios."""
        print("\n🛡️  Testing error handling...")
        
        error_tests = {}
        
        # Test 1: Non-existent file
        try:
            result = await self.processor.process_file("non_existent_file.txt")
            error_tests['non_existent_file'] = False  # Should have failed
        except Exception:
            error_tests['non_existent_file'] = True  # Correctly handled error
            print("    ✅ Non-existent file: Error handled correctly")
        
        # Test 2: Empty file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write('')  # Empty file
            empty_file = f.name
        
        try:
            result = await self.processor.process_file(empty_file)
            # Empty file should be processed successfully but with minimal content
            error_tests['empty_file'] = result.success
            print(f"    ✅ Empty file: Processed successfully ({result.success})")
        except Exception as e:
            error_tests['empty_file'] = False
            print(f"    ❌ Empty file: Unexpected error - {e}")
        finally:
            Path(empty_file).unlink()
        
        # Test 3: Unsupported format
        try:
            result = await self.processor.process_file("test.unsupported")
            error_tests['unsupported_format'] = False  # Should have failed
        except Exception:
            error_tests['unsupported_format'] = True  # Correctly handled error
            print("    ✅ Unsupported format: Error handled correctly")
        
        return error_tests
    
    async def run_comprehensive_test(self):
        """Run all comprehensive tests."""
        print("🚀 Starting Comprehensive Markitdown Integration Test")
        print("=" * 60)
        
        # Test 1: Format Support
        format_results = await self.test_format_support()
        
        # Test 2: Quality Testing
        quality_results = await self.test_text_conversion_quality()
        
        # Test 3: Performance Testing
        performance_results = await self.test_performance_benchmarks()
        
        # Test 4: Error Handling
        error_results = await self.test_error_handling()
        
        # Summary
        print("\n" + "=" * 60)
        print("📋 COMPREHENSIVE TEST SUMMARY")
        print("=" * 60)
        
        # Format support summary
        total_formats = len(format_results)
        supported_formats = sum(format_results.values())
        print(f"📁 Format Support: {supported_formats}/{total_formats} ({supported_formats/total_formats*100:.1f}%)")
        
        # Quality summary
        avg_quality = sum(r.get('quality_score', 0) for r in quality_results.values()) / len(quality_results)
        print(f"🎯 Average Quality Score: {avg_quality:.2f}")
        
        # Performance summary
        total_chars = sum(r['content_size'] for r in performance_results.values())
        total_time = sum(r['avg_time'] for r in performance_results.values())
        overall_speed = total_chars / total_time if total_time > 0 else 0
        print(f"⚡ Overall Processing Speed: {overall_speed:.0f} chars/second")
        
        # Error handling summary
        error_success_rate = sum(error_results.values()) / len(error_results) * 100
        print(f"🛡️  Error Handling: {error_success_rate:.1f}% tests passed")
        
        # Overall assessment
        print("\n🏆 OVERALL ASSESSMENT:")
        if supported_formats >= 40 and avg_quality >= 0.8 and error_success_rate >= 80:
            print("    ✅ EXCELLENT - System ready for production")
        elif supported_formats >= 30 and avg_quality >= 0.7 and error_success_rate >= 70:
            print("    ⚠️  GOOD - Minor improvements recommended")
        else:
            print("    ❌ NEEDS IMPROVEMENT - Address issues before deployment")
        
        return {
            'format_support': format_results,
            'quality_results': quality_results,
            'performance_results': performance_results,
            'error_handling': error_results
        }

async def main():
    """Run comprehensive tests."""
    test_suite = ComprehensiveMarkitdownTest()
    results = await test_suite.run_comprehensive_test()
    return results

if __name__ == '__main__':
    asyncio.run(main())
