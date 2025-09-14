#!/usr/bin/env python3
"""Command line interface for MoRAG Remote Converter."""

import argparse
import sys
import os
from pathlib import Path
import structlog
from dotenv import load_dotenv

from .config import RemoteConverterConfig, setup_logging
from remote_converter import RemoteConverter

logger = structlog.get_logger(__name__)


def test_connection(config: dict) -> bool:
    """Test connection to MoRAG API."""
    try:
        converter = RemoteConverter(config)
        return converter.test_connection()
    except Exception as e:
        logger.error("Failed to test connection", error=str(e))
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="MoRAG Remote Converter")
    parser.add_argument('--config', '-c', help='Configuration file path')
    parser.add_argument('--worker-id', help='Unique worker identifier')
    parser.add_argument('--api-url', help='MoRAG API base URL')
    parser.add_argument('--api-key', help='API authentication key')
    parser.add_argument('--content-types', help='Comma-separated list of content types to process')
    parser.add_argument('--poll-interval', type=int, help='Polling interval in seconds')
    parser.add_argument('--max-jobs', type=int, help='Maximum concurrent jobs')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Log level')
    parser.add_argument('--temp-dir', help='Temporary directory for file processing')
    parser.add_argument('--create-config', action='store_true', help='Create sample configuration file')
    parser.add_argument('--test-connection', action='store_true', help='Test API connection and exit')
    parser.add_argument('--show-config', action='store_true', help='Show current configuration and exit')
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Create sample config if requested
    if args.create_config:
        config_manager = RemoteConverterConfig()
        if config_manager.create_sample_config():
            print("Sample configuration created: remote_converter_config.yaml.example")
            print("\nNext steps:")
            print("1. Copy the example file: cp remote_converter_config.yaml.example remote_converter_config.yaml")
            print("2. Edit the configuration file with your settings")
            print("3. Test connection: python cli.py --test-connection")
            print("4. Start the converter: python cli.py")
            return 0
        else:
            print("Failed to create sample configuration")
            return 1
    
    # Load configuration
    config_manager = RemoteConverterConfig(args.config)
    
    # Override with command line arguments
    if args.worker_id:
        config_manager.config['worker_id'] = args.worker_id
    if args.api_url:
        config_manager.config['api_base_url'] = args.api_url
    if args.api_key:
        config_manager.config['api_key'] = args.api_key
    if args.content_types:
        config_manager.config['content_types'] = args.content_types.split(',')
    if args.poll_interval:
        config_manager.config['poll_interval'] = args.poll_interval
    if args.max_jobs:
        config_manager.config['max_concurrent_jobs'] = args.max_jobs
    if args.log_level:
        config_manager.config['log_level'] = args.log_level
    if args.temp_dir:
        config_manager.config['temp_dir'] = args.temp_dir
    
    # Set up logging
    log_level = config_manager.config.get('log_level', 'INFO')
    setup_logging(log_level)
    
    # Show configuration if requested
    if args.show_config:
        config_manager.print_config()
        return 0
    
    # Validate configuration
    if not config_manager.validate_config():
        logger.error("Configuration validation failed")
        print("\nConfiguration errors detected. Please check your configuration.")
        print("Use --create-config to create a sample configuration file.")
        return 1
    
    # Test connection if requested
    if args.test_connection:
        print("Testing connection to MoRAG API...")
        if test_connection(config_manager.get_config()):
            print("✅ Connection test successful!")
            return 0
        else:
            print("❌ Connection test failed!")
            return 1
    
    # Print startup information
    print("🚀 MoRAG Remote Converter")
    print("=" * 50)
    print(f"Worker ID: {config_manager.config['worker_id']}")
    print(f"API URL: {config_manager.config['api_base_url']}")
    print(f"Content Types: {', '.join(config_manager.config['content_types'])}")
    print(f"Poll Interval: {config_manager.config['poll_interval']}s")
    print(f"Max Concurrent Jobs: {config_manager.config['max_concurrent_jobs']}")
    print(f"Temp Directory: {config_manager.config['temp_dir']}")
    print("=" * 50)
    
    # Test connection before starting
    print("Testing API connection...")
    if not test_connection(config_manager.get_config()):
        print("❌ Failed to connect to MoRAG API. Please check your configuration.")
        return 1
    
    print("✅ API connection successful!")
    print("Starting remote converter...")
    
    # Create and start remote converter
    try:
        converter = RemoteConverter(config_manager.get_config())
        converter.start()
        return 0
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down")
        print("\n🛑 Remote converter stopped by user")
        return 0
    except Exception as e:
        logger.error("Remote converter failed", error=str(e))
        print(f"\n💥 Remote converter failed: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
