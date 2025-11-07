"""Example usage of the centralized configuration manager.

This file demonstrates how to use the configuration manager in different scenarios.
"""

from morag_core.config import (
    ConfigOverride,
    get_config_manager,
    get_config_value,
    get_package_config,
    override_config,
)


def basic_usage_example():
    """Basic configuration usage example."""

    # Get configuration for a specific package
    audio_config = get_package_config("morag-audio")
    print(f"Audio config: {audio_config}")

    # Get a specific configuration value
    model = get_config_value("morag-audio", "model", "default-model")
    print(f"Audio model: {model}")

    # Get nested configuration value using dot notation
    llm_model = get_config_value("morag-audio", "llm.model", "default-llm")
    print(f"LLM model: {llm_model}")


def override_usage_example():
    """Configuration override usage example."""

    # Override configuration globally
    override_config("morag-audio", {"model": "custom-whisper", "max_duration": 1800})

    # Get the overridden configuration
    config = get_package_config("morag-audio")
    print(f"Overridden config: {config}")


def context_manager_example():
    """Context manager usage for testing."""

    # Get original configuration
    original_model = get_config_value("morag-audio", "model")
    print(f"Original model: {original_model}")

    # Use context manager for temporary override
    with ConfigOverride("morag-audio", {"model": "test-model"}):
        test_model = get_config_value("morag-audio", "model")
        print(f"Test model: {test_model}")

        # Do testing with override active
        assert test_model == "test-model"

    # Configuration is automatically restored
    restored_model = get_config_value("morag-audio", "model")
    print(f"Restored model: {restored_model}")
    assert restored_model == original_model


def advanced_manager_usage():
    """Advanced configuration manager usage."""

    # Get the global configuration manager instance
    config_mgr = get_config_manager()

    # Get available packages
    packages = config_mgr.get_available_packages()
    print(f"Available packages: {packages}")

    # Set specific configuration values
    config_mgr.set_config_value("morag-audio", "processing.batch_size", 32)
    batch_size = config_mgr.get_config_value("morag-audio", "processing.batch_size")
    print(f"Batch size: {batch_size}")

    # Reload configuration from sources
    config_mgr.reload_config("morag-audio")

    # Clear all overrides
    config_mgr.clear_overrides()


if __name__ == "__main__":
    print("=== Basic Usage Example ===")
    basic_usage_example()

    print("\n=== Override Usage Example ===")
    override_usage_example()

    print("\n=== Context Manager Example ===")
    context_manager_example()

    print("\n=== Advanced Manager Usage ===")
    advanced_manager_usage()
