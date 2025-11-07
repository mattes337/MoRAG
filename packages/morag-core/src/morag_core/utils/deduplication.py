"""Generic deduplication utilities for MoRAG."""

from typing import Any, Callable, Dict, List, Optional, TypeVar

T = TypeVar("T")


class Deduplicator:
    """Generic deduplication utilities."""

    @staticmethod
    def deduplicate_by_key(
        items: List[T],
        key_func: Callable[[T], Any],
        merge_func: Optional[Callable[[T, T], T]] = None,
    ) -> List[T]:
        """
        Deduplicate items by a key function.

        Args:
            items: List of items to deduplicate
            key_func: Function to extract key for comparison
            merge_func: Optional function to merge duplicate items.
                       If provided, duplicates will be merged using this function.
                       If not provided, first occurrence is kept.

        Returns:
            List of deduplicated items
        """
        seen = {}
        result = []

        for item in items:
            key = key_func(item)
            if key not in seen:
                seen[key] = item
                result.append(item)
            elif merge_func:
                # Merge with existing item
                seen[key] = merge_func(seen[key], item)

        # If merge_func was used, return the merged values
        # Otherwise return the result list (preserves order)
        return list(seen.values()) if merge_func else result

    @staticmethod
    def deduplicate_by_multiple_keys(
        items: List[T],
        key_funcs: List[Callable[[T], Any]],
        merge_func: Optional[Callable[[T, T], T]] = None,
    ) -> List[T]:
        """
        Deduplicate items by multiple key functions (composite key).

        Args:
            items: List of items to deduplicate
            key_funcs: List of functions to extract keys for comparison
            merge_func: Optional function to merge duplicate items

        Returns:
            List of deduplicated items
        """

        def composite_key(item: T) -> tuple:
            return tuple(func(item) for func in key_funcs)

        return Deduplicator.deduplicate_by_key(items, composite_key, merge_func)

    @staticmethod
    def deduplicate_by_attribute(
        items: List[T], attribute: str, merge_func: Optional[Callable[[T, T], T]] = None
    ) -> List[T]:
        """
        Deduplicate items by a specific attribute.

        Args:
            items: List of items to deduplicate
            attribute: Attribute name to use for comparison
            merge_func: Optional function to merge duplicate items

        Returns:
            List of deduplicated items
        """

        def attr_key(item: T) -> Any:
            return getattr(item, attribute)

        return Deduplicator.deduplicate_by_key(items, attr_key, merge_func)

    @staticmethod
    def deduplicate_dicts_by_key(
        items: List[Dict[str, Any]],
        key: str,
        merge_func: Optional[
            Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]
        ] = None,
    ) -> List[Dict[str, Any]]:
        """
        Deduplicate dictionaries by a specific key.

        Args:
            items: List of dictionaries to deduplicate
            key: Dictionary key to use for comparison
            merge_func: Optional function to merge duplicate dictionaries

        Returns:
            List of deduplicated dictionaries
        """

        def dict_key(item: Dict[str, Any]) -> Any:
            return item.get(key)

        return Deduplicator.deduplicate_by_key(items, dict_key, merge_func)

    @staticmethod
    def merge_first_wins(item1: T, item2: T) -> T:
        """Merge function that keeps the first item."""
        return item1

    @staticmethod
    def merge_last_wins(item1: T, item2: T) -> T:
        """Merge function that keeps the last item."""
        return item2

    @staticmethod
    def merge_dict_update(
        dict1: Dict[str, Any], dict2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge function that updates first dict with values from second."""
        result = dict1.copy()
        result.update(dict2)
        return result

    @staticmethod
    def merge_list_extend(list1: List[Any], list2: List[Any]) -> List[Any]:
        """Merge function that extends first list with second."""
        result = list1.copy()
        result.extend(list2)
        return result
