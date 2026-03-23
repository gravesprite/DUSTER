"""
Basic import tests for the duster package.
Verifies that all core modules can be imported in a pure CPU environment.
"""
import sys


def test_core_imports():
    from duster.core.deduplicator import Deduplicator
    from duster.core.disk_manager import DiskManager
    from duster.core.indexer import Indexer, SequenceIndexer
    print("[OK] duster.core imports")


def test_segmentation_import():
    from duster.segmentation.segmentator import Segmentator
    print("[OK] duster.segmentation imports")


def test_compression_import():
    from duster.compression.octree import (
        generate_initial_octree,
        add_point_cloud_to_shared_octree,
        save_octree_to_file,
        load_octree_from_file,
        regenerate_point_cloud,
    )
    print("[OK] duster.compression imports")


def test_query_import():
    from duster.query.loader import load_queries
    print("[OK] duster.query imports")


def test_utils_import():
    from duster.utils.logger import create_logger
    print("[OK] duster.utils imports")


def test_top_level_import():
    from duster import Deduplicator, DiskManager, Indexer, SequenceIndexer, Segmentator, create_logger
    print("[OK] duster top-level imports")


def test_version():
    import duster
    assert hasattr(duster, '__version__')
    print(f"[OK] duster version = {duster.__version__}")


if __name__ == "__main__":
    tests = [
        test_core_imports,
        test_segmentation_import,
        test_compression_import,
        test_query_import,
        test_utils_import,
        test_top_level_import,
        test_version,
    ]
    failed = 0
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"[FAIL] {test.__name__}: {e}")
            failed += 1

    if failed:
        print(f"\n{failed} test(s) failed")
        sys.exit(1)
    else:
        print("\nAll import tests passed!")
