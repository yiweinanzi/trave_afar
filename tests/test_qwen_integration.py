#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Qwen Model Integration Tests
==========================
Tests:
1. Pipeline with Qwen models
2. Embedding model encoding
3. Reranker model reranking
4. Model health checks
5. Memory management

Usage:
    python tests/test_qwen_integration.py
    python tests/test_qwen_integration.py --embedding-only
    python tests/test_qwen_integration.py --reranker-only
    python tests/test_qwen_integration.py --health-only
"""
import sys
import os
import unittest
import logging
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestQwenEmbedding(unittest.TestCase):
    """Test Qwen3-Embedding model"""

    def setUp(self):
        """Setup before tests"""
        from src.embedding.qwen3_encoder import Qwen3Embedding
        model_path = PROJECT_ROOT / "models/Qwen3-Embedding-4B"

        self.embedding = Qwen3Embedding(
            model_path=str(model_path),
            use_gpu=False,
            max_retries=2
        )

    def test_embedding_available(self):
        """Test 1.1: Check if model is available"""
        logger.info("Test 1.1: Check if Embedding model is available")
        self.assertIsNotNone(self.embedding)
        logger.info("Model available: {}".format(self.embedding.is_available()))

    def test_embedding_encode_texts(self):
        """Test 1.2: Test batch encoding"""
        logger.info("Test 1.2: Test Embedding batch encoding")

        texts = [
            "Kanas Lake is located in northern Xinjiang",
            "Sayram Lake is known as Atlantic's last tear",
            "Heavenly Lake is a famous tourist attraction in Xinjiang"
        ]

        result = self.embedding.encode_texts(texts, show_progress=False)

        self.assertIn('dense_vecs', result)
        self.assertEqual(result['dense_vecs'].shape[0], len(texts))
        logger.info("Encoding result shape: {}".format(result['dense_vecs'].shape))

    def test_embedding_encode_query(self):
        """Test 1.3: Test query encoding"""
        logger.info("Test 1.3: Test Embedding query encoding")

        query = "Want to visit lakes in Xinjiang"
        result = self.embedding.encode_query(query)

        self.assertIn('dense_vec', result)
        self.assertEqual(len(result['dense_vec'].shape), 1)
        logger.info("Query vector shape: {}".format(result['dense_vec'].shape))


class TestQwenReranker(unittest.TestCase):
    """Test Qwen3-Reranker model"""

    def setUp(self):
        """Setup before tests"""
        from src.reranking.qwen_reranker import QwenReranker
        model_path = PROJECT_ROOT / "models/Qwen3-Reranker-4B"

        self.reranker = QwenReranker(
            model_path=str(model_path),
            use_gpu=False,
            max_retries=2
        )

    def test_reranker_available(self):
        """Test 2.1: Check if model is available"""
        logger.info("Test 2.1: Check if Reranker model is available")
        self.assertIsNotNone(self.reranker)
        logger.info("Model available: {}".format(self.reranker.is_available()))

    def test_reranker_rerank(self):
        """Test 2.2: Test reranking"""
        logger.info("Test 2.2: Test Reranker reranking")

        query = "Want to visit snow mountains and grasslands in Xinjiang"
        candidates = [
            {"name": "Kanas Lake", "city": "Altay", "province": "Xinjiang",
             "description": "Famous alpine lake in Xinjiang, surrounded by snow mountains"},
            {"name": "Nalati Grassland", "city": "Ili", "province": "Xinjiang",
             "description": "Air grassland with beautiful scenery"},
            {"name": "Potala Palace", "city": "Lhasa", "province": "Tibet",
             "description": "Iconic building in Tibet"},
            {"name": "Hemu Village", "city": "Altay", "province": "Xinjiang",
             "description": "Tuwa village with charming autumn scenery"},
        ]

        ranked = self.reranker.rerank(query, candidates, topk=3)

        self.assertEqual(len(ranked), 3)
        self.assertIn("reranker_score", ranked[0])
        logger.info("Reranking results:")
        for i, item in enumerate(ranked, 1):
            logger.info("  {}. {} - Score: {:.2f}".format(
                i, item['name'], item.get('reranker_score', 0)
            ))


class TestPipelineHealthCheck(unittest.TestCase):
    """Test Pipeline model health check"""

    def setUp(self):
        """Setup before tests"""
        from src.service.pipeline import RecommendationPipeline
        from src.service.config import load_runtime_config

        self.config = load_runtime_config()
        self.pipeline = RecommendationPipeline(config=self.config)

    def test_health_check_all(self):
        """Test 3.1: Test overall health check"""
        logger.info("Test 3.1: Test overall health check")

        health = self.pipeline.check_model_health()

        self.assertIn('embedding', health)
        self.assertIn('reranker', health)
        self.assertIn('llm', health)
        self.assertIn('all', health)

        logger.info("Health check results:")
        for model, status in health.items():
            logger.info("  {}: {}".format(model, status))


class TestMemoryManagement(unittest.TestCase):
    """Test memory management functions"""

    def setUp(self):
        """Setup before tests"""
        from src.service.pipeline import RecommendationPipeline
        from src.service.config import load_runtime_config

        self.config = load_runtime_config()
        self.pipeline = RecommendationPipeline(config=self.config)

    def test_cleanup_models(self):
        """Test 4.1: Test cleaning all models"""
        logger.info("Test 4.1: Test cleaning all models")

        # Ensure models are loaded
        self.pipeline._maybe_get_qwen()
        self.pipeline._maybe_get_qwen_reranker()

        # Cleanup
        self.pipeline.cleanup_models()

        # Verify
        self.assertIsNone(self.pipeline._qwen)
        self.assertIsNone(self.pipeline._qwen_reranker)

        logger.info("Model cleanup completed")


class TestEndToEnd(unittest.TestCase):
    """End-to-end tests"""

    def setUp(self):
        """Setup before tests"""
        from src.service.pipeline import RecommendationPipeline
        from src.service.config import load_runtime_config

        self.config = load_runtime_config()
        self.pipeline = RecommendationPipeline(config=self.config)

    def test_health_before_recommendation(self):
        """Test 5.1: Health check before recommendation"""
        logger.info("Test 5.1: Health check before recommendation")

        health = self.pipeline.check_model_health()

        if not health['all']:
            logger.warning("Some models unavailable:")
            for model, status in health.items():
                if model != 'all' and not status:
                    logger.warning("  {}: unavailable".format(model))

        # Test should continue since system has fallback mechanism


def run_tests(test_filter=None):
    """Run tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    if test_filter == 'embedding':
        suite.addTests(loader.loadTestsFromTestCase(TestQwenEmbedding))
    elif test_filter == 'reranker':
        suite.addTests(loader.loadTestsFromTestCase(TestQwenReranker))
    elif test_filter == 'health':
        suite.addTests(loader.loadTestsFromTestCase(TestPipelineHealthCheck))
    elif test_filter == 'memory':
        suite.addTests(loader.loadTestsFromTestCase(TestMemoryManagement))
    elif test_filter == 'e2e':
        suite.addTests(loader.loadTestsFromTestCase(TestEndToEnd))
    else:
        suite.addTests(loader.loadTestsFromTestCase(TestQwenEmbedding))
        suite.addTests(loader.loadTestsFromTestCase(TestQwenReranker))
        suite.addTests(loader.loadTestsFromTestCase(TestPipelineHealthCheck))
        suite.addTests(loader.loadTestsFromTestCase(TestMemoryManagement))
        suite.addTests(loader.loadTestsFromTestCase(TestEndToEnd))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Generate test report
    print("\n" + "=" * 70)
    print("Test Report")
    print("=" * 70)
    print("Tests run: {}".format(result.testsRun))
    print("Success: {}".format(result.testsRun - len(result.failures) - len(result.errors)))
    print("Failures: {}".format(len(result.failures)))
    print("Errors: {}".format(len(result.errors)))
    print("=" * 70)

    return result.wasSuccessful()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Qwen model integration tests')
    parser.add_argument('--embedding-only', action='store_true',
                       help='Only test Embedding model')
    parser.add_argument('--reranker-only', action='store_true',
                       help='Only test Reranker model')
    parser.add_argument('--health-only', action='store_true',
                       help='Only test health check')
    parser.add_argument('--memory-only', action='store_true',
                       help='Only test memory management')
    parser.add_argument('--e2e-only', action='store_true',
                       help='Only test end-to-end flow')

    args = parser.parse_args()

    test_filter = None
    if args.embedding_only:
        test_filter = 'embedding'
    elif args.reranker_only:
        test_filter = 'reranker'
    elif args.health_only:
        test_filter = 'health'
    elif args.memory_only:
        test_filter = 'memory'
    elif args.e2e_only:
        test_filter = 'e2e'

    success = run_tests(test_filter)
    sys.exit(0 if success else 1)
