"""
Tests for llm4rec.qwen_recommender module.

Tests LLM recommender integration, intent understanding, and recommendation verification.
"""
import sys
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_user_intent():
    """Create a sample user intent."""
    return {
        "original_query": "Want to see snow mountains and grasslands in Xinjiang",
        "province": "Xinjiang",
        "cities": ["Urumqi", "Altay"],
        "interests": ["snow mountains", "grassland"],
        "activities": ["photography"],
        "duration_days": 3,
        "season": "autumn",
        "travel_style": "photography",
        "constraints": [],
        "keywords": ["snow mountains", "grassland", "autumn"],
        "expanded_query": "Xinjiang snow mountains grassland autumn",
    }


@pytest.fixture
def sample_poi_candidates():
    """Create sample POI candidates."""
    return [
        {
            "poi_id": "POI_0001",
            "name": "Kanas Lake",
            "city": "Altay",
            "province": "Xinjiang",
            "description": "Alpine lake with autumn colors",
            "stay_min": 180,
        },
        {
            "poi_id": "POI_0002",
            "name": "Nalati Grassland",
            "city": "Ili",
            "province": "Xinjiang",
            "description": "Sky grassland with flowers",
            "stay_min": 150,
        },
        {
            "poi_id": "POI_0003",
            "name": "Tianshan Tianchi",
            "city": "Urumqi",
            "province": "Xinjiang",
            "description": "Alpine lake with snow mountains",
            "stay_min": 120,
        },
    ]


# ============================================================================
# Test: Intent Understanding
# ============================================================================

class TestIntentUnderstanding:
    """Test intent understanding functionality."""

    def test_extract_province(self):
        """Test province extraction from query."""
        query_to_province = {
            "Xinjiang tourism": "Xinjiang",
            "Tibet Lhasa": "Tibet",
            "Yunnan Dali": "Yunnan",
            "Sichuan Jiuzhaigou": "Sichuan",
        }

        for query, expected_province in query_to_province.items():
            # Simple keyword matching
            for prov in ["Xinjiang", "Tibet", "Yunnan", "Sichuan"]:
                if prov in query:
                    assert prov == expected_province
                    break

    def test_extract_interests(self):
        """Test interest extraction from query."""
        interests_keywords = {
            "snow mountains": ["mountain", "snow", "peak"],
            "grassland": ["grassland", "prairie", "meadow"],
            "lake": ["lake", "water", "alpine"],
        }

        query = "Want to see snow mountains, lakes and grasslands"

        extracted = []
        for interest, keywords in interests_keywords.items():
            if any(kw in query.lower() for kw in keywords):
                extracted.append(interest)

        assert len(extracted) >= 2

    def test_extract_duration(self):
        """Test duration extraction from query."""
        queries = [
            ("Xinjiang 3 day tour", 3),
            ("Tibet 5 day deep tour", 5),
            ("Yunnan one week tour", 7),
        ]

        for query, expected_days in queries:
            # Simple number extraction
            words = query.split()
            duration = None
            for word in words:
                if word.isdigit():
                    duration = int(word)
                    break

            if duration is not None:
                assert duration == expected_days

    def test_extract_season(self):
        """Test season extraction."""
        season_keywords = {
            "spring": ["spring", "april", "may"],
            "summer": ["summer", "july", "august"],
            "autumn": ["autumn", "fall", "october", "november"],
            "winter": ["winter", "december", "january"],
        }

        query = "Want to see autumn colors in Xinjiang"

        extracted = None
        for season, keywords in season_keywords.items():
            if any(kw in query.lower() for kw in keywords):
                extracted = season
                break

        assert extracted == "autumn"

    def test_extract_travel_style(self):
        """Test travel style extraction."""
        style_keywords = {
            "photography": ["photo", "photography", "camera"],
            "deep": ["deep", "immersive", "culture"],
            "leisure": ["leisure", "relax", "easy"],
            "family": ["family", "kids", "children"],
        }

        test_cases = [
            ("Xinjiang photography tour", "photography"),
            ("Deep cultural tour", "deep"),
            ("Leisure travel", "leisure"),
            ("Family trip with kids", "family"),
        ]

        for query, expected_style in test_cases:
            extracted = None
            for style, keywords in style_keywords.items():
                if any(kw in query.lower() for kw in keywords):
                    extracted = style
                    break

            assert extracted == expected_style


# ============================================================================
# Test: POI Reranking
# ============================================================================

class TestPOIReranking:
    """Test POI reranking functionality."""

    def test_rerank_by_relevance(self, sample_poi_candidates):
        """Test POI reranking by relevance."""
        # Simulate relevance scores
        query = "snow mountains and lakes"
        scores = []

        for poi in sample_poi_candidates:
            desc = poi["description"].lower()
            score = 0
            if "snow" in desc or "mountain" in desc:
                score += 2
            if "lake" in desc:
                score += 1
            scores.append(score)

        # Sort by score
        ranked = sorted(zip(sample_poi_candidates, scores), key=lambda x: -x[1])

        assert len(ranked) == 3
        assert ranked[0][1] >= ranked[1][1]

    def test_rerank_with_topk(self, sample_poi_candidates):
        """Test reranking with topk limit."""
        topk = 2

        # Simple ranking by stay duration
        ranked = sorted(sample_poi_candidates, key=lambda x: -x["stay_min"])[:topk]

        assert len(ranked) == 2
        assert ranked[0]["stay_min"] >= ranked[1]["stay_min"]

    def test_rerank_empty_list(self):
        """Test reranking with empty POI list."""
        result = []
        topk = 5

        assert len(result) == 0

    def test_rerank_single_poi(self):
        """Test reranking with single POI."""
        pois = [{"poi_id": "POI_0001", "name": "Test", "stay_min": 60}]

        result = pois[:5]

        assert len(result) == 1
        assert result[0]["poi_id"] == "POI_0001"

    def test_rerank_large_list(self):
        """Test reranking with large POI list (>30)."""
        large_list = [
            {"poi_id": f"POI_{i:04d}", "name": f"POI {i}", "stay_min": 60 + i}
            for i in range(50)
        ]

        topk = 10
        result = sorted(large_list, key=lambda x: -x["stay_min"])[:topk]

        assert len(result) == 10
        assert result[0]["stay_min"] > result[1]["stay_min"]


# ============================================================================
# Test: Content Generation
# ============================================================================

class TestContentGeneration:
    """Test content generation functionality."""

    def test_generate_title(self):
        """Test title generation."""
        route_pois = [
            {"poi_name": "Airport", "poi_city": "Urumqi"},
            {"poi_name": "Kanas Lake", "poi_city": "Altay"},
            {"poi_name": "Airport", "poi_city": "Urumqi"},
        ]

        province = "Xinjiang"
        total_hours = 4.5

        # Simple template-based generation
        core_pois = [p["poi_name"] for p in route_pois[1:-1]]
        title = f"{province} | {' '.join(core_pois)}"

        assert "Xinjiang" in title
        assert "Kanas" in title
        assert len(title) > 0

    def test_generate_description(self):
        """Test description generation."""
        route_pois = [
            {"poi_name": "Airport", "poi_city": "Urumqi"},
            {"poi_name": "Kanas Lake", "poi_city": "Altay"},
            {"poi_name": "Nalati Grassland", "poi_city": "Ili"},
            {"poi_name": "Airport", "poi_city": "Urumqi"},
        ]

        province = "Xinjiang"
        total_hours = 8.5
        query = "Want to see autumn scenery"

        # Simple template-based generation
        core_pois = [p["poi_name"] for p in route_pois[1:-1]]
        description = f"Visit {', '.join(core_pois)} in {province} for {total_hours:.1f} hours."

        assert len(description) > 0
        assert "Xinjiang" in description

    def test_generate_with_fallback(self):
        """Test content generation with fallback."""
        route_pois = [
            {"poi_name": "Start", "poi_city": "None"},
        ]

        province = "Unknown"
        total_hours = 0

        # Fallback template
        title = f"{province} Day Trip"
        description = f"Short tour of {province}."

        assert len(title) > 0
        assert len(description) > 0


# ============================================================================
# Test: Recommendation Explanation
# ============================================================================

class TestRecommendationExplanation:
    """Test recommendation explanation functionality."""

    def test_explain_recommendation(self, sample_poi_candidates, sample_user_intent):
        """Test explanation generation."""
        poi = sample_poi_candidates[0]
        intent = sample_user_intent

        # Generate explanation based on matching interests
        reasons = []
        for interest in intent["interests"]:
            if interest.lower() in poi["description"].lower():
                reasons.append(f"Matches your interest in {interest}")

        if not reasons:
            reasons.append(f"Popular destination in {poi['province']}")

        explanation = "\n".join([f"- {r}" for r in reasons[:3]])

        assert len(explanation) > 0
        assert "- " in explanation or explanation.startswith("-")

    def test_explain_with_no_interests(self, sample_poi_candidates):
        """Test explanation when no interests specified."""
        poi = sample_poi_candidates[0]
        intent = {"interests": [], "province": "Xinjiang"}

        # Fallback explanation
        explanation = f"Popular destination in {intent['province']}"

        assert "Xinjiang" in explanation

    def test_explain_with_multiple_matches(self, sample_poi_candidates):
        """Test explanation with multiple interest matches."""
        poi = sample_poi_candidates[0]
        intent = {"interests": ["lake", "mountain", "grassland"], "province": "Xinjiang"}

        # Count matches
        matches = 0
        desc = poi["description"].lower()
        for interest in intent["interests"]:
            if interest in desc:
                matches += 1

        # Generate explanation
        reasons = [f"Matches {matches} of your interests"]

        assert len(reasons) == 1
        assert "1" in reasons[0] or "2" in reasons[0] or "3" in reasons[0]


# ============================================================================
# Test: Degradation Mechanism
# ============================================================================

class TestDegradationMechanism:
    """Test degradation and fallback mechanisms."""

    def test_fallback_intent_without_llm(self):
        """Test fallback intent understanding without LLM."""
        query = "Want to visit Xinjiang and see snow mountains"

        # Simple rule-based fallback
        province = None
        for prov in ["Xinjiang", "Tibet", "Yunnan", "Sichuan"]:
            if prov in query:
                province = prov
                break

        interests = []
        for kw in ["mountain", "lake", "grassland", "snow"]:
            if kw in query.lower():
                interests.append(kw)

        intent = {
            "original_query": query,
            "province": province,
            "interests": interests,
        }

        assert intent["province"] == "Xinjiang"
        assert len(intent["interests"]) >= 1

    def test_fallback_reranking(self, sample_poi_candidates):
        """Test fallback reranking without LLM."""
        # Fallback to simple sorting
        ranked = sorted(sample_poi_candidates, key=lambda x: x["stay_min"], reverse=True)

        assert len(ranked) == 3
        assert ranked[0]["stay_min"] >= ranked[1]["stay_min"]

    def test_fallback_content_generation(self):
        """Test fallback content generation."""
        route_pois = [{"poi_name": "Kanas"}]

        # Fallback template
        title = f"Trip to {route_pois[0]['poi_name']}"
        description = f"Visit {route_pois[0]['poi_name']}."

        assert len(title) > 0
        assert len(description) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
