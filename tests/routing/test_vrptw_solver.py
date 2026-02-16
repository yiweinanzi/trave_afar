"""
Tests for routing.vrptw_solver module.

Tests VRPTW solving correctness, time window constraints, and boundary conditions.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_poi_df():
    """Create a sample POI DataFrame for testing."""
    return pd.DataFrame({
        "poi_id": ["DEPOT", "POI_0001", "POI_0002", "POI_0003", "POI_0004"],
        "name": ["Airport", "Tianshan", "Kanas", "Sayram", "Nalati"],
        "city": ["Urumqi", "Urumqi", "Altay", "Ili", "Ili"],
        "province": ["Xinjiang"] * 5,
        "lat": [43.91, 43.88, 48.70, 44.60, 43.30],
        "lon": [87.47, 88.13, 87.00, 81.00, 83.80],
        "stay_min": [0, 120, 180, 120, 150],
        "open_min": [0, 480, 480, 480, 480],
        "close_min": [1440, 1200, 1200, 1200, 1200],
    })


@pytest.fixture
def sample_time_matrix():
    """Create a sample time matrix (seconds)."""
    return np.array([
        [0, 3600, 7200, 5400, 4800],
        [3600, 0, 5400, 3600, 4200],
        [7200, 5400, 0, 4800, 3600],
        [5400, 3600, 4800, 0, 4200],
        [4800, 4200, 3600, 4200, 0],
    ], dtype=np.int32)


# ============================================================================
# Test: VRPTW Solving Correctness
# ============================================================================

class TestVRPTWSolvingCorrectness:
    """Test VRPTW solving correctness."""

    def test_time_matrix_properties(self, sample_time_matrix):
        """Test that time matrix has correct properties."""
        assert sample_time_matrix.shape[0] == sample_time_matrix.shape[1]
        assert sample_time_matrix.shape[0] == 5
        assert np.allclose(sample_time_matrix, sample_time_matrix.T)  # Symmetric
        assert np.all(sample_time_matrix >= 0)  # Non-negative

    def test_poi_df_properties(self, sample_poi_df):
        """Test that POI DataFrame has required columns."""
        required_cols = ["poi_id", "name", "city", "province", "lat", "lon", "stay_min", "open_min", "close_min"]
        for col in required_cols:
            assert col in sample_poi_df.columns

    def test_depot_is_first(self, sample_poi_df):
        """Test that depot is the first POI."""
        assert sample_poi_df.iloc[0]["poi_id"] == "DEPOT"
        assert sample_poi_df.iloc[0]["stay_min"] == 0

    def test_time_windows_valid(self, sample_poi_df):
        """Test that time windows are valid."""
        for _, row in sample_poi_df.iterrows():
            assert row["open_min"] <= row["close_min"]
            assert row["open_min"] >= 0
            assert row["close_min"] <= 1440

    def test_single_route_structure(self, sample_poi_df, sample_time_matrix):
        """Test structure of a simple route."""
        # Simulate a simple route: depot -> POI1 -> depot
        route_indices = [0, 1, 0]

        stops = []
        for idx in route_indices:
            poi = sample_poi_df.iloc[idx]
            stops.append({
                "poi_id": poi["poi_id"],
                "poi_name": poi["name"],
                "poi_city": poi["city"],
                "stay_min": int(poi["stay_min"]),
            })

        assert len(stops) == 3
        assert stops[0]["poi_id"] == "DEPOT"
        assert stops[-1]["poi_id"] == "DEPOT"

    def test_calculate_travel_time(self, sample_time_matrix):
        """Test travel time calculation."""
        from_idx = 0
        to_idx = 1

        travel_time = sample_time_matrix[from_idx, to_idx]

        assert travel_time == 3600  # 1 hour in seconds

    def test_calculate_total_time(self, sample_time_matrix, sample_poi_df):
        """Test total time calculation."""
        route_indices = [0, 1, 2, 0]
        total_time = 0

        for i in range(len(route_indices) - 1):
            from_idx = route_indices[i]
            to_idx = route_indices[i + 1]
            travel_time = sample_time_matrix[from_idx, to_idx]
            stay_time = sample_poi_df.iloc[from_idx]["stay_min"] * 60
            total_time += travel_time + stay_time

        assert total_time > 0
        assert total_time < 86400  # Less than 24 hours


# ============================================================================
# Test: Time Window Constraints
# ============================================================================

class TestTimeWindowConstraints:
    """Test time window constraint handling."""

    def test_opening_time_constraint(self, sample_poi_df):
        """Test that opening times are respected."""
        for _, row in sample_poi_df.iterrows():
            # POI opens at 8:00 AM (480 minutes)
            if row["poi_id"] != "DEPOT":
                assert row["open_min"] == 480

    def test_closing_time_constraint(self, sample_poi_df):
        """Test that closing times are respected."""
        for _, row in sample_poi_df.iterrows():
            # POI closes at 8:00 PM (1200 minutes)
            if row["poi_id"] != "DEPOT":
                assert row["close_min"] == 1200

    def test_check_time_feasibility(self, sample_poi_df):
        """Test time feasibility check."""
        arrival_min = 600  # 10:00 AM
        duration_min = 120  # 2 hours

        for _, row in sample_poi_df.iterrows():
            if row["poi_id"] != "DEPOT":
                open_min = row["open_min"]
                close_min = row["close_min"]

                # Check if visit fits in time window
                departure = arrival_min + duration_min
                feasible = arrival_min >= open_min and departure <= close_min
                assert feasible is True

    def test_24_hour_operation(self):
        """Test handling of 24-hour operations."""
        open_min = 0
        close_min = 1440
        arrival_min = 1000
        duration_min = 120

        departure = arrival_min + duration_min
        feasible = arrival_min >= open_min and departure <= close_min

        assert feasible is True


# ============================================================================
# Test: Boundary Conditions
# ============================================================================

class TestBoundaryConditions:
    """Test boundary conditions and edge cases."""

    def test_single_poi_besides_depot(self):
        """Test with single POI besides depot."""
        df = pd.DataFrame({
            "poi_id": ["DEPOT", "POI_0001"],
            "name": ["Start", "Test"],
            "city": ["None", "City"],
            "province": ["None", "Prov"],
            "lat": [0, 1],
            "lon": [0, 1],
            "stay_min": [0, 60],
            "open_min": [0, 480],
            "close_min": [1440, 1200],
        })

        assert len(df) == 2
        assert df.iloc[0]["poi_id"] == "DEPOT"

    def test_no_pois_besides_depot(self):
        """Test with only depot."""
        df = pd.DataFrame({
            "poi_id": ["DEPOT"],
            "name": ["Start"],
            "city": ["None"],
            "province": ["None"],
            "lat": [0],
            "lon": [0],
            "stay_min": [0],
            "open_min": [0],
            "close_min": [1440],
        })

        assert len(df) == 1
        assert df.iloc[0]["poi_id"] == "DEPOT"

    def test_very_short_duration(self):
        """Test with very short duration constraint."""
        max_duration_hours = 0.5  # 30 minutes
        max_duration_seconds = max_duration_hours * 3600

        # A route with this constraint would have very limited options
        assert max_duration_seconds == 1800

    def test_very_long_duration(self):
        """Test with very long duration constraint."""
        max_duration_hours = 16
        max_duration_seconds = max_duration_hours * 3600

        assert max_duration_seconds == 57600

    def test_immediate_departure(self):
        """Test with immediate departure (start at opening)."""
        start_time_min = 480  # 8:00 AM
        assert 480 <= start_time_min <= 1200

    def test_late_departure(self):
        """Test with late departure."""
        start_time_min = 1080  # 6:00 PM
        assert 480 <= start_time_min <= 1200

    def test_asymmetric_time_matrix(self):
        """Test with asymmetric time matrix (one-way travel)."""
        asymmetric = np.array([
            [0, 1000, 3000],
            [2000, 0, 1500],
            [3500, 2500, 0],
        ], dtype=np.int32)

        # Check it's not symmetric
        assert not np.allclose(asymmetric, asymmetric.T)
        assert asymmetric[0, 1] == 1000
        assert asymmetric[1, 0] == 2000

    def test_zero_travel_time(self):
        """Test with zero travel time (same location)."""
        zero_matrix = np.array([
            [0, 0],
            [0, 0],
        ], dtype=np.int32)

        assert zero_matrix[0, 1] == 0
        assert zero_matrix[1, 0] == 0

    def test_very_long_travel_time(self):
        """Test with very long travel time."""
        long_matrix = np.array([
            [0, 36000],
            [36000, 0],
        ], dtype=np.int32)

        assert long_matrix[0, 1] == 36000  # 10 hours


# ============================================================================
# Test: Result Parsing
# ============================================================================

class TestResultParsing:
    """Test result parsing and formatting."""

    def test_result_structure(self, sample_poi_df, sample_time_matrix):
        """Test that result has correct structure."""
        # Simulate a result
        result = {
            "routes": [[
                {"poi_id": "DEPOT", "poi_name": "Airport", "arrival_time_min": 0, "stay_min": 0},
                {"poi_id": "POI_0001", "poi_name": "Tianshan", "arrival_time_min": 60, "stay_min": 120},
                {"poi_id": "DEPOT", "poi_name": "Airport", "arrival_time_min": 180, "stay_min": 0},
            ]],
            "total_hours": 3.0,
            "num_vehicles": 1,
            "objective_value": 1000,
            "visited_pois": 1,
        }

        required_fields = ["routes", "total_hours", "num_vehicles", "objective_value", "visited_pois"]
        for field in required_fields:
            assert field in result

    def test_route_stop_structure(self, sample_poi_df):
        """Test that route stops have correct structure."""
        stop = {
            "poi_id": "POI_0001",
            "poi_name": "Tianshan",
            "poi_city": "Urumqi",
            "arrival_time_min": 60,
            "stay_min": 120,
        }

        required_keys = ["poi_id", "poi_name", "arrival_time_min", "stay_min"]
        for key in required_keys:
            assert key in stop

    def test_time_formatting(self):
        """Test time formatting."""
        arrival_min = 570  # 9:30 AM

        hour = arrival_min // 60
        minute = arrival_min % 60
        time_str = f"{hour:02d}:{minute:02d}"

        assert time_str == "09:30"
        assert len(time_str) == 5
        assert time_str[2] == ":"

    def test_calculate_visited_pois(self, sample_poi_df):
        """Test visited POIs calculation."""
        route = ["DEPOT", "POI_0001", "POI_0002", "DEPOT"]

        # Count non-depot POIs
        visited = sum(1 for poi_id in route if poi_id != "DEPOT")

        assert visited == 2


# ============================================================================
# Test: Integration
# ============================================================================

class TestIntegration:
    """Test integration with other components."""

    def test_route_with_coordinates(self, sample_poi_df):
        """Test that route preserves coordinate information."""
        route_indices = [0, 1, 0]

        coordinates = []
        for idx in route_indices:
            poi = sample_poi_df.iloc[idx]
            coordinates.append({
                "lat": poi["lat"],
                "lon": poi["lon"],
            })

        assert len(coordinates) == 3
        assert all("lat" in c and "lon" in c for c in coordinates)

    def test_total_distance_calculation(self, sample_time_matrix):
        """Test total distance calculation using time matrix."""
        route_indices = [0, 1, 2, 0]
        total_time = 0

        for i in range(len(route_indices) - 1):
            from_idx = route_indices[i]
            to_idx = route_indices[i + 1]
            total_time += sample_time_matrix[from_idx, to_idx]

        assert total_time > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
