#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

print("=" * 60)
print("GoAfar Full Pipeline Experiment")
print("=" * 60)
print(f"Project Root: {PROJECT_ROOT}")

# 1. Check data
print("\n[1/5] Checking data files...")
emb_path = PROJECT_ROOT / "outputs/emb/poi_emb.npy"
meta_path = PROJECT_ROOT / "outputs/emb/poi_meta.csv"
time_matrix_path = PROJECT_ROOT / "outputs/routing/time_matrix.npy"

print(f"  poi_emb.npy: {emb_path.stat().st_size/1024/1024:.1f} MB" if emb_path.exists() else "  MISSING")
print(f"  poi_meta.csv: {meta_path.stat().st_size/1024:.1f} KB" if meta_path.exists() else "  MISSING")
print(f"  time_matrix.npy: {time_matrix_path.stat().st_size/1024:.1f} KB" if time_matrix_path.exists() else "  MISSING")

# 2. Test vector retrieval
print("\n[2/5] Testing vector retrieval...")
try:
    import numpy as np
    import pandas as pd
    
    poi_emb = np.load(str(emb_path))
    poi_meta = pd.read_csv(str(meta_path))
    
    print(f"  Loaded: {poi_emb.shape} vectors, {len(poi_meta)} POIs")
    print(f"  Sample POIs:")
    for i in range(min(5, len(poi_meta))):
        row = poi_meta.iloc[i]
        print(f"    {i+1}. {row.get('name', 'N/A')} ({row.get('province', 'N/A')})")
except Exception as e:
    print(f"  Error: {e}")

# 3. Test routing
print("\n[3/5] Testing routing...")
try:
    import pandas as pd
    import numpy as np
    from math import radians, cos, sin, asin, sqrt
    
    test_pois = pd.DataFrame([
        {'name': 'Tianchi', 'lat': 43.88, 'lon': 88.13, 'stay_minutes': 120},
        {'name': 'Kanas', 'lat': 48.70, 'lon': 87.02, 'stay_minutes': 180},
        {'name': 'Grand Bazaar', 'lat': 43.82, 'lon': 87.60, 'stay_minutes': 60},
    ])
    
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        return 2 * R * asin(sqrt(a))
    
    n = len(test_pois)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist_matrix[i][j] = haversine(
                    test_pois.iloc[i]['lat'], test_pois.iloc[i]['lon'],
                    test_pois.iloc[j]['lat'], test_pois.iloc[j]['lon']
                )
    
    print(f"  Distance matrix: {dist_matrix.shape}")
    print(f"  Min distance: {dist_matrix[dist_matrix > 0].min():.1f} km")
except Exception as e:
    print(f"  Error: {e}")

# 4. Test recommendation flow
print("\n[4/5] Testing recommendation flow...")
try:
    import pandas as pd
    
    request = {
        'query': 'Xinjiang 7 days snow mountain and grassland',
        'city': 'Xinjiang',
        'days': 7,
        'budget': 5000,
        'interests': ['nature', 'photography']
    }
    
    print(f"  Request:")
    print(f"    City: {request['city']}")
    print(f"    Days: {request['days']}")
    print(f"    Interests: {', '.join(request['interests'])}")
    
    poi_meta = pd.read_csv(str(meta_path))
    
    if 'province' in poi_meta.columns:
        filtered = poi_meta[poi_meta['province'].str.contains('新疆', na=False)]
    else:
        filtered = poi_meta.head(50)
    
    print(f"  Filtered candidates: {len(filtered)}")
    
    # Simulate routing
    num_days = request['days']
    pois_per_day = min(3, len(filtered) // num_days)
    routes = []
    
    for day in range(num_days):
        start_idx = day * pois_per_day
        end_idx = min(start_idx + pois_per_day, len(filtered))
        day_pois = []
        
        for idx in range(start_idx, end_idx):
            row = filtered.iloc[idx]
            day_pois.append({
                'name': row.get('name', 'Unknown'),
                'category': row.get('category', 'N/A'),
            })
        
        if day_pois:
            routes.append(day_pois)
    
    print(f"  Planned {len(routes)} days")
    if routes:
        print(f"  Day 1: {[p['name'] for p in routes[0]]}")
    
except Exception as e:
    print(f"  Error: {e}")
    traceback.print_exc()

# 5. Performance test
print("\n[5/5] Performance benchmark...")
try:
    import numpy as np
    import pandas as pd
    
    t0 = time.time()
    poi_emb = np.load(str(emb_path))
    poi_meta = pd.read_csv(str(meta_path))
    t1 = time.time()
    print(f"  Data loading: {(t1-t0)*1000:.1f}ms")
    
    t0 = time.time()
    for _ in range(100):
        _ = np.sum(poi_emb[:100])
    t1 = time.time()
    print(f"  Vector ops (100x): {(t1-t0)*1000:.1f}ms")
    
except Exception as e:
    print(f"  Error: {e}")

print("\n" + "=" * 60)
print("Experiment Complete")
print("=" * 60)
