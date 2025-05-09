'''
Compares the feature importance map from v2 and v3 with XGBoost,
and the PCA feature importance map.
'''
v3 = [
    "WSNO", "CAatINLET", "CAatOUTLET", "Startx", "StartY", "CHM_STD", "CNVRGI_MEDIAN",
    "Sinuosity", "CNVRGI_MAX", "CNVRGI MIN", "TRI_MIN", "Z_Mean", "CHM_MAX", "CNVRGI_STO",
    "CHM_PCT90", "TPL MAX", "Avg_Slope", "TPI_PCT90", "Min_Slope", "TRI_RANGE"
]

v2 = [
    "Sinuosity", "Slope", "WSNO", "CAatINLET", "CAatOUTLET", "DEM25m_MIN", "Drop_",
    "COMPAC_STD", "DROUGHT_STD", "DEM25m_RANGE", "DEM25m_STD", "Straight_L",
    "EXCESSMOIST_STD", "COMPAC_MEAN", "WINDTHROW_STD", "DEM25m_SUM", "SOILTEMP_STD",
    "COMPAC_SUM", "DEM25m_MAX", "Magnitude"
]

pca = [
    "TPI_MAX", "CNVRGI_SUM", "CNVRGI_MEAN", "CNVRGI_MEDIAN", "CNVRGI_MIN", "DTW_SUM",
    "DroughtHaz_PCT90", "TRI_MIN", "CNVRGI_MAX", "DroughtHaz_MIN", "TRI_SUM",
    "DroughtHaz_MEDIAN", "Min_Slope", "CHM_STD", "TPI_SUM", "Avg_Slope", "DTW_MEDIAN",
    "Edge_Flow", "TPI_PCT90", "WSNO"
]

random_forest = ['CAatOUTLET', 'CAatINLET', 'Magnitude', 'WSNO', 'TPI_MIN', 'TPI_MEAN',
       'TRI_MEDIAN', 'Order_', 'Avg_Slope', 'StartY', 'TRI_MAX', 'DEM5m_MAX',
       'EndY', 'EndX', 'DEM5mPCT90', 'StartX', 'TPI_MEDIAN', 'Z_Min',
       'DEM5m_MEAN', 'Z_Max']




print(f"Length of v2: {len(v2)}")
print(f"Length of v3: {len(v3)}")
print(f"Length of pca: {len(pca)}")

print(f"Columns in pca but not in v2: {set(pca) - set(v2)}")
print(f"Columns in pca but not in v3: {set(pca) - set(v3)}")
print(f"Common columns in pca and v3: {set(pca).intersection(set(v3))}")

print(f"Common columns in v3 and random_forest: {set(v3).intersection(set(random_forest))}")
print(f"Columns in v3 but not in random_forest: {set(v3) - set(random_forest)}")
print(f"Columns in random_forest but not in v3: {set(random_forest) - set(v3)}")


