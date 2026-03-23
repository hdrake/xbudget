import xarray as xr
import xgcm

def construct_grid(ds):
    # define the connectivity between faces
    face_connections = {'tile':
                        {0: {'X':  ((12, 'Y', False), (3, 'X', False)),
                             'Y':  (None,             (1, 'Y', False))},
                         1: {'X':  ((11, 'Y', False), (4, 'X', False)),
                             'Y':  ((0, 'Y', False),  (2, 'Y', False))},
                         2: {'X':  ((10, 'Y', False), (5, 'X', False)),
                             'Y':  ((1, 'Y', False),  (6, 'X', False))},
                         3: {'X':  ((0, 'X', False),  (9, 'Y', False)),
                             'Y':  (None,             (4, 'Y', False))},
                         4: {'X':  ((1, 'X', False),  (8, 'Y', False)),
                             'Y':  ((3, 'Y', False),  (5, 'Y', False))},
                         5: {'X':  ((2, 'X', False),  (7, 'Y', False)),
                             'Y':  ((4, 'Y', False),  (6, 'Y', False))},
                         6: {'X':  ((2, 'Y', False),  (7, 'X', False)),
                             'Y':  ((5, 'Y', False),  (10, 'X', False))},
                         7: {'X':  ((6, 'X', False),  (8, 'X', False)),
                             'Y':  ((5, 'X', False),  (10, 'Y', False))},
                         8: {'X':  ((7, 'X', False),  (9, 'X', False)),
                             'Y':  ((4, 'X', False),  (11, 'Y', False))},
                         9: {'X':  ((8, 'X', False),  None),
                             'Y':  ((3, 'X', False),  (12, 'Y', False))},
                         10: {'X': ((6, 'Y', False),  (11, 'X', False)),
                              'Y': ((7, 'Y', False),  (2, 'X', False))},
                         11: {'X': ((10, 'X', False), (12, 'X', False)),
                              'Y': ((8, 'Y', False),  (1, 'X', False))},
                         12: {'X': ((11, 'X', False), None),
                              'Y': ((9, 'Y', False),  (0, 'X', False))}}}

    coords = {
        "X": {"center": "i", "left": "i_g"},
        "Y": {"center": "j", "left": "j_g"},
        "T": {"center": "time", "outer": "time_bounds"},
        "Z": {"center": "k", "left": "k_l"},
    }
    
    metrics = {
        ("X",): ["dxG"],          # distances between two X faces
        ("Y",): ["dyG"],          # distances between two Y faces
        ("Z",): ["drF"],  # 1D Z distances between cell_boundaries
        ("X", "Y"): ["rA", "rAw", "rAs"],            # horizontal areas (cell center, west-face, south-face)
    }

    boundary = {"X":None, "Y":None, "Z": "fill", "T":None}
    fill_value = {"Z": 0.0}

    grid = xgcm.Grid(
        ds,
        coords=coords,
        metrics=metrics,
        boundary=boundary,
        periodic = False,
        fill_value=fill_value,
        face_connections=face_connections,
        autoparse_metadata=False,
    )
    return grid