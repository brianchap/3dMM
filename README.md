# 3dMM
Adapted from https://github.com/YadiraF/face3d. Please use proper Git etiquette.

Minor modifications have been introduced to enable MATLAB data extraction from the AFLW2000-3D and 300W-3D datasets. The model generalizes poorly for darker skin tones, and a scaling of colors is recommended in MorphabelModel.py.

s in compute_strength is dimensionless. d is in centimeters. Refer to "A Comprehensive Examination of Topographic Thickness of Skin in the Human Face" for accurate values. The normals function itself is currently buggy.


