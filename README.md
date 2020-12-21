# 3dMM
Adapted from https://github.com/YadiraF/face3d. Please use proper Git etiquette.

Minor modifications have been introduced to enable MATLAB data extraction from the AFLW2000-3D and 300W-3D datasets. The model generalizes poorly for darker skin tones, and a scaling of colors is recommended in MorphabelModel.py.

s in compute_strength is dimensionless. d is in centimeters. Refer to "A Comprehensive Examination of Topographic Thickness of Skin in the Human Face" for accurate values. The normals function itself is currently buggy.

The following files must be individually downloaded/incorporated:

- shape_predictor_68_face_landmarks.dat
  This file is being phased from implementation, but may still be required for smooth operation. It should   
  be included in face3d/examples.
- face3d/examples/Data/<>
  The 3ddfa, BFM, out, and raw folders should be included in the Data folder, alongside the existing stn 
  folder and other contents.
