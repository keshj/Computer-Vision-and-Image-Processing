import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_images(img1_path, img5_path):
    img1 = cv2.imread(img1_path)
    img5 = cv2.imread(img5_path)
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img5_gray = cv2.cvtColor(img5, cv2.COLOR_BGR2GRAY)
    return img1, img5, img1_gray, img5_gray

def detect_and_match_sift(img1_gray, img5_gray):
    sift = cv2.SIFT_create(nfeatures=10000, contrastThreshold=0.01, edgeThreshold=10, sigma=1.2)     # Create SIFT detector
    # Detect keypoints and compute descriptors
    kp1, desc1 = sift.detectAndCompute(img1_gray, None)
    kp5, desc5 = sift.detectAndCompute(img5_gray, None)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False) # Create BFMatcher
    matches = bf.knnMatch(desc1, desc5, k=2) # Match descriptors using KNN
    
    # Apply Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    
    print(f"SIFT keypoints in img1: {len(kp1)}")
    print(f"SIFT keypoints in img5: {len(kp5)}")
    print(f"Good matches after ratio test: {len(good_matches)}")
    
    return kp1, kp5, good_matches

def normalize_points(pts):
    c = np.mean(pts, axis=0)
    s = np.sqrt(2) / np.mean(np.linalg.norm(pts - c, axis=1))
    T = np.array([[s,0,-s*c[0]],[0,s,-s*c[1]],[0,0,1]], dtype=np.float64)
    ph = np.hstack([pts, np.ones((len(pts),1))])
    nph = (T @ ph.T).T
    return nph[:,:2], T

def compute_homography_dlt(src_pts, dst_pts):
    sN, Ts = normalize_points(src_pts)
    dN, Td = normalize_points(dst_pts)
    A = []
    for (x,y),(u,v) in zip(sN, dN):
        A.append([-x,-y,-1, 0, 0, 0, u*x, u*y, u])
        A.append([ 0, 0, 0,-x,-y,-1, v*x, v*y, v])
    A = np.asarray(A)
    _,_,Vt = np.linalg.svd(A)
    Hn = Vt[-1].reshape(3,3)
    H = np.linalg.inv(Td) @ Hn @ Ts
    return H / H[2,2]



def compute_homography_ransac(src_pts, dst_pts, thresh=50.0, max_iters=2000):
    best_H = None
    best_inliers = []
    max_inliers = 0
    n_points = src_pts.shape[0]
    
    for _ in range(max_iters):
        idx = np.random.choice(n_points, 4, replace=False)
        src_sample = src_pts[idx]
        dst_sample = dst_pts[idx]
        
        try:
            H = compute_homography_dlt(src_sample, dst_sample)
        except:
            continue
        
        src_homo = np.hstack([src_pts, np.ones((n_points, 1))])
        dst_pred = (H @ src_homo.T).T
        dst_pred = dst_pred[:, :2] / dst_pred[:, 2:]
        
        errors = np.sqrt(np.sum((dst_pts - dst_pred)**2, axis=1))
        inliers = np.where(errors < thresh)[0]
        
        if len(inliers) > max_inliers:
            max_inliers = len(inliers)
            best_inliers = inliers
            best_H = H
    
    if len(best_inliers) >= 4:
        best_H = compute_homography_dlt(src_pts[best_inliers], dst_pts[best_inliers])
    
    print(f"\nRANSAC Results:")
    print(f"Total inliers: {len(best_inliers)} / {n_points}")
    print(f"Inlier ratio: {len(best_inliers)/n_points:.2%}")
    
    return best_H, best_inliers

def visualize_matches(img1, img5, kp1, kp5, matches, inliers=None):
    if inliers is not None:
        matches_to_draw = [matches[i] for i in inliers[:500]]  # Draw first 50 inliers
        title = f"SIFT Matches (Inliers: {len(inliers)})"
    else:
        matches_to_draw = matches[:500]  # Draw first 50 matches
        title = f"SIFT Matches (Total: {len(matches)})"
    
    img_matches = cv2.drawMatches(img1, kp1, img5, kp5, matches_to_draw, None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    plt.figure(figsize=(15, 8))
    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('matches.png', dpi=150, bbox_inches='tight')
    plt.show()

def warp_and_stitch(img1, img5, H):
    h1, w1 = img1.shape[:2]
    h5, w5 = img5.shape[:2]
    
    # Get corners of img1
    corners_img1 = np.array([[0, 0], [w1, 0], [w1, h1], [0, h1]], dtype=np.float32)
    corners_homo = np.hstack([corners_img1, np.ones((4, 1))])
    
    # Transform corners
    corners_transformed = (H @ corners_homo.T).T
    corners_transformed = corners_transformed[:, :2] / corners_transformed[:, 2:]
    
    # Calculate output size
    all_corners = np.vstack([corners_transformed, [[0, 0], [w5, 0], [w5, h5], [0, h5]]])
    [x_min, y_min] = np.int32(all_corners.min(axis=0))
    [x_max, y_max] = np.int32(all_corners.max(axis=0))
    
    # Translation to keep all pixels
    translation = np.array([[1, 0, -x_min],
                           [0, 1, -y_min],
                           [0, 0, 1]], dtype=np.float32)
    
    # Warp img1
    output_size = (x_max - x_min, y_max - y_min)
    img1_warped = cv2.warpPerspective(img1, translation @ H, output_size)
    
    # Create canvas and place img5 FIRST
    result = np.zeros((y_max - y_min, x_max - x_min, 3), dtype=img1.dtype)
    result[-y_min:-y_min+h5, -x_min:-x_min+w5] = img5
    
    # Blend: only overwrite where img1_warped has valid pixels
    mask = (img1_warped > 0).any(axis=2)
    result[mask] = img1_warped[mask]
    
    return result


img1_path = 'graf/graf/img1.ppm'
img5_path = 'graf/graf/img5.ppm'
img1, img5, img1_gray, img5_gray = load_images(img1_path, img5_path)

# (a) Detect and match SIFT features
print("Task (a): SIFT Feature Detection and Matching")
kp1, kp5, good_matches = detect_and_match_sift(img1_gray, img5_gray)

# Visualize initial matches
visualize_matches(img1, img5, kp1, kp5, good_matches)

# Extract matched point coordinates
src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
dst_pts = np.float32([kp5[m.trainIdx].pt for m in good_matches])

# (b) Compute homography using RANSAC
print("Task (b): Homography Estimation with RANSAC")
H_custom, inliers = compute_homography_ransac(src_pts, dst_pts, thresh=25, max_iters=5000)  # Lower threshold and more iterations for better accuracy

print("\nCustom Homography Matrix:")
print(H_custom)

# Compare with OpenCV's homography
# H_opencv, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
H_opencv = H_opencv = np.loadtxt('graf/graf/H1to5p') 
print("\nOpenCV Homography Matrix:")
print(H_opencv)

print("\nDifference between matrices:")
print(np.abs(H_custom - H_opencv))
print(f"Max absolute difference: {np.max(np.abs(H_custom - H_opencv)):.6f}")

# Visualize inlier matches
visualize_matches(img1, img5, kp1, kp5, good_matches, inliers)


# Warp img1 to img5's plane using the estimated homography
h5, w5 = img5.shape[:2]
warped_img1 = cv2.warpPerspective(img1, H_custom, (w5, h5))

# Visualize side by side
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img5, cv2.COLOR_BGR2RGB))
plt.title('Reference Image (img5)')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(warped_img1, cv2.COLOR_BGR2RGB))
plt.title('Warped img1 using Homography')
plt.axis('off')

plt.tight_layout()
plt.savefig('warped_homography_result.png', dpi=150, bbox_inches='tight')
plt.show()


# (c) Stitch images
print("Task (c): Image Stitching")
result = warp_and_stitch(img1, img5, H_custom)

print(f"Stitched image size: {result.shape}")

# Display result
plt.figure(figsize=(15, 10))
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title(f'Stitched Image ({img1_path} onto {img5_path})')
plt.axis('off')
plt.tight_layout()
plt.savefig('stitched_result.png', dpi=150, bbox_inches='tight')
plt.show()

# Save result
cv2.imwrite('stitched_image.jpg', result)
print("\nStitched image saved as 'stitched_image.jpg'")

