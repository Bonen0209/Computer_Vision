import time
import numpy as np
import cv2

def solve_homography(u, v, svd=True):
    N = u.shape[0]
    if v.shape[0] is not N:
        print('u and v should have the same size')
        return None
    if N < 4:
        print('At least 4 points should be given')
        return None
    
    if svd:
        A = np.zeros((2*N, 9))   
    else:
        A = np.zeros((2*N, 8))
    B = np.zeros((2*N, 1))
    H = np.zeros((3, 3))

    for i in range(N):
        A[2*i, :2] = u[i]
        A[2*i, 2] = 1
        A[2*i, 6] = - u[i, 0] * v[i, 0]
        A[2*i, 7] = - u[i, 1] * v[i, 0]

        A[2*i+1, 3:5] = u[i]
        A[2*i+1, 5] = 1
        A[2*i+1, 6] = - u[i, 0] * v[i, 1]
        A[2*i+1, 7] = - u[i, 1] * v[i, 1]

        if svd:
            A[2*i, 8] = - v[i, 0]
            A[2*i+1, 8] = - v[i, 1]
        else:
            B[2*i:2*i+2] = v[i].reshape(2, 1)

    if svd:
        U, s, V = np.linalg.svd(A)
        H = V.T[:, -1].reshape(3, 3)
    else:
        H_flat = np.linalg.solve(A, B)

        H[:2, :,] = H_flat[:6].reshape(2, 3)
        H[2, :2] = H_flat[6:].reshape(2)
        H[2, 2] = 1
    
    return H
    
def bilinear(img, x, y):
    w_l = x - int(x)
    w_r = 1 - w_l
    w_d = y - int(y)
    w_u = 1 - w_d

    i_x, i_y = int(x), int(y)

    return w_l * w_d * img[i_y, i_x] + w_r * w_d * img[i_y, i_x+1] + w_l * w_u * img[i_y+1, i_x] + w_r * w_u * img[i_y+1, i_x+1]

def nearest(img, x, y):
    pass

# corners are 4-by-2 arrays, representing the four image corner (x, y) pairs
def transform(img, canvas, corners, backward_warping=False):
    h, w, ch = img.shape

    img_corners = np.array([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]])
    homography = solve_homography(img_corners, corners, svd=False)

    for x in range(w):
        for y in range(h):
            u = np.array([x, y, 1]).T
            v = np.dot(homography, u)
            d = (v / v[2])
            
            if backward_warping:
                px = bilinear(canvas, d[0], d[1])
                d = d.astype(np.int64)
                img[y, x, :] = px
            else:
                px = img[y, x, :]
                d = d.astype(np.int64)
                canvas[d[1], d[0], :] = px

def main():
    # Part 1
    ts = time.time()
    canvas = cv2.imread('./input/Akihabara.jpg')
    img1 = cv2.imread('./input/lu.jpg')
    img2 = cv2.imread('./input/kuo.jpg')
    img3 = cv2.imread('./input/haung.jpg')
    img4 = cv2.imread('./input/tsai.jpg')
    img5 = cv2.imread('./input/han.jpg')

    canvas_corners1 = np.array([[779,312],[1018,175],[737,750],[981,645]])
    canvas_corners2 = np.array([[1194,496],[1537,458],[1168,961],[1523,932]])
    canvas_corners3 = np.array([[2693,250],[2886,390],[2754,1344],[2955,1403]])
    canvas_corners4 = np.array([[3563,475],[3882,803],[3614,921],[3921,1158]])
    canvas_corners5 = np.array([[2006,887],[2622,900],[2008,1349],[2640,1357]])
    
    transform(img1, canvas, canvas_corners1)
    transform(img2, canvas, canvas_corners2)
    transform(img3, canvas, canvas_corners3)
    transform(img4, canvas, canvas_corners4)
    transform(img5, canvas, canvas_corners5)

    cv2.imwrite('./output/part1.png', canvas)
    te = time.time()
    print('Elapse time: {}...'.format(te-ts))

    # Part 2
    ts = time.time()
    img_QRcode = cv2.imread('./input/QR_code.jpg')

    output2 = np.zeros((100, 100, 3))
    corner_QRcode = np.array([[1980, 1239], [2041, 1213], [2025, 1396], [2083, 1364]])
    transform(output2, img_QRcode, corner_QRcode, backward_warping=True)

    cv2.imwrite('./output/part2.png', output2)
    te = time.time()
    print('Elapse time: {}...'.format(te-ts))

    # Part 3
    ts = time.time()
    img_crosswalk = cv2.imread('./input/crosswalk_front.jpg')
    
    output3 = np.zeros((400, 500, 3))
    corner_crosswalk = np.array([[140, 130], [600, 130], [5, 335], [700, 335]])
    transform(output3, img_crosswalk, corner_crosswalk, backward_warping=True)

    cv2.imwrite('./output/part3.png', output3)
    te = time.time()
    print('Elapse time: {}...'.format(te-ts))

if __name__ == '__main__':
    main()
