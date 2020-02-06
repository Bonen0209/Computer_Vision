import sys
import time
import numpy as np
import cv2
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from tqdm import tqdm

class Video(object):
    MIN_MATCHES = 15

    def __init__(self, ref_path, tem_path, video_path):
        self.ref_path = ref_path
        self.tem_path = tem_path
        self.video_path = video_path

        self.ref_img, \
        self.tem_img, \
        self.video, \
        self.videowriter = self._getdata()
        self.frames = self._getframe()

        self.surf = cv2.xfeatures2d.SURF_create()
        self.bf = cv2.BFMatcher_create()

        self.kp_tem, self.des_tem = self.surf.detectAndCompute(self.tem_img, None)
    
    def _getdata(self):
        tem_image = cv2.imread(self.tem_path)
        ref_image = cv2.imread(self.ref_path)
        video = cv2.VideoCapture(self.video_path)

        ref_h, ref_w = tem_image.shape[0], tem_image.shape[1]
        ref_image = cv2.resize(ref_image, (ref_h, ref_w))
        
        film_h, film_w = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        film_fps = video.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        videowriter = cv2.VideoWriter("./output/ar_video.mp4", fourcc, film_fps, (film_w, film_h))

        return ref_image, tem_image, video, videowriter
    
    def _getframe(self):
        frames = []

        while(self.video.isOpened()):
            ret, frame = self.video.read()

            if not ret:
                break
            
            frames.append(frame)
        
        return frames

    def process_frame(self, frame_id):
        kp_f, des_f = self.surf.detectAndCompute(self.frames[frame_id], None)
        knn_matches = self.bf.knnMatch(self.des_tem, des_f, k=2)

        ratio_thresh = 0.75
        good_matches = []
        for m,n in knn_matches:
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)

        if len(good_matches) > Video.MIN_MATCHES:
            p_template = np.array([self.kp_tem[m.queryIdx].pt for m in good_matches], dtype=np.float64).reshape(-1, 1, 2)
            p_frame = np.array([kp_f[m.trainIdx].pt for m in good_matches], dtype=np.float64).reshape(-1, 1, 2)
            homography, _ = cv2.findHomography(p_template, p_frame, cv2.RANSAC, 5.0)

        h, w, _ = self.ref_img.shape
        self.frames[frame_id] = cv2.warpPerspective(self.ref_img, homography,
                                                   (self.frames[frame_id].shape[1], self.frames[frame_id].shape[0]),
                                                   dst=self.frames[frame_id],
                                                   borderMode=cv2.BORDER_TRANSPARENT)
        
    def run(self):
        ts = time.time()

        for idx in range(len(self.frames)):
            print(idx)
            self.videowriter.write(self.process_frame(idx))

        te = time.time()
        print(te-ts)
        
        for frame in self.frames:
            self.videowriter.write(frame)

        self._clean()

    def _clean(self):
        self.video.release()
        self.videowriter.release()
        cv2.destroyAllWindows()

def main(ref_image,template,video):
    AR_video = Video(ref_image, template, video)
    AR_video.run()

if __name__ == '__main__':
    ref_path = './input/sychien.jpg'
    template_path = './input/marker.png'
    video_path = sys.argv[1]
    main(ref_path,template_path,video_path)