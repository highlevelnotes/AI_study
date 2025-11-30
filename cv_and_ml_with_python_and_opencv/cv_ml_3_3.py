import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture('video1.mp4')

if not cap.isOpened():
    print("비디오 파일을 열 수 없습니다.")
else:
    bgMethod1 = cv2.createBackgroundSubtractorMOG2()
    bgMethod2 = cv2.createBackgroundSubtractorKNN()
    bgMethod1_blur = cv2.createBackgroundSubtractorMOG2()
    bgMethod2_blur = cv2.createBackgroundSubtractorKNN()
    
    imgIndex = 1
    frame = None
    bgMOG = None
    bgKNN = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame = cv2.resize(frame, (320, 240))
        
        fgMOG = bgMethod1.apply(frame, learningRate=-1)
        fgKNN = bgMethod2.apply(frame)
        fgMOG_blur = bgMethod1_blur.apply(cv2.blur(frame, (5, 5)), learningRate=-1)
        fgKNN_blur = bgMethod2_blur.apply(cv2.blur(frame, (5, 5)))
        
        if imgIndex > 1:
            bgMOG = bgMethod1.getBackgroundImage()
            bgKNN = bgMethod2.getBackgroundImage()
            bgMOG_blur = bgMethod1_blur.getBackgroundImage()
            bgKNN_blur = bgMethod2_blur.getBackgroundImage()
        
        imgIndex += 1

    cap.release()

    # 결과 표시 (마지막 프레임 기준)
    if frame is not None:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes[0, 0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Frame')
        axes[0, 1].imshow(fgMOG, cmap='gray')
        axes[0, 1].set_title('MOG2 Foreground')
        axes[0, 2].imshow(fgKNN, cmap='gray')
        axes[0, 2].set_title('KNN Foreground')
        axes[1, 0].imshow(fgMOG_blur, cmap='gray')
        axes[1, 0].set_title('MOG2 Foreground (Blurred)')
        axes[1, 1].imshow(fgKNN_blur, cmap='gray')
        axes[1, 1].set_title('KNN Foreground (Blurred)')
        if bgMOG is not None:
            axes[1, 2].imshow(cv2.cvtColor(bgMOG, cv2.COLOR_BGR2RGB))
            axes[1, 2].set_title('MOG2 Background')
        for ax in axes.flat:
            ax.axis('off')
        plt.tight_layout()
        plt.show()
    else:
        print("프레임을 읽을 수 없습니다.")
        