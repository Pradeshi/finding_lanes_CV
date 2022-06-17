import cv2
import numpy as np
import matplotlib.pyplot as plot
import random

def return_canny(image):
    gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    blur=cv2.GaussianBlur(gray,(5,5),0)
    canny=cv2.Canny(blur,50,150)
    return canny

def mask_region_of_interest(image):
    height=image.shape[0]
    polygons=np.array([[(200,height),(1100,height),(550,250)]])
    mask=np.zeros_like(image)
    cv2.fillPoly(mask,polygons,255)
    masked_image=cv2.bitwise_and(image,mask)
    return masked_image

def display_lane_lines(image,lines):
    lane_image=np.zeros_like(image)
    points=[]
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2=line.reshape(4)
            points.append([int(x1),int(y1)])
            points.append([int(x2),int(y2)])
    newpoints=[points[0],points[2],points[3],points[1]]
    cv2.fillPoly(lane_image,np.int32([newpoints]), (0,255,0),20)
    return lane_image


def average_slope_intercept(image,lines):
    left_side=[]
    right_side=[]
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2=line.reshape(4)
            parameters=np.polyfit((x1,x2),(y1,y2),1)
            slope=parameters[0]
            intercept=parameters[1]
            if slope <0:
                left_side.append((slope,intercept))
            else:
                right_side.append((slope,intercept))
    left_side_average=np.average(left_side,axis=0)
    right_side_average=np.average(right_side,axis=0)
    left_line=make_cordinates(image, left_side_average)
    right_line=make_cordinates(image, right_side_average)
    return np.array([left_line,right_line])


def make_cordinates(image,line_parameters):
    slope,intercept=line_parameters
    y1=image.shape[0]
    y2=int(y1*(3/5))
    x1=int((y1-intercept)/slope)
    x2=int((y2-intercept)/slope)
    return np.array([x1,y1,x2,y2])
    
image=cv2.imread('test_image.jpg')
lane_image=np.copy(image)
canny= return_canny(lane_image)
masked_image=mask_region_of_interest(canny)
lines=cv2.HoughLinesP(masked_image, 3, np.pi/180,100,minLineLength=40,maxLineGap=10)
averaged_lines=average_slope_intercept(lane_image, lines)
lane_lines=display_lane_lines(lane_image, averaged_lines)
combined_final_image=cv2.addWeighted(lane_image, 0.85, lane_lines, 1, 1)
plot.imshow(combined_final_image)
plot.show()

capture_video=cv2.VideoCapture("test2.mp4")
while(capture_video.isOpened()):
    _, frame=capture_video.read()
    canny= return_canny(frame)
    masked_image=mask_region_of_interest(canny)
    lines=cv2.HoughLinesP(masked_image, 3, np.pi/180,100,minLineLength=40,maxLineGap=10)
    averaged_lines=average_slope_intercept(frame, lines)
    lane_lines=display_lane_lines(frame, averaged_lines)
    combined_final_image=cv2.addWeighted(frame, 0.85, lane_lines, 1, 1)
    plot.imshow(combined_final_image)
    plot.show()
    
    
