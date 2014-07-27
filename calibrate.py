import cv2
import numpy as np
import math

def inRange(crange, fdata, bgdata):
    featinrange = sum(cv2.inRange(np.array([fdata], dtype=np.uint8), np.array(crange[0], dtype=np.uint8),
        np.array(crange[1], dtype=np.uint8))[0])/255
    bginrange = sum(cv2.inRange(np.array([bgdata], dtype=np.uint8), np.array(crange[0], dtype=np.uint8),
        np.array(crange[1], dtype=np.uint8))[0])/255
    print(len(fdata), featinrange, len(bgdata), bginrange)
    return (float(featinrange)/len(fdata), float(featinrange)/(featinrange + bginrange))

def calibrate(filename, mode, minfeat = .5, minratio = .5):
    clicks = []
    rclicks = []
    featuredata = []
    backgrounddata = []
    
    def get_click_pos(event,x,y,flags,param):
        if event==cv2.EVENT_LBUTTONDOWN:
            if len(clicks)%2 == 0:
                cv2.circle(disp, (x, y), 2, (255, 0, 0), thickness = 3)
                clicks.append((x, y))
            else:
                radius = math.sqrt((clicks[-1][0]-x)**2+(clicks[-1][1]-y)**2)
                cv2.circle(disp, clicks[-1], int(radius), (255, 0, 0), thickness = 3)
                clicks.append(radius)

        if event==cv2.EVENT_RBUTTONDOWN:
            if len(rclicks)%2 == 0:
                cv2.circle(disp, (x, y), 2, (0, 0, 255), thickness = 3)
                rclicks.append((x, y))
            else:
                radius = math.sqrt((rclicks[-1][0]-x)**2+(rclicks[-1][1]-y)**2)
                cv2.circle(disp, rclicks[-1], int(radius), (0, 0, 255), thickness = 3)
                rclicks.append(radius)

    cap = cv2.VideoCapture(filename)
    _, img = cap.read()
    disp = img.copy()
    img = cv2.GaussianBlur(img, (5,5), 10) 
    cv2.namedWindow('Calibration')
    cv2.setMouseCallback('Calibration', get_click_pos)
    cv2.imshow('Calibration', disp)
    while cv2.waitKey(10) != ord(' '):
        cv2.imshow('Calibration', disp)
    
    #cv2.destroyWindow('Calibration')

    img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    testpoints = len(clicks)/2
    rtestpoints = len(rclicks)/2
    
    for y, c in enumerate(img):
        for x, p in enumerate(c):
            for c in range(testpoints):
                if math.sqrt((clicks[c*2][0]-x)**2+(clicks[c*2][1]-y)**2) < clicks[c*2+1]:
                    featuredata.append(p)
                    break
            else:
                for c in range(rtestpoints):
                    if math.sqrt((rclicks[c*2][0]-x)**2+(rclicks[c*2][1]-y)**2) < rclicks[c*2+1]:
                        break
                backgrounddata.append(p)
    
    cv2.destroyAllWindows()
    if mode == 1:
        return [[min([x[0] for x in featuredata]), min([x[1] for x in featuredata]), min([x[2] for x in featuredata])],
                [max([x[0] for x in featuredata]), max([x[1] for x in featuredata]), max([x[2] for x in featuredata])]]

    #Seperate feature data by clustering hues
    #NOTE: Currently assumes only one color
    colors = 1
    colorgroups = [featuredata]

    #For each cluster, find the centroid
    centroids = []
    for cg in colorgroups:
        h = sum([d[0] for d in cg])/len(cg)
        s = sum([d[1] for d in cg])/len(cg)
        v = sum([d[2] for d in cg])/len(cg)
        centroids.append([h, s, v])

    print(centroids)

    print(inRange([[7, 80, 200],[25, 255, 255]], featuredata, backgrounddata))

    #from matplotlib import pyplot as pplot
    #pplot.scatter([p[0] for p in backgrounddata], [p[1] for p in backgrounddata], c='b')
    #pplot.scatter([p[0] for p in featuredata], [p[1] for p in featuredata], c='r')
    #pplot.show()

    #Expand a region around the centroid to optimize feature detection
    for c, cg in zip(centroids, colorgroups):
        colorrange = [[c[0]-2, c[1]-2, c[2]-2],[c[0]+2, c[1]+2, c[2]+2]]
        pf, ps = inRange(colorrange, featuredata, backgrounddata)
        while pf < minfeat:
            pf1, ps1 = inRange([[max(colorrange[0][0]-2,0), colorrange[0][1], colorrange[0][2]],
                [min(colorrange[1][0]+2, 180), colorrange[1][1], colorrange[1][2]]], featuredata, backgrounddata)
            pf2, ps2 = inRange([[colorrange[0][0], max(colorrange[0][1]-2, 0), colorrange[0][2]],
                [colorrange[1][0], min(colorrange[1][1]+2, 255), colorrange[1][2]]], featuredata, backgrounddata)
            pf3, ps3 = inRange([[colorrange[0][0], colorrange[0][1], max(colorrange[0][2]-2, 0)],
                [colorrange[1][0], colorrange[1][1], min(colorrange[1][2]+2, 255)]], featuredata, backgrounddata)
            if pf1 >= pf2 and pf1 >= pf3:
                colorrange[0][0] = max(colorrange[0][0]-2, 0)
                colorrange[1][0] = min(colorrange[1][0]+2, 180)
                pf = pf1
            elif pf2 >= pf1 and pf2 >= pf3:
                colorrange[0][1] = max(colorrange[0][1]-2, 0)
                colorrange[1][1] = min(colorrange[1][1]+2, 255)
                pf = pf2
            else:
                colorrange[0][2] = max(colorrange[0][2]-2, 0)
                colorrange[1][2] = min(colorrange[1][2]+2, 255)
                pf = pf3
            print(colorrange)
            t = cv2.inRange(img, np.array(colorrange[0], dtype = np.uint8), np.array(colorrange[1], dtype = np.uint8))
            cv2.imshow("Calibration", t)
    return colorrange
