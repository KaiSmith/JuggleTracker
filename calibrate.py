import cv2
import numpy as np

def inRange(crange, fdata, bgdata):
    featinrange = sum(cv2.inRange(np.array([fdata], dtype=np.uint8), np.array(crange[0], dtype=np.uint8),
        np.array(crange[1], dtype=np.uint8))[0])/255
    bginrange = sum(cv2.inRange(np.array([bgdata], dtype=np.uint8), np.array(crange[0], dtype=np.uint8),
        np.array(crange[1], dtype=np.uint8))[0])/255
    print(len(fdata), featinrange, len(bgdata), bginrange)
    return (float(featinrange)/len(fdata), float(featinrange)/(featinrange + bginrange))


def calibrate(filename, minfeat = .8, minratio = .8):
    clicks = []
    featuredata = []
    backgrounddata = []
    
    def get_click_pos(event,x,y,flags,param):
        if event==cv2.EVENT_LBUTTONDOWN:
            clicks.append((x, y))

    cap = cv2.VideoCapture(filename)
    _, img = cap.read()
    cv2.namedWindow('Calibration')
    cv2.setMouseCallback('Calibration', get_click_pos)
    cv2.imshow('Calibration', img)
    while cv2.waitKey(10) != ord(' '):
        pass
    
    cv2.destroyAllWindows()

    img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    testpoints = len(clicks)/2
    for c in range(testpoints):
        clicks[c*2+1] = ((clicks[c*2+1][0]-clicks[c*2][0])**2+(clicks[c*2+1][1]-clicks[c*2][1])**2)*.9#.9 is a buffer for off-center guesses
    
    for y, c in enumerate(img):
        for x, p in enumerate(c):
            for c in range(testpoints):
                if (clicks[c*2][0]-x)**2+(clicks[c*2][1]-y)**2 < clicks[c*2+1]:
                    featuredata.append(p)
                    break
            else:
                backgrounddata.append(p)

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

    #Expand a region around the centroid to optimize feature detection
    performance = []
    for c, cg in zip(centroids, colorgroups):
        colorrange = [[c[0]-1, c[1]-10, c[2]-10],[c[0]+1, c[1]+10, c[2]+10]]
        pf, ps = inRange(colorrange, featuredata, backgrounddata)
        while pf < minfeat and ps < minratio:
            if inRange([[max(colorrange[0][0]-1,0), colorrange[0][1], colorrange[0][2]],
                [min(colorrange[1][0]+1, 180), colorrange[1][1], colorrange[1][2]]], featuredata, backgrounddata)[0] >= ps:
                colorrange[0][0] = max(colorrange[0][0]-1, 0)
                colorrange[1][0] = min(colorrange[1][0]+1, 180)
            elif inRange([[colorrange[0][0], max(colorrange[0][1]-10, 0), colorrange[0][2]],
                [colorrange[1][0], min(colorrange[1][1]+10, 255), colorrange[1][2]]], featuredata, backgrounddata)[0] >= ps:
                colorrange[0][1] = max(colorrange[0][1]-10, 0)
                colorrange[1][1] = min(colorrange[1][1]+10, 255)
            elif inRange([[colorrange[0][0], colorrange[0][1], max(colorrange[0][2]-10, 0)],
                [colorrange[1][0], colorrange[1][1], min(colorrange[1][2]+10, 255)]], featuredata, backgrounddata)[0] >= ps:
                colorrange[0][2] = max(colorrange[0][2]-10, 0)
                colorrange[1][2] = min(colorrange[1][2]+10, 255)
            else:
                break
            print(colorrange)
            pf, ps = inRange(colorrange, featuredata, backgrounddata)
    return colorrange
