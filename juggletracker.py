import cv2, sys, math
import cv2.cv as cv
import numpy as np
import time

cap = cv2.VideoCapture(sys.argv[1])

class Ball:
    def __init__(self, color = (255, 0, 0), id = 0):
        self.id = id
        self.color = color
        self.times = []
        self.positions = []
        self.dx = []
        self.dy = []
        self.dx2 = []
        self.dy2 = []
        self.dx3 = []
        self.thrown = True
        self.throws = 0
        self.catches = 0
        self.hands = []
        self.lost = False

    
    def update(self, frame, center):
        self.times.append(frame)
        self.positions.append(center)
        dx, dy, times = self.dx, self.dy, self.times
        dx2, dy2, dx3 = self.dx2, self.dy2, self.dx3
        if len(self.positions) >= 2:
            #print(times[-5:])
            dx.append([(times[-1]+times[-2])/2.0, float(self.positions[-1][0]-self.positions[-2][0])/(times[-1]-times[-2])])
            dy.append([(times[-1]+times[-2])/2.0, float(self.positions[-1][1]-self.positions[-2][1])/(times[-1]-times[-2])])
        if len(self.positions) >= 3:
            dx2.append([(dx[-1][0]+dx[-2][0])/2.0, float(dx[-1][1]-dx[-2][1])/(dx[-1][0]-dx[-2][0])])
            dy2.append([(dy[-1][0]+dy[-2][0])/2.0, float(dy[-1][1]-dy[-2][1])/(dy[-1][0]-dy[-2][0])])
            #self.draw(f, (255, 0, max(min(140+10*dy2[-1][1], 255), 0)))
        if len(self.positions) >= 4:
            dx3.append([(dx2[-1][0]+dx2[-2][0])/2.0, float(dx2[-1][1]-dx2[-2][1])/(dx2[-1][0]-dx2[-2][0])])

        self.draw(f)
        self.status()

    def predict(self, dt = 1.5):
        #dt = frame-self.times[-1]+.5
        if len(self.times) == 1:
            return self.positions[-1]
        if len(self.times) == 2:
            return (self.positions[-1][0]+self.dx[-1][1]*dt, self.positions[-1][1]+self.dy[-1][1]*dt)
        else:
            #nvx, nvy = self.dx[-1][1] + self.dx2[-1][1]*dt, self.dy[-1][1] + self.dy2[-1][1]*dt
            #return (self.positions[-1][0]+nvx*dt, self.positions[-1][1]+nvy*dt)
            return (self.positions[-1][0]+dt*self.dx[-1][1]+self.dx2[-1][1]*dt**2/2.0,
                    self.positions[-1][1]+dt*self.dy[-1][1]+self.dy2[-1][1]*dt**2/2.0)

    def status(self):
        score = 0
        for i in range(max(0, len(self.dy2)-5), len(self.dy2)):
            if self.dy2[i][1] > 0:
                score += 1
            if self.dy2[i][1] < 0:
                score -= 1
        if len(self.dx) > 0 and (score >= 3 and self.thrown == False and self.dy[-1][1] < 0 or abs(self.dx[-1][1]) > 30 and abs(self.dx[-1][1])>abs(self.dy[-1][1])):
            self.thrown = True
            self.throws += 1
            if abs(self.dx[-1][1]) > 30 and score < 3:
                self.color = (255, 0, 0)
            else:
                self.color = (0, 255, 0)
        elif score <= -3 and self.thrown == True:
            self.color = (0, 0, 255)
            self.thrown = False
            self.catches += 1
            hscore = 0
            for i in range(max(0, len(self.dx2)-5), len(self.dy2)):
                if self.dx2[i][1] > 0:
                    hscore += 1
                if self.dx2[i][1] < 0:
                    hscore -= 1
            if hscore < 0:
                self.hands.append('r')
            else:
                self.hands.append('l')
        cv2.putText(f, str(self.id)+":"+str(self.catches), (self.positions[-1][0],self.positions[-1][1]), cv2.FONT_HERSHEY_SIMPLEX,
                3, (0, 255, 0), 2)

    def draw(self, img, color = None):
        if color == None:
            color =self.color
        if len(self.times)>0:
            cv2.circle(img, (self.positions[-1][0],self.positions[-1][1]), 3, color, thickness=3, lineType=8, shift=0)
    
    def debug(self):
        print "times", self.times[:5]
        print "positions", self.positions[:5]
        print "dy", self.dy[:5]
        print "dy2", self.dy2[:5]

class Pattern:
    def __init__(self, balls):
        self.balls = balls
        self.n = len(balls)
        self.oldsnapshot = [True, True, True]
        self.throworder = []
        self.siteswap = []

    def catches(self):
        return sum(map(lambda x: x.catches, self.balls))

    def update(self):
        snapshot = map(lambda x: x.thrown, self.balls)
        for i in range(self.n):
            if self.oldsnapshot[i] == False and snapshot[i] == True:
                if i in self.throworder:
                    self.siteswap[len(self.throworder)-(self.throworder[::-1].index(i)+1)] = self.throworder[::-1].index(i)+1
                self.throworder.append(i)
                self.siteswap.append('')
        self.oldsnapshot = snapshot
        self.draw_count()
        self.draw_siteswap()

    def draw_count(self):
        cv2.putText(f, 'Catches: '+str(self.catches()), (0,75), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2)

    def draw_siteswap(self):
        cv2.putText(f, 'Siteswap: '+str(''.join(map(str,self.siteswap))), (0,700), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

balls = [Ball((255, 0, 0), 1), Ball((0, 0, 255), 2), Ball((255, 0, 255), 3)]

p = Pattern(balls)

def dist(ball, blob):
    return math.sqrt((ball[0]-blob[0])**2+(ball[1]-blob[1])**2)

def updateBalls(frame, blobs, balls):
    scores = []
    for n, ball in enumerate(balls):
        if len(ball.times) == 0:
            ball.update(frame, blobs[n])
            continue
        d = [dist(ball.predict(), blob) for blob in blobs]
        [cv2.circle(f, (int(ball.predict(t)[0]), int(ball.predict(t)[1])), 3, (0, 255, 0), thickness = 3, lineType = 8, shift = 0)\
                for t in [1.5]]
        scores.extend(d)

    if len(scores) > 0:
        for d in range(min(len(balls), len(blobs))):
            b = min(scores)
            n = scores.index(b)
            if dist(balls[n/len(blobs)].predict(), blobs[n%len(blobs)]) < 100*(frame-balls[n/len(blobs)].times[-1]+1):
                balls[n/len(blobs)].update(frame, blobs[n%len(blobs)])
            else:
                balls[n/len(blobs)].lost = True
            for s in range(len(scores)):
                if s/len(blobs) == n/len(blobs) or s%len(blobs) == n%len(blobs):
                    scores[s] = 10000
        
        for ball in p.balls:
            if ball.times[-1] != frame and ball.dy[-1][1] > 0 and ball.thrown == True:
                ball.thrown = False
                ball.catches += 1

frame = 0
while(1):
    _, f = cap.read()
    if f == None:
        break
    b = cv2.GaussianBlur(f,(5,5),10)
    hsv = cv2.cvtColor(b,cv2.COLOR_BGR2HSV)
    thresh = cv2.inRange(hsv,np.array([7, 80, 200],np.uint8),np.array([25, 255, 255],np.uint8))
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.erode(thresh, kernel, iterations = 2)
    thresh = cv2.dilate(thresh, kernel, iterations = 3)
    t = np.copy(thresh)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours_by_area = sorted(contours, key=lambda x:cv2.contourArea(x))
    #bc = contours_by_area[-4:][::-1]
    bc = contours_by_area[::-1]
    cv2.drawContours(f, bc,-1,(0,255,0),1)
    M = [cv2.moments(c)for c in bc]
    centers = [(int(m['m10']/m['m00']), int(m['m01']/m['m00'])) for m in M]
    centers = [c for c in centers if c[0]>350]
    updateBalls(frame, centers, balls)
    p.update()
    cv2.imshow('Video', f)
    if cv2.waitKey(5)==27:
        break
    if cv2.waitKey(1000)==ord(' '):
        pass
    frame += 1
cv2.destroyAllWindows()
print(p.siteswap)
bn = 1
from matplotlib import pyplot as pplot
from math import sqrt
pplot.scatter([x for x in balls[bn].times], [x[1] for x in balls[bn].positions], c='r')
pplot.scatter([x[0] for x in balls[bn].dx], [x[1] for x in balls[bn].dx], c='g')
pplot.scatter([x[0] for x in balls[bn].dy2], [x[1] for x in balls[bn].dy2], c='b')#[0 if x[1]==0 else abs(10*x[1])/x[1] for x in balls[bn].dy2], c='b')
pplot.show()
