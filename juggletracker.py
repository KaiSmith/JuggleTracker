import cv2, math
import cv2.cv as cv
import numpy as np
from math import sqrt
import time

class Ball:
    def __init__(self, color = (255, 0, 0), id = 0):
        '''Initializing the ball object, mostly with empty data'''
        self.id = id
        self.color = color
        self.times = []
        self.positions = []
        self.dx = []
        self.dy = []
        self.dx2 = []
        self.dy2 = []
        self.thrown = True
        self.throws = 0
        self.catches = 0
        self.hands = []
        self.lost = False
   
    def update(self, frame, center):
        '''After choosing the correct new center for the ball, this function updates the kinetics of the ball (position, velocity, etc.)'''
        self.times.append(frame)
        self.positions.append(center)
        dx, dy, times = self.dx, self.dy, self.times
        dx2, dy2 = self.dx2, self.dy2
        if len(self.times) >= 2:
            dx.append([(times[-1]+times[-2])/2.0, float(self.positions[-1][0]-self.positions[-2][0])/(times[-1]-times[-2])])
            dy.append([(times[-1]+times[-2])/2.0, float(self.positions[-1][1]-self.positions[-2][1])/(times[-1]-times[-2])])
        if len(self.times) >= 3:
            dx2.append([(dx[-1][0]+dx[-2][0])/2.0, float(dx[-1][1]-dx[-2][1])/(dx[-1][0]-dx[-2][0])])
            dy2.append([(dy[-1][0]+dy[-2][0])/2.0, float(dy[-1][1]-dy[-2][1])/(dy[-1][0]-dy[-2][0])])

        self.draw(f)
        self.status(35)

    def predict(self, dt = 1.5):
        '''This function uses the ball's kinetic data to predict its position in the next frame, which is used for tracking the balls'''
        #dt = frame-self.times[-1]+.5
        if len(self.times) <= 1:
            return self.positions[-1]
        if len(self.times) == 2:
            return (self.positions[-1][0]+self.dx[-1][1]*dt, self.positions[-1][1]+self.dy[-1][1]*dt)
        else:
            return (self.positions[-1][0]+dt*self.dx[-1][1]+self.dx2[-1][1]*dt**2/2.0,
                    self.positions[-1][1]+dt*self.dy[-1][1]+self.dy2[-1][1]*dt**2/2.0)

    def status(self, onespeed = 30):
        '''Determines if the ball was caught or thrown and acts accordingly'''
        score = 0
        for i in range(max(0, len(self.dy2)-5), len(self.dy2)):
            if self.dy2[i][1] > 0:
                score += 1
            if self.dy2[i][1] < 0:
                score -= 1
        if len(self.dx) > 0 and (score >= 3 and self.thrown == False and self.dy[-1][1] < 0 or (abs(self.dx[-1][1]) > onespeed and abs(self.dx[-1][1])>abs(self.dy[-1][1]))):
            self.thrown = True
            self.throws += 1
        elif score <= -3 and self.thrown == True:
            self.thrown = False
            self.catches += 1
            #Determines which hand the catch was in. Not very reliable yet
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
        '''Draws the center of the ball on the specified image'''
        if color == None:
            color = (255, 0, 255)
        if len(self.times)>0:
            cv2.circle(img, (self.positions[-1][0],self.positions[-1][1]), 3, color, thickness=3, lineType=8, shift=0)
    
    def debug(self):
        '''Prints some kinetic data of the ball'''
        print("times", self.times[:5])
        print("positions", self.positions[:5])
        print("dy", self.dy[:5])
        print("dy2", self.dy2[:5])

class Pattern:
    def __init__(self, balls):
        '''Initializes the pattern's data'''
        self.balls = balls
        self.active = []
        self.n = len(balls)
        self.oldsnapshot = [True, True, True]
        self.throworder = []
        self.siteswap = []

    def updateBalls(self, frame, blobs):
        '''Matches each ball with a new contour'''
        scores = []
        for n, ball in enumerate(self.active):
            d = [dist(ball.predict(), blob) for blob in blobs]
            [cv2.circle(f, (int(ball.predict(t)[0]), int(ball.predict(t)[1])), 3, (0, 255, 0), thickness = 3, lineType = 8, shift = 0)\
                    for t in [1.5]]
            scores.extend(d)

        #Matches each ball to new center by taking the overall minimum scores and matching those first.
        if len(scores) > 0:
            for d in range(min(len(self.active), len(blobs))):
                best_match = min(scores)
                n = scores.index(best_match)
                c = 100*(frame-self.active[n/len(blobs)].times[-1]+1)
                if dist(self.active[n/len(blobs)].predict(), blobs[n%len(blobs)]) < c:
                    self.active[n/len(blobs)].update(frame, blobs[n%len(blobs)])
                    blobs[n%len(blobs)] = None
                else:
                    self.active[n/len(blobs)].lost = True
                for s in range(len(scores)):
                    if s/len(blobs) == n/len(blobs) or s%len(blobs) == n%len(blobs):
                        scores[s] = 10000

            #If ball was not accounted for and other conditions are met, assume that the ball was caught and the hand is covering the ball.
            for ball in self.active:
                if  ball.times[-1] != frame and ball.thrown == True and (ball.dy[-1][1] > 0):# or\
                        #(abs(float(ball.dx[-1][1])/ball.dy[-1][1]) <= 2 and abs(ball.dx[-1][1]) >= 10)):
                    ball.thrown = False
                    ball.catches += 1

        #Chooses biggest blobs for initial values.
        blobs = filter(lambda x: x != None, blobs)
        balls = filter(lambda x: len(x.times) == 0,self.balls)
        for n, ball in enumerate(balls):
            if n < len(blobs):
                ball.update(frame, blobs[n])
                self.active.append(self.balls[n])

    def catches(self):
        '''Returns the total number of catches in the pattern'''
        return sum(map(lambda x: x.catches, self.balls))

    def updateSiteswap(self, on_catch = True):
        '''Updates the siteswap based on new catches or throws'''
        snapshot = map(lambda x: x.thrown, self.balls)
        for i in range(self.n):
            if (self.oldsnapshot[i] == True and snapshot[i] == False and on_catch == True)\
                    or (self.oldsnapshot[i] == False and snapshot[i] == True and on_catch == False):
                if i in self.throworder:
                    self.siteswap[len(self.throworder)-(self.throworder[::-1].index(i)+1)] = self.throworder[::-1].index(i)+1
                self.throworder.append(i)
                self.siteswap.append('')
        self.oldsnapshot = snapshot
        self.draw_count()
        self.draw_siteswap()

    def draw_count(self):
        '''Draws the total catch count on the top left corner of the image'''
        cv2.putText(f, 'Catches: '+str(self.catches()), (0,75), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2)

    def draw_siteswap(self):
        '''Draws the current siteswap at the bottom of the screen'''
        cv2.putText(f, 'Siteswap: '+str(''.join(map(str,self.siteswap))), (0,700), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

def dist(ball, blob):
        '''Calculates Euclidean distance'''
        return math.sqrt((ball[0]-blob[0])**2+(ball[1]-blob[1])**2)

if __name__ == '__main__':
    import sys
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option('-i', '--input', help = 'Filename of the video to be analyzed. Exclude if using webcam.',
            dest = 'input', default = 0)
    parser.add_option('-o', '--output', help = 'Filename in which to save the resulting video. Exclude if no output is desired.',
            dest = 'output', default = None)
    parser.add_option('-c', '--calibrate', help = 'Method of calibration to be used. Options are "naive" (can also use "1"),\
            or "smart" (can also use "2"). If none is used, the default orange color will be used for now.', dest = 'cal', default = None)
    parser.add_option('-b', '--balls', help = 'The numebr of balls that will be juggled. Default is 3',
            dest = 'balls', default = 3, type = 'int')

    (options, args) = parser.parse_args(sys.argv)
    
    cap = cv2.VideoCapture(options.input)
    if options.output != None:
        video = cv2.VideoWriter(options.output, cv2.cv.CV_FOURCC(*'MJPG'), 15, (1280, 720))#TODO: Get actual size
        video.open(options.output, cv2.cv.CV_FOURCC(*'MJPG'), 15, (1280, 720))
        if video.isOpened() == False:
            print('Something is wrong with the video output settings.')

    if options.cal != None:
        import calibrate
        mode = 1
        if options.cal.lower() in ['smart', '2']:
            mode = 2
        colorrange = calibrate.calibrate(options.input, mode)
        print('Color range used (in HSV with scales of 180, 255, and 255 respectively): '+ str(colorrange))
    else:
        colorrange = [[6, 128, 181], [14, 240, 253]]#[[6, 77, 180],[25, 255, 255]]
    
    balls = [Ball((255, 0, 0), 1), Ball((0, 0, 255), 2), Ball((255, 0, 255), 3)]

    p = Pattern(balls)

    frame = 0
    while(1):
        _, f = cap.read()
        if f == None:
            break
        b = cv2.GaussianBlur(f,(5,5),10)
        #Filters by color
        hsv = cv2.cvtColor(b,cv2.COLOR_BGR2HSV)
        thresh = cv2.inRange(hsv,np.array(colorrange[0],np.uint8),np.array(colorrange[1],np.uint8))
        #Attempts to remove background contours
        kernel = np.ones((5, 5), np.uint8)
        #thresh = cv2.erode(thresh, kernel, iterations = 2)
        #thresh = cv2.dilate(thresh, kernel, iterations = 3)
        t = np.copy(thresh)
        #Finds contours in thresholded image
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contours_by_area = sorted(contours, key=lambda x:cv2.contourArea(x))
        bc = contours_by_area[::-1]
        cv2.drawContours(f, bc,-1,(0,255,0),1)
        M = [cv2.moments(c)for c in bc]
        centers = [(int(m['m10']/m['m00']), int(m['m01']/m['m00'])) for m in M]
        centers = [c for c in centers if c[0]>350] #NOTE: This is only temporary until some sort of background subtraction can be implemented.
        p.updateBalls(frame, centers)
        p.updateSiteswap()
        cv2.imshow('Video', f)
        if options.output != None:
            video.write(f)
        if cv2.waitKey(5)==27:
            break
        #Play one frame per second unless holding space bar.
        if cv2.waitKey(1000)==ord(' '):
            pass
        frame += 1
    cv2.destroyAllWindows()
    cap.release()
    print('Siteswap: '+''.join(map(str, p.siteswap[:p.siteswap.index('')])))
    #Plot kinetics data for specified ball (bn = ball.id-1 for chosen ball)
    bn = 1
    from matplotlib import pyplot as pplot
    pplot.scatter([x for x in balls[bn].times], [x[1] for x in balls[bn].positions], c='r')
    pplot.scatter([x[0] for x in balls[bn].dx], [x[1] for x in balls[bn].dy], c='g')
    pplot.scatter([x[0] for x in balls[bn].dy2], [x[1] for x in balls[bn].dy2], c='b')
    pplot.show()

    ### Ideal settings for claw: onespeed = 40
    ### Ideal settings for 441 : onespeed = 35