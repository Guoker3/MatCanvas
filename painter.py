import matplotlib.pyplot as plt
import numpy as np
import math
import copy
from numpy.random import randint
import sys
sys.path.append("../d_apyori/")
import aprioriRule as ar
from queue import Queue
#the arribute name of class Canvas and class Element shouldn't be same
##TODO class translator:        to connect the data,for convience use the direct temporary
class ShapeGenerator:
    def __init__(self):
        # the color should be between vacancy and blank
        self.cmap = plt.cm.cubehelix
        self.vacancy = 0  # black,where out of screen
        self.blank = 256  # white,where is the screen have nothing to show
        self.vacancyPoint = copy.deepcopy(self.vacancy) #where is a point of the generated Shape,can't equal to zero
        self.pos=None

    def generateShape(self,shape,maxHeight,maxWidth,args=None):
        canvas=np.zeros((maxHeight,maxWidth))
        for h in range(maxHeight):
            for w in range(maxWidth):
                canvas[h][w]=self.blank

        if shape == "rectangle":
            for h in range(maxHeight):
                for w in range(maxWidth):
                    canvas[h][w]=self.vacancyPoint

        elif shape == "isosceles triangle":
            for h in range(maxHeight):
                for w in range(maxWidth):
                    if abs(w+1-maxWidth/2)/(h+1)<=((maxWidth/2)/maxHeight):
                        canvas[h][w]=self.vacancyPoint

        elif shape == "upper triangle":
            for h in range(maxHeight):
                for w in range(maxWidth):
                    if w<=h:
                        canvas[h][w]=self.vacancyPoint

        elif shape == "lower triangle":
            for h in range(maxHeight):
                for w in range(maxWidth):
                    if w>=h:
                        canvas[h][w]=self.vacancyPoint

        elif shape == "ellipse":
            if maxWidth>=maxHeight:
                a=maxWidth/2
                b=maxHeight/2
                c=math.sqrt(a**2-b**2)
                for h in range(maxHeight):
                    y=abs(maxHeight/2-h)
                    for w in range(maxWidth):
                        x =abs(maxWidth/2-w)
                        t=math.sqrt((c - x) ** 2 +y**2) + math.sqrt((c + x) ** 2+y**2)
                        if t <= 2*a:
                            canvas[h][w]=self.vacancyPoint
            else:
                a=maxHeight/2
                b=maxWidth/2
                c=math.sqrt(a**2-b**2)
                for w in range(maxWidth):
                    y=abs(maxWidth/2-w)
                    for h in range(maxHeight):
                        x =abs(maxHeight/2-h)
                        t=math.sqrt((c - x) ** 2 +y**2) + math.sqrt((c + x) ** 2+y**2)
                        if t <= 2*a:
                            canvas[h][w]=self.vacancyPoint
        return canvas

    def showShape(self,shapeCanvas,title="shape"):
        plt.matshow(shapeCanvas,cmap="gist_yarg")
        plt.title(title)
        plt.show()

class Canvas(ShapeGenerator):
    def __init__(self,height,width,ID=-1):
        ShapeGenerator.__init__(self)
        self.CanvasWidth=width
        self.CanvasHeight=height
        self.basemat=None
        self.canvas=None
        self.clean=None
        self.canvasTid=0
        self.ID=ID
        #self.remain balnk

    #generator the max cubic storage mat
    def prepareBoard(self):
        self.basemat = np.zeros((self.CanvasHeight+1,self.CanvasWidth))
        for h in range(self.CanvasHeight):
            for w in range(self.CanvasWidth):
                self.basemat[h][w]=self.blank   #default to be paintful
        #to control and show color reflect in matshow
        for w in range(self.CanvasWidth):
            self.basemat[self.CanvasHeight][w]=self.vacancy
        self.basemat[self.CanvasHeight][self.CanvasWidth-1]=self.blank
        self.canvas=copy.deepcopy(self.basemat)

    def tailorCanvas(self,shapeMat,y,x):#start point left-top (x,y)
        hh=shapeMat.shape[0]
        ww=shapeMat.shape[1]
        for h in range(hh):
            for w in range(ww):
                if shapeMat[h][w]==self.vacancyPoint and (y+h)<self.CanvasHeight and (x+w)<self.CanvasWidth:
                    self.canvas[y+h][x+w]=self.vacancy
        return (y,x)

    def paintElement(self,element,y,x):
        elementMat=element.mat
        hh = elementMat.shape[0]
        ww = elementMat.shape[1]
        for h in range(hh):
            for w in range(ww):
                if elementMat[h][w] != element.blank  and (y + h) < self.CanvasHeight and (x + w) < self.CanvasWidth:
                    if self.canvas[y + h][x + w]==self.vacancy:
                        print("drop into the vacancy")
                    elif self.canvas[y + h][x + w]!=self.blank:
                        print("cover other element")
                    else:
                        self.canvas[y + h][x + w] = elementMat[h][w]
        return (y,x)    #return position to Element sample

    def cleanElement(self,element,y,x):
        elementMat=element.mat
        hh = elementMat.shape[0]
        ww = elementMat.shape[1]
        for h in range(hh):
            for w in range(ww):
                if elementMat[h][w] != element.blank  and (y + h) < self.CanvasHeight and (x + w) < self.CanvasWidth:
                    if self.canvas[y + h][x + w]==self.vacancy:
                        print("drop into the vacancy")
                    else:
                        self.canvas[y + h][x + w] = self.blank
        return (y,x)    #return position to Element sample

    def showCanvas(self,title="canvas"):
        plt.matshow(self.canvas,cmap=self.cmap)
        plt.title(title)
        plt.savefig("../save/"+str(self.ID)+"_"+str(self.canvasTid)+"_"+"Canvas.png")
        self.canvasTid+=1
        plt.show()



    def generateRandomCanvasForTest(self):
        self.prepareBoard()
        for i in range(randint(2,5)):
            shapes = ["rectangle", "isosceles triangle", "upper triangle", "lower triangle", "ellipse"]
            shapeName = shapes[randint(len(shapes))]
            dig = self.generateShape(shapeName, randint(self.CanvasHeight/6,self.CanvasHeight/3), randint(self.CanvasWidth/6,self.CanvasWidth/3))
            self.tailorCanvas(dig, randint(self.CanvasHeight/2),randint(self.CanvasWidth/2))
        for i in range(randint(1,2)):
            shapeName = shapes[randint(len(shapes))]
            dig = self.generateShape(shapeName, randint(self.CanvasHeight / 3, self.CanvasHeight / 1.5),
                                     randint(self.CanvasWidth / 3, self.CanvasWidth / 1.5))
            self.tailorCanvas(dig, randint(self.CanvasHeight/2,self.CanvasHeight/1.5), randint(self.CanvasWidth/2,self.CanvasWidth/1.5 ))


class Element(ShapeGenerator):

    def __init__(self,type="default"):
        ShapeGenerator.__init__(self)
        self.type=type
        self.pos=None #(y,x) returned by Canvas sample
        self.hasPainted=False
        self.mat=None
        self.shapeMat=None
        self.height=None
        self.width=None
        self.shapeName=None
        self.feature=dict()
        self.attachElement=None
        self.attachDistance=None
        if type=="default":
            pass
        else:
            raise Exception("ELEMENT for paint TYPE UNKOWN")
    def generateElementFromPicture(self,rawPicture):
        pass

    def generateRandomElementForTest(self,canvas):
        maxHeight=canvas.CanvasHeight
        maxWidth=canvas.CanvasWidth
        shapes=["rectangle","isosceles triangle","upper triangle","lower triangle","ellipse"]
        shapeName=shapes[randint(len(shapes))]
        self.height=randint(int(maxHeight/10)+1,int(maxHeight/3)+1)
        self.width=randint(int(maxWidth/10)+1,int(maxWidth/3)+1)
        self.shapeMat=self.generateShape(shapeName,self.height,self.width)
        self.mat=np.zeros((self.height,self.width))
        for h in range(self.height):
            for w in range(self.width):
                self.mat[h][w]=self.blank
        #random color
        for h in range(self.height):
            for w in range(self.width):
                if self.shapeMat[h][w]==self.vacancyPoint:
                    self.mat[h][w]=randint(self.vacancy+1,self.blank)
        self.shapeName=shapeName

    def showElement(self,title="element"):
        tempMat = np.zeros((self.height+1,self.width))
        for h in range(self.height):
            for w in range(self.width):
                tempMat[h][w]=self.mat[h][w]   #default to be paintful
        #to control and show color reflect in matshow
        for w in range(self.width):
            tempMat[self.height][w]=self.vacancy
        tempMat[self.height][self.width-1]=self.blank
        plt.matshow(tempMat,cmap=self.cmap)
        plt.title(title)
        plt.show()

class Position():
    def __init__(self,y,x,canvas=None):
        self.canvas=canvas
        self.y=y
        self.x=x

        self.feature=dict()

    def extractFeature(self,feature):
        if feature=="lineNumber":
            self.feature["lineNumber"]=self.y/self.canvas.CanvasHeight

class Painter(Canvas,Element,ar.rules):
    def __init__(self,height,width,allHeaders,ID=-1):
        Element.__init__(self)
        Canvas.__init__(self,height,width,ID)
        ar.rules.__init__(self,allHeaders)
        self.paintedElement=dict()
        self.waitingPaint=Queue()
        self.waitingAttach=Queue()

    def getRandomMatKernal(self,element):
        pointList=list()
        retPointList=list()
        #count total pixel number
        for h in range(element.height):
            for w in range(element.width):
                if element.mat[h][w] != element.blank:
                    pointList.append((h,w))
        #pickPoint
        count=len(pointList)
        stepNumber=10
        step=int(count/10)
        for i in range(stepNumber):
            retPointList.append(pointList[step*i+randint(step)])
        return retPointList

    def checkIsBlank(self,element,pos):
        y=pos.y
        x=pos.x
        for h in range(element.height):
            for w in range(element.width):
                if element.mat[h][w]!=element.blank:
                    try:
                        if self.canvas[h+y][w+x]!=self.blank:
                            return False
                    except Exception:
                        return False
        return True

    def searchPossibleBlank(self,element):
        kernel=self.getRandomMatKernal(element)
        possiblePosition=list()
        maxHorizonStep=self.CanvasHeight/3
        maxVerticalStep=self.CanvasWidth/3
        y=0
        while(y+element.height<self.CanvasHeight):
            x=0
            while (x+element.width<self.CanvasWidth):
                flag=True
                for p in kernel:
                    if self.canvas[y+p[0]][x+p[1]]!=self.blank:
                        flag=False
                if (flag):
                    possiblePosition.append(Position(y,x,self))
                x+=randint(5,maxHorizonStep)
            y+=randint(5,maxVerticalStep)
        return possiblePosition


    def searchRules(self,feature):
        self.addRulesFromPickle("rules_LnCv")#"lineNumber" "colorVariety"
        self.addRulesFromPickle("rules_EdCt")#"embeddedDepth"  "contrast"
        # for rule in rl.rules:
        fr = self.searchFeatureLeft(feature)
        fr = self.sortRules(fr,amount = -1)
        #print(rl.showFeatureName(rule, allHeaders))
        return fr

    def extractElement(self,element):
        #colorVariety
        possibleColor=element.blank - element.vacancy
        flag=np.zeros(possibleColor)
        for h in range(element.height):
            for w in range(element.width):
                color=element.mat[h][w]
                if color!=element.blank:
                    flag[int(color)]=1
        element.feature["colorVariety"]=sum(flag)/possibleColor


    def compareElementAndRules(self,elementFeature,element,posFeature,pos):
        usefulRules = list()
        if elementFeature=="colorVariety":
            rules=self.searchRules(elementFeature)
            for rule in rules:
                for v in list(rule[0][0]):
                    if self.totalHeader[int(v/3)]==elementFeature:
                        if elementFeature not in element.feature:
                            self.extractElement(element)
                        p=abs(v-int(v)-element.feature[elementFeature])
                        if p<0.5:
                            if self.comparePosAndRule(posFeature, pos,rule):
                                if self.checkIsBlank(element, pos):
                                    return True
            return False

    def comparePosAndRule(self, posFeature, pos,rule):
        if posFeature=="lineNumber":
            for v in list(rule[0][1]):
                if self.totalHeader[int(v/3)]==posFeature:
                    if posFeature not in pos.feature:
                        pos.extractFeature(posFeature)
                    p=abs(v-int(v)-pos.feature[posFeature])
                    if p<0.5:
                        return True
        return False

    def cleanCanvas(self,element):
        y=element.pos.y
        x=element.pos.x
        self.cleanElement(element,y,x)
        self.paintedElement.pop(element)

    def pushElement(self,element,show=False):
        pb=self.searchPossibleBlank(element)
        for pos in pb:
            if self.compareElementAndRules(elementFeature="colorVariety",posFeature="lineNumber",element=element,pos=pos):
                    self.paintElement(element,pos.y,pos.x)
                    self.paintedElement[element]=pos
                    print("good shoot")
                    self.pos=pos
                    if show == True:
                        self.showCanvas("push independence")
                    return True
        print("no good space for that element")
        for pos in pb:
            if self.checkIsBlank(element,pos):
                self.paintElement(element,pos.y,pos.x)
                self.paintedElement[element] = pos
                self.pos=pos
                if show==True:
                    self.showCanvas("push independence")
                return True
        print("and no spare random space for it,too")
        return False

    def pushElementAttached(self,element,show=False):
        y=element.attachDistance.y
        x=element.attachDistance.x
        origin=self.paintedElement[element.attachElement]
        y=y+origin.y
        x=x+origin.x
        pos=Position(y,x,self.canvas)
        if self.checkIsBlank(element,pos):
            self.paintElement(element,y,x)
            self.paintedElement[element]=pos
            if show == True:
                self.showCanvas("push attach")
            self.pos=pos
            return True
        else:
            print("attach failed")
            return False

    def pushElementIndependence(self,element):
        self.waitingPaint.put(element)

    def pushElementAttach(self,element):
        self.waitingAttach.put(element)


    def working(self):
        while(True):
            if (not self.waitingPaint.empty()):
                element=self.waitingPaint.get()
                if(not self.pushElement(element,show=True)):
                    if self.paintedElement != dict():
                        self.waitingPaint.put(element)
                        toClean=list(self.paintedElement.keys())[randint(len(self.paintedElement.keys()))]
                        toClean.pos=self.paintedElement[toClean]
                        self.cleanCanvas(toClean)
                        self.showCanvas("independence clean")
            if(not self.waitingAttach.empty()):
                element=self.waitingAttach.get()
                if element.attachElement in self.paintedElement.keys():
                    if  (not self.pushElementAttached(element,show=True)):
                        element.attachElement.pos=self.paintedElement[element.attachElement]
                        self.cleanCanvas(element.attachElement)
                        self.waitingAttach.put(element)
                        self.showCanvas("attach clean")
                elif self.waitingPaint.empty():
                    pass
                else:
                    self.waitingAttach.put(element)
            if (self.waitingPaint.empty() and self.waitingAttach.empty()):
                print("all done")
                inp=input()
                if inp =="quit":
                    print("exist")
                    break




if __name__ == "__main__":
    #mat = np.arange(64*64).reshape(64, 64)
    #mat = np.array([[1,1,1],[1,1,-1],[1.5,2,2]])
    #plt.matshow(mat,cmap=plt.cm.cubehelix)
    #plt.show()

    #cv=Canvas(160,100)
    #cv.generateRandomCanvasForTest()
    #cv.showCanvas()
    #ellipse=cv.generateShape("ellipse",50,30)
    #cv.showShape(ellipse)
    #cv.prepareBoard()
    #cv.tailorCanvas(ellipse,50,50)
    #cv.showCanvas(title="after tailor")

    #el=Element()
    #el.generateRandomElementForTest(cv)
    #el.showElement()
    #cv.paintElement(el,10,10)
    #cv.showCanvas(title="after paint")

    allHeaders = ["embeddedDepth", "lineNumber", "imgWidth", "imgHeight", "widthHeightRatio", "red", "green",
                  "blue", "colorVariety", "contrast", "levelDistanceLowRatio", "levelDistanceHighRatio",
                  "levelSimiliarDistanceLowRatio", "levelSimiliarDistanceHighRatio", "verticalZeroRatio",
                  "verticalMinusRatio", "verticalPositiveRatio", "verticalSimiliarZeroRatio",
                  "verticalSimiliarMinusRatio", "verticalSimiliarPositiveRatio", "horizonDistanceCloserRatio",
                  "horizonDistanceFatherRatio", "horizonDistanceInFoundLevelCloserRatio",
                  "horizonDistanceInFoundLevelFatherRatio", "childNumber", "childTagNumber", "siblingNumber",
                  "siblingTagNumber", "uncleNumber", "uncleTagNumber"]
    pt=Painter(100,70,allHeaders)
    pt.generateRandomCanvasForTest()
    pt.showCanvas(title="canvas")

    #el.generateRandomElementForTest(pt)
    #el.showElement(title="element")
    #pt.pushElement(el)
    #pt.showCanvas()

    q=Queue()
    for i in range(3):
        el=Element()
        el2=Element()
        el.generateRandomElementForTest(pt)
        #el.showElement(title="element")
        pt.pushElementIndependence(el)
        q.put(copy.deepcopy(el))
        #pt.showCanvas()

        el2.generateRandomElementForTest(pt)
        #el.showElement(title="element")
        el2.attachElement=el
        pos=Position(el.height,0,pt.canvas)
        el2.attachDistance=pos
        pt.pushElementAttach(el2)



    for i in range(10):
        el=Element()
        el.generateRandomElementForTest(pt)
        #el.showElement(title="element")
        pt.pushElementIndependence(el)

    pt.working()