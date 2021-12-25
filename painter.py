import matplotlib.pyplot as plt
import numpy as np
import math
import copy
from numpy.random import randint

class ShapeGenerator:
    def __init__(self):
        # the color should be between vacancy and blank
        self.cmap = plt.cm.cubehelix
        self.vacancy = 0  # black,where out of screen
        self.blank = 256  # white,where is the screen have nothing to show
        self.vacancyPoint = copy.deepcopy(self.vacancy) #where is a point of the generated Shape,can't equal to zero

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
    def __init__(self,height,width):
        ShapeGenerator.__init__(self)
        self.width=width
        self.height=height
        self.basemat=None
        self.canvas=None
        self.clean=None
        #self.remain balnk

    #generator the max cubic storage mat
    def prepareBoard(self):
        self.basemat = np.zeros((self.height+1,self.width))
        for h in range(self.height):
            for w in range(self.width):
                self.basemat[h][w]=self.blank   #default to be paintful
        #to control and show color reflect in matshow
        for w in range(self.width):
            self.basemat[self.height][w]=self.vacancy
        self.basemat[self.height][self.width-1]=self.blank

        self.canvas=copy.deepcopy(self.basemat)

    def tailorCanvas(self,shapeMat,y,x):#start point left-top (x,y)
        hh=shapeMat.shape[0]
        ww=shapeMat.shape[1]
        for h in range(hh):
            for w in range(ww):
                if shapeMat[h][w]==self.vacancyPoint and (y+h)<self.height and (x+w)<self.width:
                    self.canvas[y+h][x+w]=self.vacancy


    def paintElement(self,element,y,x):
        elementMat=element.mat
        hh = elementMat.shape[0]
        ww = elementMat.shape[1]
        for h in range(hh):
            for w in range(ww):
                if elementMat[h][w] != element.blank  and (y + h) < self.height and (x + w) < self.width:
                    self.canvas[y + h][x + w] = elementMat[h][w]
        return (y,x)    #return position to Element sample

    def showCanvas(self,title="canvas"):
        plt.matshow(self.canvas,cmap=self.cmap)
        plt.title(title)
        plt.show()

    def cleanCanvas(self,elementMat):
        def paintElement(self, elementMat, y, x):
            hh = elementMat.shape[0]
            ww = elementMat.shape[1]
            for h in range(hh):
                for w in range(ww):
                    if elementMat[h][w] == self.vacancyPoint and (y + h) < self.height and (x + w) < self.width:
                        self.canvas[y + h][x + w] = elementMat[h][w]

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
        if type=="default":
            pass
        else:
            raise Exception("ELEMENT for paint TYPE UNKOWN")
    #def format(self,rawPicture):
    #    pass
    def generateRandomElementForTest(self,canvas):
        maxHeight=canvas.height
        maxWidth=canvas.width
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

class Painter(Canvas,Element):
    def __init__(self):
        self.paintedElementList=list()
        self.waitingPaintList=list()

    def checkIsBlank(self):
        pass

    def searchPossibleBlank(self):
        pass

    def searchRules(self):
        pass

    def pushElement(self):
        pass

    def removeElement(self):
        pass

def canvasOne():
    pass

def canvasTwo():
    pass

if __name__ == "__main__":
    #mat = np.arange(64*64).reshape(64, 64)
    #mat = np.array([[1,1,1],[1,1,-1],[1.5,2,2]])
    #plt.matshow(mat,cmap=plt.cm.cubehelix)
    #plt.show()

    cv=Canvas(160,100)
    ellipse=cv.generateShape("ellipse",50,30)
    cv.showShape(ellipse)

    cv.prepareBoard()
    cv.showCanvas()

    cv.tailorCanvas(ellipse,50,50)
    cv.showCanvas(title="after tailor")

    el=Element()
    el.generateRandomElementForTest(cv)
    el.showElement()
    cv.paintElement(el,10,10)
    cv.showCanvas(title="after paint")