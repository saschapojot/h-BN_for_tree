from sympy import *



quarter=Rational(1,4)
half=Rational(1,2)
quarter3=Rational(3,4)
half3=Rational(3,2)
def F0(x,y,z):
    return x,y,z

def F1(x,y,z):
    return half-x,half+y,half+z

def F2(x,y,z):
    return x,half-y,z

def F3(x,y,z):
    return half+x,y,half-z

def F4(x,y,z):
    return -x,-y,-z

def F5(x,y,z):
    return half+x,half-y,half-z

def F6(x,y,z):
    return -x,half+y,-z

def F7(x,y,z):
    return half-x,-y,half+z

xGd,yGd=symbols("xGd,yGd",cls=Symbol,real=True)

zGd=quarter


xO1,yO1=symbols("xO1,yO1",cls=Symbol,real=True)

xO2,yO2,zO2=symbols("xO2,yO2,zO2",cls=Symbol,real=True)


xO1Next=half+xO1
yO1Next= yO1
zO1Next=quarter
# pprint(xO1Next)
# pprint(yO1Next)
# pprint(zO1Next)
xO1Next,yO1Next,zO1Next=F7(xO1Next,yO1Next,zO1Next)
pprint(xO1Next)
pprint(yO1Next)
pprint(zO1Next)