# Each node in the CPPN is like a typical neural net node, but instead of the funciton being a threshold,
# it can be any arbitrary function we think up. Each CPPN will 'end' with a single output node, which can
# have its eval() called to evaulate the entire CPP

# The inputs_dict will hold the actual input values needed to evaulate the compuatations. Some special atom, 'input'
# atoms will 'bottom out' on values that are held in the input_dict. So that they can then bubble back up
# as nodes are evaluated.
import math, ipdb
from PIL import Image

class CPPNAtom:
    def __init__(self, t):
        self.type = t
        self.inputs  = []
        self.weights = []

    def __str__(self):
        return "node type: " + self.type + " input: " + str(self.inputs)

    def eval(self, input_dict):
        value= 0
        ret = 0
        for i in range(len(self.inputs)):
            in_node   = self.inputs[i]
            in_weight = self.weights[i]
            value += in_node.eval(input_dict)
        
        # Here's where you'd define the actual functions to be used.
        # could do this with some classes and inheritance, but this 
        # will do for a start.
        if   self.type == "sine":
            ret = math.sin(value)
        if   self.type == "cosine":
            ret = math.cos(value)
        elif self.type == "gauss":
            # TODO - gaussian
            ret = value*1
        elif self.type == "abs":
            ret = math.fabs(value)
        
        return ret*in_weight

class CPPNInputAtom(CPPNAtom):
    # TODO - if the type is 'input', then you search the
    #        dict for the value corresponding to the 'tag'
    #        or name of the input. 
    #        Needs to somehow handle the possible error of looking for a nonexistant input tag/name
    def __init__(self, tag):
        self.tag = tag

    def __str__(self):
        return "CPPNInputAtom " + self.tag
    
    def eval(self, input_dict):
        return input_dict[self.tag]
    
# When the above is done, then just need to create a method for filling in random trees. I can borrow a lot
# From work in GP to do this.

# The eval done above will be decidedly NOT lazy and therefore be a little wasteful, but who gives a shit.
inputs = {'x': 2, 'y': 64}

A = CPPNAtom("abs")
B = CPPNAtom("cosine")

A.inputs = [B]
A.weights = [1.0]

X = CPPNInputAtom('x')
Y = CPPNInputAtom('y')

B.inputs = [X, Y]
B.weights = [255,255]


size = (64, 64)
mode = "RGB"

img = Image.new(mode, size)

for x in range(64):
    for y in range(64):
        coor = {'x': x, 'y': y}
        Aout = A.eval(coor)
        value = int(Aout)
        rgbVal = (value, value, value)

        img.putpixel((x,y), rgbVal)

img.save("cppn.jpg")