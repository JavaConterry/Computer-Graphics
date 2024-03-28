import turtle as tt
from tkinter import *


COLORS = [(i,i,i) for i in range(0, 250, 10)]

class Rule:
    def __init__(self, base, rule):
        self.base = base
        self.rule = rule


class RuleHandler:
    def get_rule_from_base(rules, base):
        for rule in rules:
            if(rule.base == base):
                return rule.rule
        return 'NF'



class Fractal:
    def __init__(self, axiom, rules, angle):
        self.axiom = axiom
        self.rules = rules
        self.angle = angle

    def calculate(self, level):
        for i in range(1, level):
            new_axiom = ''
            for a in self.axiom:
                a_rule = RuleHandler.get_rule_from_base(self.rules, a)
                if(a_rule != 'NF'):
                    new_axiom = new_axiom+RuleHandler.get_rule_from_base(self.rules, a)
                else:
                    new_axiom = new_axiom+a
            self.axiom = new_axiom

    def paint(self, length):
        tt.setpos(20, 20)
        tt.setheading(90)
        tt.tracer(False)
        tt.speed(0); tt.delay(0)
        expl = {'F': f'tt.forward({length})', 
                '+': f'tt.left({self.angle})', 
                '-': f'tt.right({self.angle})', 
                'X': f'tt.forward({length})', 
                'Y': f'tt.forward({length})'}
        for v in self.axiom:
            exec(expl[v])



# ### star
# axiom = 'F--F--F'
# rules = [Rule('F', 'F+F--F+F')]
# angle = 60
# level = int(input('level = ')) #suggested level: 5
# length = 4

# fractal = Fractal(axiom, rules, angle)    
# fractal.calculate(level)
# fractal.paint()
# print('finished')
# tt.update()

# ### dragon
# axiom = 'FX'
# rules = [Rule('X', 'X+YF+'), Rule('Y', '-FX-Y')]  #suggested level: 14
# angle = 90
# level = int(input('level = '))
# length = 2

# fractal = Fractal(axiom, rules, angle)    
# fractal.calculate(level)
# fractal.paint()
# print('finished')
# tt.update()

### cube
tt.clearscreen()
axiom = 'F+F+F+F'
rules = [Rule('F', 'FF+F+F+F+FF')]
angle = 90
level = int(input('level = ')) #suggested level: 5
length = 3
angle_range = int(input('angle range = '))


def run_fractal(axiom, rules, angle, level, length, angle_range):
    delta_color = len(COLORS)/(2*angle_range)

    multiplyier = 5
    for iangle in range(angle - multiplyier*angle_range, angle + multiplyier*angle_range, multiplyier):
        fractal = Fractal(axiom, rules, iangle)    
        fractal.calculate(level)
        tt.colormode(255)
        tt.color(COLORS[int(delta_color*(iangle-(angle - multiplyier*angle_range))/multiplyier)])
        fractal.paint(length)
        print('finished', iangle)
        tt.update()

run_fractal(axiom, rules, angle, level, length, angle_range)

tt.done()