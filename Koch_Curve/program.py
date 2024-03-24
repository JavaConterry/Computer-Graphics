import turtle as tt
from tkinter import *

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

    def paint(self):
        tt.clearscreen()
        tt.tracer(False)
        tt.speed(0); tt.delay(0)
        expl = {'F': f'tt.forward({length})', 
                '+': f'tt.left({self.angle})', 
                '-': f'tt.right({self.angle})', 
                'X': f'tt.forward({length})', 
                'Y': f'tt.forward({length})'}
        for v in self.axiom:
            exec(expl[v])

# class Window(Tk):
#     def __init__(self, title, geometry):
#         super().__init__()
#         self.running = True
#         self.geometry(geometry)
#         self.title(title)
#         self.protocol("WM_DELETE_WINDOW", self.destroy_window)
#         self.canvas = Canvas(self)
#         self.canvas.pack(side=LEFT, expand=True, fill=BOTH)
#         self.turtle = tt.RawTurtle(tt.TurtleScreen(self.canvas))

#     def update_window(self):
#         if self.running:
#             self.update()

#     def destroy_window(self):
#         self.running = False
#         self.destroy()


axiom = 'F--F--F'
rules = [Rule('F', 'F+F--F+F')]
angle = 60
level = int(input('level = ')) #suggested level: 5
length = 4

fractal = Fractal(axiom, rules, angle)    
fractal.calculate(level)
fractal.paint()
print('finished')
tt.update()

axiom = 'FX'
rules = [Rule('X', 'X+YF+'), Rule('Y', '-FX-Y')]  #suggested level: 14
angle = 90
level = int(input('level = '))
length = 2

fractal = Fractal(axiom, rules, angle)    
fractal.calculate(level)
fractal.paint()
print('finished')
tt.update()


axiom = 'F+F+F+F'
rules = [Rule('F', 'FF+F+F+F+FF')]
angle = 90
level = int(input('level = ')) #suggested level: 5
length = 2

fractal2 = Fractal(axiom, rules, angle)    
fractal2.calculate(level)
fractal2.paint()
print('finished')
tt.update()


tt.done()