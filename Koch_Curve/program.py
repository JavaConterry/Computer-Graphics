import turtle as tt
import math

tt.tracer(False)
tt.speed(0); tt.delay(0)


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

axiom = 'FX'
# rules = [Rule('F', 'F+F--F+F')]
rules = [Rule('X', 'X+YF+'), Rule('Y', '-FX-Y')]
angle = 90
level = int(input('level = '))
length = 2

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
        expl = {'F': f'tt.forward({length})', '+': f'tt.left({self.angle})', '-': f'tt.right({self.angle})', 'X': f'tt.forward({length})', 'Y': f'tt.forward({length})'}
        for v in self.axiom:
            exec(expl[v])

fractal = Fractal(axiom, rules, angle)    
fractal.calculate(level)
fractal.paint()
print('finished')
tt.update()
tt.done()