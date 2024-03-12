import turtle as tt
import math

tt.tracer(False)
tt.speed(0); tt.delay(0)

length = 15
level = 4


def build_strategy(level):
    axiom = 'F--F--F'
    rule = 'F+F--F+F'
    for i in range(1, level):
        new_axiom = ''
        for i in range(len(axiom)):
            if(axiom[i] == 'F'):
                new_axiom = new_axiom+rule
            else:
                # new_axiom = new_axiom.join(axiom[i])
                new_axiom = new_axiom+axiom[i]
        axiom = new_axiom

    return axiom
    

for i in range(1,5):
    print(build_strategy(i))

def paint_by_strategy(strategy):
    theta = 60
    expl = {'F': f'tt.forward({length})', '+': 'tt.left(theta)', '-': 'tt.right(theta)'}
    for v in strategy:
        exec(expl[v])


# length = 100//(length**(level-1) * (2+math.sqrt(2))**level)
# print(length)
paint_by_strategy(build_strategy(level))
tt.update()
tt.done()