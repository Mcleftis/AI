from kanren import run, var, fact 
from kanren.assoccomm import eq_assoccomm as eq 
from kanren.assoccomm import commutative, associative

#orismos praksewn
add="add"
mul="mul"

#dhlwsh idiothtwn(prosetaristikh kai antimetathetikh)
fact(commutative, mul)
fact(commutative, add)
fact(associative, mul)
fact(associative, add)

#dhlwsh metavlhtwn
a, b=var('a'), var('b')

#dhmiourgia pattern
original_pattern=(mul, (add, 5, a), b)

#ekfraseis pros elegxo
exp1 = (mul, 2, (add, 3, 1))
exp2 = (add, 5, (mul, 8, 1))

#matching
print(run(0, (a,b), eq(original_pattern, exp1)))
print(run(0, (a,b), eq(original_pattern, exp2)))