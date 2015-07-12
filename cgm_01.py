import seaborn as sns
import pylab as pl
#from plotting import load_experiments
import numpy as np
import pandas as pd
import scipy.io as scio
import sys
sys.path.append('../VentIPyN/Experiments/')
from models.covFunctions import *

from venture import shortcuts

sys.path.append('../VentIPyN/SPs/')
import venture.lite.types as t
from venture.lite.function import VentureFunction
import gp_der
import csv
from models.tools import array
from kernel_interpreter import GrammarInterpreter
from subset import Subset
from venture.lite.builtin import typed_nr
figlength = 40
figheigth = 10


sns.set(font_scale=3)


patient_number = 5
steps = "1"

for i in range(1, len(sys.argv)):
    if str(sys.argv[i]) == "-s":  # structure posterior
        steps = str(sys.argv[i + 1])
    if str(sys.argv[i]) == "-p":  # structure posterior
        patient_number = str(sys.argv[i + 1])



mat_contents =scio.loadmat("../data/glucose"+str(patient_number))
X = mat_contents['Glucose']


print(np.count_nonzero(X))
print(X.shape)


X = X[np.nonzero(X)]
#fig = plt.figure(figsize=(figlength,figheigth), dpi=200)
#pl.plot(X)


def array(xs):
  return t.VentureArrayUnboxed(np.array(xs),  t.NumberType())

def makeObservations(x,y,gp_str='(gp '):
    xString = genSamples(x,gp_str)
    ripl.observe(xString, array(y))

def genSamples(x,gp_str='(gp '):
    sampleString=gp_str+' (array '
    for i in range(len(x)):
        sampleString+= str(x[i]) + ' '
    sampleString+='))'
    #print(sampleString)
    return sampleString
    
ripl = shortcuts.make_lite_church_prime_ripl()

ripl.bind_foreign_sp("make_gp_part_der",gp_der.makeGPSP)
ripl.bind_foreign_sp("covfunc_interpreter",typed_nr(GrammarInterpreter(), [t.AnyType()], t.AnyType())) 
ripl.bind_foreign_sp("subset",typed_nr(Subset(), [t.ListType(),t.IntegerType()], t.ListType()))   

ripl.assume('make_const_func', VentureFunction(makeConstFunc, [t.NumberType()], constantType))
ripl.assume('zero', "(apply_function make_const_func 0)")
ripl.assume("func_times", makeLiftedMult(lambda x1, x2: np.multiply(x1,x2)))
ripl.assume("func_plus", makeLiftedAdd(lambda x1, x2: x1 + x2))

ripl.assume('make_linear', VentureFunction(makeLinear, [t.NumberType()], t.AnyType("VentureFunction")))
ripl.assume('make_periodic', VentureFunction(makePeriodic, [t.NumberType(), t.NumberType(), t.NumberType()], t.AnyType("VentureFunction")))
ripl.assume('make_se',VentureFunction(makeSquaredExponential,[t.NumberType(), t.NumberType()], t.AnyType("VentureFunction")))
ripl.assume('make_rq',VentureFunction(makeRQ, [t.NumberType(), t.NumberType(), t.NumberType()], t.AnyType("VentureFunction")))

    
ripl.assume('a','(tag (quote parameter) 0 (log  (uniform_continuous  0 5)))')
ripl.assume('sf1','(tag (quote parameter) 1 (log (uniform_continuous  0 5 )))')
ripl.assume('sf2',' (tag (quote parameter) 2 (log (uniform_continuous  0 5 )))')
ripl.assume('p',' (tag (quote parameter) 3 (log (uniform_continuous  0.01 5)))')
ripl.assume('l',' (tag (quote parameter) 4 (log (uniform_continuous  0 5)))')

ripl.assume('l1',' (tag (quote parameter) 5 (log (uniform_continuous  0 5)))')
ripl.assume('l2',' (tag (quote parameter) 6 (log (uniform_continuous  0 5)))')
ripl.assume('sf_rq','(tag (quote hypers) 7 (log (uniform_continuous 0 5)))')
ripl.assume('l_rq','(tag (quote hypers) 8 (log (uniform_continuous 0 5)))')
ripl.assume('alpha','(tag (quote hypers)9 (log (uniform_continuous 0 5)))')
ripl.assume('sf',' (tag (quote parameter) 10 (log (uniform_continuous  0 5)))')

ripl.assume('lin1', "(apply_function make_linear a   )")
ripl.assume('per1', "(apply_function make_periodic l  p  sf ) ")
ripl.assume('se1', "(apply_function make_se sf1 l1)")
ripl.assume('se2', "(apply_function make_se sf2 l2)")
ripl.assume('rq', "(apply_function make_rq l_rq sf_rq alpha)")


ripl.assume('interp','covfunc_interpreter')

# Grammar
ripl.assume('cov_compo',"""
 (lambda (l ) 
    (if (lte ( size l) 1) 
         (first l)
             (if (flip) 
                 (apply_function func_plus (first l) (cov_compo (rest l)))
                 (apply_function func_times (first l) (cov_compo (rest l)))
        )
))
""")
ripl.assume('n_in','(tag (quote hyper) 0 (uniform_discrete 1 8))')
ripl.assume('s','(tag (quote hyper) 1 (subset (list lin1 per1 se1 se2 rq) 2))')
ripl.assume('cov','(tag (quote hyper) 2 (cov_compo s))')

ripl.assume('gp',"""(tag (quote model) 0
                    (make_gp_part_der zero cov))""")


makeObservations([i for i in range(X.shape[0])],X)
print("Structure before Inference")
print(ripl.sample('(interp cov )'))
ripl.infer("(repeat "+steps+" (do (mh (quote hyper) one 1) (mh (quote parameter) one 5)))")
print("Structure after Inference")
print(ripl.sample('(interp cov )'))

fig = pl.figure(figsize=(figlength,figheigth), dpi=200)

for i in range(200):
    xpost= np.random.uniform(0,300,100)
    sampleString=genSamples(xpost)
    ypost = ripl.sample(sampleString)
    yp = [y_temp for (x,y_temp) in sorted(zip(xpost,ypost))]
    pl.plot(sorted(xpost),yp,c="red",alpha=0.1,linewidth=2)

pl.plot([i for i in range(X.shape[0])],X,linewidth=2.0,c="blue")
fig.savefig('/home/ulli/Dropbox/cgm/CGM_patient_'+str(patient_number)+'_mh_'+steps+'.svg', dpi=fig.dpi)
fig.savefig('/home/ulli/Dropbox/cgm/CGM_patient_'+str(patient_number)+'_mh_'+steps+'.png', dpi=fig.dpi)

