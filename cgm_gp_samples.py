import seaborn as sns
import pylab as pl
#from plotting import load_experiments
import numpy as np
import pandas as pd
import scipy.io as scio
import sys
import itertools
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

import os
def touch(path,name):
    with open(path+name,'a'):
        os.utime(path, None)
import seaborn as sns
import pandas as pd
import scipy.io as scio
import pylab as pl


import matplotlib.pyplot as plt


number_predictive_samples = 200
number_curves = 100
steps = "4000"
path = '/home/ulli/Dropbox/cgm/'
alpha_value = 0.008

for i in range(1, len(sys.argv)):
    if str(sys.argv[i]) == "-s":  # steps mh
        steps = str(sys.argv[i + 1])
    if str(sys.argv[i]) == "-p":  # path
        patient = str(sys.argv[i + 1])
    if str(sys.argv[i]) == "-n":  # number predictions
        number_predictive_samples = int(sys.argv[i + 1])
    if str(sys.argv[i]) == "-c":  # number predictions
        number_curves = int(sys.argv[i + 1])
    if str(sys.argv[i]) == "-a":  # number predictions
        alpha_value = float(sys.argv[i + 1])
    if str(sys.argv[i]) == "--fl":  # number predictions
        figlength = int(sys.argv[i + 1])
    if str(sys.argv[i]) == "--fh":  # number predictions
        figheigth = int(sys.argv[i + 1])
    if str(sys.argv[i]) == "--fs":  # number predictions
        font_size = float(sys.argv[i + 1])




#fig = plt.figure(figsize=(figlength,figheigth), dpi=200)
#pl.plot(X)
def get_time_stamps(patient):
    with open ('/home/ulli/data/CGM_data-'+patient+'.csv', "r") as myfile:
        data=myfile.readlines()
    data = data[1:]
    time_stamps = []
    for line in data:
        time_stamps.append(line.split(',')[0])
    return list(reversed(time_stamps))

def array(xs):
  return t.VentureArrayUnboxed(np.array(xs),  t.NumberType())

def makeObservations(x,y,ripl,gp_str='(gp '):
    xString = genSamples(x,gp_str)
    ripl.observe(xString, array(y))

def genSamples(x,ripl,gp_str='(gp '):
    sampleString=gp_str+' (array '
    for i in range(len(x)):
        sampleString+= str(x[i]) + ' '
    sampleString+='))'
    #print(sampleString)
    return sampleString
def run_cgm_experiment(patient_number):
    mat_contents =scio.loadmat("../data/glucose"+patient_number)
    X = mat_contents['Glucose']

    X = X[np.nonzero(X)]


    ripl = shortcuts.make_lite_church_prime_ripl()

    ripl.bind_foreign_sp("make_gp_part_der",gp_der.makeGPSP)
    ripl.bind_foreign_sp("covfunc_interpreter",typed_nr(GrammarInterpreter(), [t.AnyType()], t.AnyType()))
    ripl.bind_foreign_sp("subset",typed_nr(Subset(), [t.ListType(),t.SimplexType()], t.ListType()))

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
    ripl.assume('p',' (tag (quote parameter) 2 (log (uniform_continuous  0.01 5)))')
    ripl.assume('l',' (tag (quote parameter) 3 (log (uniform_continuous  0 5)))')

    ripl.assume('l1',' (tag (quote parameter) 4 (log (uniform_continuous  0 5)))')
    ripl.assume('sf_rq','(tag (quote parameter) 5 (log (uniform_continuous 0 5)))')
    ripl.assume('l_rq','(tag (quote parameter) 6 (log (uniform_continuous 0 5)))')
    ripl.assume('alpha','(tag (quote parameter)7 (log (uniform_continuous 0 5)))')
    ripl.assume('sf',' (tag (quote parameter) 8 (log (uniform_continuous  0 5)))')

    ripl.assume('lin1', "(apply_function make_linear a   )")
    ripl.assume('per1', "(apply_function make_periodic l  p  sf ) ")
    ripl.assume('se1', "(apply_function make_se sf1 l1)")
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
    number = 4
    total_perms =0
    perms = []
    for i in range(number):
        perms.append((len(list(itertools.permutations([j for j in range(i+1)])))))
        total_perms+=perms[i]
    simplex = "( simplex  "
    for i in range(number):
        simplex+=str(float(perms[i])/total_perms) + " "

    simplex+=" )"
    #print(' (tag (quote grammar) 0 (subset (list lin1 per1 se1 se2 rq) '+simplex + ' ))')
    ripl.assume('s',' (tag (quote hyper) 0 (subset (list lin1 per1 se1 rq) '+simplex + ' ))')
    ripl.assume('cov','(tag (quote hyper) 1 (cov_compo s))')

    ripl.assume('gp',"""(tag (quote model) 0
                        (make_gp_part_der zero cov))""")


    makeObservations([i for i in range(X.shape[0])],X,ripl)
    ripl.infer("(repeat "+steps+" (do (mh (quote hyper) one 1) (mh (quote parameter) one 5)))")
    # prediction

    fig = plt.figure(figsize=(figlength,figheigth), dpi=200)
    for i in range(number_curves):
        xpost= np.random.uniform(np.min(X),np.max(X),200)
        sampleString=genSamples(xpost,ripl)
        ypost = ripl.sample(sampleString)
        yp = [y_temp for (x_temp,y_temp) in sorted(zip(xpost,ypost))]
        pl.plot(sorted(xpost),yp,c="red",alpha=alpha_value,linewidth=2)

    #pl.locator_params(nbins=4)
    #plt.axis((-2,2,-1,3))
    #pl.plot(X,color='blue')
    pl.scatter(range(X.shape[0]),X,color='black',marker='x',s=50,edgecolor='blue',linewidth='1.5')
    non_zero_index = np.nonzero(X)
    X = X[non_zero_index]
    time_stamps = get_time_stamps(patient)
    time_stamps=[time_stamps[i] for i in non_zero_index[0]]
    pl.xlabel('Time',fontsize=font_size)
    pl.ylabel('mg/dL',fontsize=font_size)
    pl.xticks(fontsize=font_size)
    pl.yticks(fontsize=font_size)
    ax = pl.gca()
    ax.set_xlim([np.min(X), np.max(X)+6])
    ticks = ax.get_xticks()
    time_stamps[0]=''
    ax.set_xticklabels([time_stamps[int(i)] for i in ticks[:-1]])

    fig.savefig(path+'posterior_samples_patient_'+patient_number+'_'+str(number_predictive_samples)+'_'+str(number_curves)+'a_'+str(alpha_value)+'.png', dpi=fig.dpi,bbox_inches='tight')

run_cgm_experiment(patient)




