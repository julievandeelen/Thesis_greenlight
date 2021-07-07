import numpy as np
import pandas as pd
import functools

from ema_workbench import (save_results, Constraint, SequentialEvaluator, TimeSeriesOutcome,
                           RealParameter, ScalarOutcome, CategoricalParameter, ema_logging, MultiprocessingEvaluator,
                           perform_experiments, Policy)

from ema_workbench.connectors.vensim import VensimModel
from ema_workbench.connectors import vensim
from ema_workbench.em_framework.parameters import Scenario
from ema_workbench.em_framework.optimization import (HyperVolume,
                                                     EpsilonProgress)
from ema_workbench.analysis import (prim, dimensional_stacking)

get_10_percentile = functools.partial(np.percentile, q=10)

def get_last_outcome(outcome,time):
    index = np.where(time == 2100) #model runs until 2100
    last_outcome = outcome[index][0]
    return last_outcome

def get_SD(outcome):
    sd=np.std(outcome)
    return sd

def lookup_list(time_range):
    list = np.arange(0, time_horizon, 2).tolist()
    return list

class MyVensimModel(VensimModel):

    def run_experiment(self, experiment):
        # 'Look up harvesting quota'
        # f"Proposed harvesting quota {t}", 0, 10) for t in range(45)]
        lookup = []
        decisions = []
        for t in range(45):
            value = experiment.pop(f"Proposed harvesting quota {t}")
            decisions.append(value)
            lookup.append((2010+2*t, value))
        experiment['Look up harvesting quota'] = lookup
        results = super(MyVensimModel, self).run_experiment(experiment)
        nvars = len(decisions)
        # decisions = np.array(decisions)
        results['Inertia'] = np.sum(np.abs(np.diff(decisions)) > 0.1)/float(nvars-1) #0.1
        return results

results_policies = pd.read_excel('./Data/filtered_policies.xlsx')

if __name__ == "__main__":
    ema_logging.log_to_stderr(ema_logging.INFO)

    wd = './Vensim models'
    vensimModel = MyVensimModel("simpleModel", wd=wd,
                               model_file='model_thesis.vpmx')

    vensimModel.uncertainties = [#structural uncertainties
                                 CategoricalParameter('Switch risk reward mechanism', (1,2,3) ),
                                 CategoricalParameter('Switch profitability change MF fisheries', (1,2) ),#1
                                 CategoricalParameter('Switch price change', (1,2) ),
                                 # CategoricalParameter('Switch risk perception climate', (1,2,3) ),
                                 # CategoricalParameter('Switch risk perception biomass', (1,2) ),
                                 # CategoricalParameter('Switch influence sunlight on phytoplankton', (1,2) ),
                                 # CategoricalParameter('Switch influence CO2 on phytoplankton', (1,2) ),
                                 # CategoricalParameter('Switch population growth', (1,2,3) ),
                                 # CategoricalParameter("Living depth myctophidae", (550, 650, 750, 850, 950) ),  # 950
                                 # CategoricalParameter("Living depth mauriculus", (250, 350, 450) ),  # 350

                                 #parametric uncertainties
                                 RealParameter("SWITCH lanternfish to mauriculus", 0.4 , 1),  #1
                                 RealParameter("Initial weight juvinile MF", 1*0.6 , 1*1.4),  #1
                                 RealParameter("Initial weight adult MF", 9*0.8 , 9*1.2),  #9
                                 # RealParameter("Initial juvinile predator weight", 6*0.8, 6*1.2),  #6
                                 # RealParameter("Initial predator weight", 4*0.8, 4*1.2),  #4
                                 # RealParameter("Average weight per predator", 0.3*0.8, 0.3*1.2),  #0.3
                                 # RealParameter("Average weight per juvinile predator", 0.08*0.8, 0.08*1.2),  #0.08
                                 RealParameter("Initial zooplankton", 4*0.8, 4*1.2),  #4
                                 # RealParameter("Initial phytoplankton", 1*0.8, 1*1.2),  #1
                                 RealParameter("Initial surface C", 600*0.95,600*1.05),  #600
                                 # RealParameter("Initial sediment C", 3390*0.8, 3390*1.2),  #3390
                                 RealParameter("Conversion factor to ppm", 2.0619 * 0.9, 2.0619 * 1.1),  # 2.0619
                                 RealParameter("Surface ocean", 363000000000000*0.9, 363000000000000*1.1),  # 3.63*10^14
                                 # RealParameter("Total atmospheric volume", 3990000000000000000*0.8, 3990000000000000000*1.2),  # 3.99*10^18
                                 RealParameter("Fraction grazed C ending up in surface", 0.4*0.6, 0.4*1.4),  # 0.4
                                 # RealParameter("Other carbon fluxes", 5*0.8, 5*1.2),  # 5
                                 # RealParameter("Carbon loss underway", 0.04*0.6, 0.04*1.4),  # 0.04
                                 # RealParameter("Carbon loss at living depth", 0.4*0.6, 0.4*1.4),  # 0.4
                                 # RealParameter("Average sinking time", 380*0.8, 380*1.2),  # 380
                                 # RealParameter("Delay sedimentation", 10000*0.8, 10000*1.2),  # 10000
                                 # RealParameter("Delay weathering", 10000*0.8, 10000*1.2),  # 10000
                                 # RealParameter("Efficiency factor fisheries", 7 * 0.8, 7 * 1.2), #7
                                 # RealParameter("Growth period predator", 3*0.8, 3*1.2),  # 3
                                 # RealParameter("Other food sources", 2*0.8, 2*1.2),  # 2
                                 # RealParameter("Turnover time phytoplankton", 0.077*0.8, 0.077*1.2),  # 0.077
                                 RealParameter("Consumption by zooplankton in bodyweight", 3*0.8, 3*1.2),  # 3
                                 # RealParameter("Harvest information delay", 3*0.8, 3*1.2),  # 3
                                 # RealParameter("Costs regular fish", 339000000000*0.8, 339000000000*1.2),  # 3.39e+11
                                 # RealParameter("Sale price regular fish", 3100000000000*0.8, 3100000000000*1.2),  # 3.1e+12
                                 # RealParameter("Costs status quo myctophidae", 450000000000*0.6, 450000000000*1.4),  # 450*10^9
                                 # RealParameter("Costs status quo mauriculus", 300000000000*0.8, 300000000000*1.2),  # 225*10^9
                                 # RealParameter("Information delay risk perception", 5*0.8, 5*1.2),  # 5
                                 # RealParameter("ppm conversion for ocean", 2.1*0.9, 2.1*1.1),  # 2.1
                                 RealParameter("Downwelling water", 1700000000000000*0.8, 1700000000000000*1.2),  # 1.7*10^15
                                 RealParameter("Residence time deep carbon", 1000*0.9, 1000*1.1),  # 1000
                                 RealParameter("Upwelling delay surface", 8*0.9, 8*1.1),  # 8
                                 # RealParameter("Share of aquaculture", 0.5*0.8, 0.5*1.2),  # 0.5
                                 # RealParameter("Share of irreplaceable fishmeal", 0.8, 1),  # 0.1
                                 # RealParameter("Annual fish consumption per capita", 0.000017*0.8, 0.000017*1.2),  # 1.7*10^-5
                                 # RealParameter("Percentage discarded fish", 0.08*0.8, 0.08*1.2), #0.08
                                 # RealParameter("Specialist capacity building time", 5*0.8, 5*1.2),  # 5
                                 RealParameter("Catchability myctophidae", 0.14*0.6, 0.14*1.4),  # 0.14
                                 # RealParameter("Catchability mauriculus", 0.28*0.8, 0.28*1.2),  # 0.28
                                 # RealParameter("Fraction of migrating MF constant", 0.37*0.6, 0.37*1.4),  # 0.37
                                 RealParameter("Grazing in surface by MF", 0.4*0.6, 0.4*1.4),  # 0.4
                                 RealParameter("C content copepods", 0.51*0.8, 0.51*1.2),  # 0.51
                                 RealParameter("Depth euphotic zone", 100*0.9, 100*1.1),  # 100#CHANGED FROM 40% to 25%
                                 # RealParameter("Total atmospheric volume", 3990000000000000000*0.8, 3990000000000000000*1.2),  # 3.99*10^18
                                 # RealParameter("Export efficiency", 0.97*0.6, 0.97*1.4),  # 0.97
                                 RealParameter("Transfer velocity for GtC per year", 1.12169*0.8, 1.12169*1.2),  # 1.12169
                                 # RealParameter("Average weight per adult MF", 0.000001*0.8, 0.000001*1.2),  # 1*10^-6
                                 # RealParameter("Average weight per juvinile MF", 0.00000007*0.8, 0.00000007*1.2),  # 0.07*10^-6
                                 # RealParameter("Life expectancy myctophidae adult", 4*0.8, 4*1.2),  # 4
                                 # RealParameter("Life expectancy mauriculus", 2*0.8, 2*1.2),  # 2
                                 # RealParameter("Growth period myctophidae", 0.583*0.9, 0.583*1.1),  # 0.583
                                 # RealParameter("Growth period mauriculus", 1*0.9, 1*1.1),  # 1
                                 RealParameter("Consumption by MF in bodyweight", 7*0.8, 7*1.2),  # 7
                                 # RealParameter("Annual consumption predator", 1.6*0.8, 1.6*1.2),  # 1.6
                                 # RealParameter("Predator life expectancy", 6*0.8, 6*1.2),  # 6
                                 # RealParameter("Survived larvea", 147*0.8, 147*1.2),  # 147
                                 RealParameter("Spawning fraction", 0.18*0.8, 0.18*1.2),  # 0.18
                                 RealParameter("Female fraction", 0.515*0.8, 0.515*1.2),  # 0.515
                                 RealParameter("Fraction spawning mauriculus vs myctophidae", 0.75*0.9, 0.75*1.1),  # 0.75
                                 RealParameter("Fishmeal to fish factor", 4*0.8, 4*1.2)  # 4
                                 ]


    vensimModel.levers = [RealParameter(f"Proposed harvesting quota {t}", 0, 6) for t in range(45)]

    vensimModel.outcomes = [ScalarOutcome('Average food provision by MF', variable_name='Food provision by MF',
                                          kind=ScalarOutcome.MAXIMIZE
                                    , function=np.mean),
                            ScalarOutcome('Average vertical migration', variable_name='Total vertical migration',
                                          kind=ScalarOutcome.MAXIMIZE
                                   , function=np.mean),
                            ScalarOutcome('Biomass MF 10th percentile', variable_name='Biomass mesopelagic fish',
                                          kind=ScalarOutcome.MAXIMIZE
                                   , function=get_10_percentile),
                            ScalarOutcome('Final atmospheric C level', variable_name=['Atmospheric C', 'TIME'],
                                          kind=ScalarOutcome.MINIMIZE
                                          , function=get_last_outcome),
                            ScalarOutcome('Inertia', kind=ScalarOutcome.MINIMIZE)
                            ]

    policies_to_evaluate = []

    for i, policy in results_policies.iterrows():
        policies_to_evaluate.append(Policy(str(i), **policy.to_dict()))


    ema_logging.log_to_stderr(ema_logging.INFO)

    n_scenarios = 10000 # 10.000
    n_policies = policies_to_evaluate

    with MultiprocessingEvaluator(vensimModel) as evaluator:
        results = evaluator.perform_experiments(n_scenarios, n_policies)

    save_results(results, './Data/results_step4_7.tar.gz')

    experiments, outcomes = results

    # delete outcome TIME
    del outcomes['TIME']
    # make dataframe
    experiments = pd.DataFrame.from_dict(experiments)
    outcomes = pd.DataFrame.from_dict(outcomes)

    # drop
    experiments = experiments.drop(labels=[l.name for l in vensimModel.levers], axis=1)

    experiments.to_excel('./Data/experiments_st4.xlsx')
    outcomes.to_excel('./Data/outcomes_st4.xlsx')

    # performing prim
    # y = outcomes['Biomass MF 10th percentile'] < 9

    # prim_alg = prim.Prim(experiments, y, threshold=0.8)
    # box1 = prim_alg.find_box()
    #
    # # plotting results
    # import matplotlib.pyplot as plt
    #
    # box1.show_tradeoff()
    # plt.savefig("./Figures/prim_box_7.png")
    # plt.show()
    #
    #
    # # generate a scatterplot
    # box1.select(2)
    # box1.show_pairs_scatter()
    # fig = plt.gcf()
    # fig.set_size_inches(15, 15)
    # plt.savefig('./Figures/step_4_scatter_plot_7.png')
    # plt.show()
    #
    # dimensional_stacking.create_pivot_plot(experiments, y)
    # plt.savefig('./Figures/step_4_dmn_stack_7.png')
    # plt.show()


