import os
from decimal import Decimal
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from collections import defaultdict
import operator
import random
from wandbCSV_to_latexTable import get_approxMC_estAndTime, get_F2_estAndTime, get_exact_ln_modelCount, get_exact_ln_modelCount
import json
from SAT_train_test_split import ALL_TRAIN_PROBLEMS

def make_SAT_estimate_figure(runtimes_dir='/atlas/u/jkuck/learn_BP/data/SAT_BPNN_runtimes/', problem_category='or_50_problems'):
    '''
    should be able to load dictionary of all training problem results for BPNN, from running learn_BP_SAT.py in testing mode
    results = {'squared_errors': squared_errors, 'runtimes': runtimes, 
               'BPNN_estimated_ln_counts': BPNN_estimated_counts, 
               'exact_ln_solution_counts': exact_solution_counts,
               'problem_names': problem_names}    
    '''
    
    # with open(runtimes_dir + "trainSet_runtimesAndErrors_3layer.json", 'r') as json_file:
    with open(runtimes_dir + "or50_trainSet_runtimesAndErrors_5layer.json", 'r') as json_file:
        results = json.load(json_file)

    print('sanity check, BPNN RMSE =', np.sqrt(np.mean(results['squared_errors'])))
        
    assert(len(results['runtimes']) == len(results['problem_names']))
    approxMC_ests = []
    exact_counts_approxMC = []
    
    F2_ests = []
    exact_counts_F2 = []
    
    BPNN_ests = []
    exact_counts_BPNN = []
    # print("results:")
    # print(results)
    for idx, BPNN_est in enumerate(results['BPNN_estimated_ln_counts']):
        problem_name = results['problem_names'][idx]
        if not (problem_name in [problem['problem'] for problem in  ALL_TRAIN_PROBLEMS[problem_category]]):
            continue
        BPNN_ests.append(BPNN_est - results['exact_ln_solution_counts'][idx])
        exact_counts_BPNN.append(results['exact_ln_solution_counts'][idx])
        
        approxMC_lnZ_est, approxMC_time = get_approxMC_estAndTime(problem_name)
        if approxMC_lnZ_est is not None:
            approxMC_ests.append(approxMC_lnZ_est - results['exact_ln_solution_counts'][idx])
            exact_counts_approxMC.append(results['exact_ln_solution_counts'][idx])
        F2_lnZ_est, F2_varDeg3_time = get_F2_estAndTime(problem_name)
        if F2_lnZ_est is not None:
            F2_ests.append(F2_lnZ_est - results['exact_ln_solution_counts'][idx])        
            exact_counts_F2.append(results['exact_ln_solution_counts'][idx])
    
#     plt.plot(x_vals, perfect, '-', label='Zero Error')

    print("len(BPNN_ests):", len(BPNN_ests))
    print("len(exact_counts_approxMC):", len(exact_counts_approxMC))
    print("len(exact_counts_F2):", len(exact_counts_F2))

    
    plt.plot(exact_counts_BPNN, BPNN_ests, 'x', color='blue', label='BPNN')
    plt.plot(exact_counts_approxMC, approxMC_ests, '+', color='tab:orange', label='ApproxMC3')
    plt.plot(exact_counts_F2, F2_ests, '1', color='green', label='F2')


    # plt.xlabel('(f_max, c_max)', fontsize=14)
#     plt.xlabel(r'$\ln(\textrm{Exact Model Count})$', fontsize=14)
    plt.xlabel("ln(Exact Model Count)", fontsize=14)

    # plt.xlabel('\\textit{Velocity (\N{DEGREE SIGN}/sec)}', fontsize=14)

#     plt.ylabel(r'$\ln(\textrm{Estimate}) - \ln(\textrm{Exact Model Count})$', fontsize=15)
    plt.ylabel("ln(Estimate) - ln(Exact Model Count)", fontsize=14)

               
    # plt.yscale('symlog')
    # plt.xscale('log')


    plt.title('Exact vs. Estimated Model Counts', fontsize=20)
    # plt.legend(fontsize=12)    
    lgd = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13),
            fancybox=True, ncol=3, fontsize=12, prop={'size': 11})
    #make the font bigger
    matplotlib.rcParams.update({'font.size': 10})        

    plt.grid(True)

#     plt.xticks(np.arange(9), ['(.1, 5)', '(.1, 10)', '(.1, 50)', '(.2, 5)', '(.2, 10)', '(.2, 50)', '(1, 5)', '(1, 1)', '(1, 50)'])
    # plot_name = 'quick_plot.png'
    # plt.savefig(ROOT_DIR + 'sat_plots/' + plot_name) 
    plt.savefig('./exactVsEstModelCounts.eps', bbox_extra_artists=(lgd,), bbox_inches='tight', format='eps')   
    # plt.show()    
    

def get_BPNN_runtime_ratios(runtimes_dir='/atlas/u/jkuck/learn_BP/data/SAT_BPNN_runtimes/'):
    '''
    should be able to load dictionary of all training problem results for BPNN, from running learn_BP_SAT.py in testing mode
    results = {'squared_errors': squared_errors, 'runtimes': runtimes, 
               'BPNN_estimated_ln_counts': BPNN_estimated_counts, 
               'exact_ln_solution_counts': exact_solution_counts,
               'problem_names': problem_names}    
    '''
    # with open(runtimes_dir + "trainSet_runtimesAndErrors_2layer.json", 'r') as json_file:
    with open(runtimes_dir + "or50_trainSet_runtimesAndErrors_5layer.json", 'r') as json_file:
        results = json.load(json_file)

    print('sanity check, BPNN RMSE =', np.sqrt(np.mean(results['squared_errors'])))
        
    assert(len(results['runtimes']) == len(results['problem_names']))
    approxMC_over_BPNN_runtime_ratios = []
    F2_over_BPNN_runtime_ratios = []
    approxMC_over_F2_runtime_ratios = []
    
    BPNN_runtimes = []
    approxMC_runtimes = []
    F2_runtimes = []

    BPNN_errors = []
    max_ratio=1
    for idx, problem_name in enumerate(results['problem_names']):
        BPNN_runtime = results['runtimes'][idx]
        BPNN_runtimes.append(BPNN_runtime)

        approxMC_lnZ_est, approxMC_time = get_approxMC_estAndTime(problem_name)
        if approxMC_time is not None:
            # approxMC_runtimes.append(max(approxMC_time, .01)) #I believe some ApproxMC/F2 times are incorrectly recorded as 0
            approxMC_runtimes.append(approxMC_time) #I believe some ApproxMC/F2 times are incorrectly recorded as 0
            if approxMC_time/BPNN_runtime > max_ratio:
                max_ratio = approxMC_time/BPNN_runtime
                print('new max_ratio=', max_ratio, "approxMC_time=", approxMC_time, "BPNN_runtime=", BPNN_runtime)
            approxMC_over_BPNN_runtime_ratios.append(approxMC_time/BPNN_runtime)
        else:
            pass
            # approxMC_runtimes.append(5000) #represent timeout

        F2_lnZ_est, F2_varDeg3_time = get_F2_estAndTime(problem_name)
        if F2_varDeg3_time is not None:
            F2_over_BPNN_runtime_ratios.append(F2_varDeg3_time/BPNN_runtime)        
            # F2_runtimes.append(max(F2_varDeg3_time, .01)) #I believe some ApproxMC/F2 times are incorrectly recorded as 0
            F2_runtimes.append(F2_varDeg3_time) #I believe some ApproxMC/F2 times are incorrectly recorded as 0
        else:
            pass
            # F2_runtimes.append(5000) #represent timeout
        if (F2_varDeg3_time is not None) and (approxMC_time is not None):
            approxMC_over_F2_runtime_ratios.append(approxMC_time/F2_varDeg3_time)        




    print("total problem count =", len(results['problem_names']))
    print("fraction of problems completed by approxMC =", len(approxMC_over_BPNN_runtime_ratios)/len(results['problem_names']))
    print("problem count completed by approxMC =", len(approxMC_over_BPNN_runtime_ratios))
    print("fraction of problems completed by F2 =", len(F2_over_BPNN_runtime_ratios)/len(results['problem_names']))
    print()

    # bins = np.linspace(0, 6000, 100)


    #for or_50, this model https://app.wandb.ai/jdkuck/learn_BP_sat_MLP34_CompareDoubleCount_andBethe/runs/szfxspl1?workspace=user-jdkuck
    BPNN_runtimes_seq_gpu = [0.060622215270996094, 0.060539960861206055, 0.060422658920288086, 0.05814838409423828, 0.05730605125427246, 0.059450626373291016, 0.0597033500673867, 0.05996990203857422, 0.06163763999938965, 0.058892250061035156, 0.06026339530944824, 0.06047701835632324, 0.059780120849609375, 0.057590484619140625, 0.05834650993347168, 0.05983996391296387, 0.06048274040222168, 0.05918431282043457, 0.05893731117248535, 0.05782723426818848, 0.06079387664794922, 0.06841516494750977, 0.059058189392089844, 0.057517290115356445, 0.057523488998413086, 0.06108689308166504, 0.06038212776184082, 0.05892515182495117, 0.05756950378417969, 0.0595550537109375, 0.06154036521911621, 0.059813737869262695, 0.060307979583740234, 0.05775809288024902, 0.058362722396850586, 0.06273961067199707, 0.05844473838806152, 0.0576326847076416, 0.06247711181640625, 0.05849575996398926, 0.06070280075073242, 0.058536529541015625, 0.058138370513916016, 0.06125783920288086, 0.05908656120300293, 0.059549570083618164, 0.059453487396240234, 0.05890989303588867, 0.0604095458984375, 0.06146955490112305, 0.05936861038208008, 0.06851601600646973, 0.059723854064941406, 0.05856633186340332, 0.06199145317077637, 0.05920743942260742, 0.058744192123413086, 0.05825495719909668, 0.05923819541931152, 0.060155630111694336, 0.060860633850097656, 0.05896353721618652, 0.058365821838378906, 0.06010627746582031, 0.06031608581542969, 0.06064271926879883, 0.05863761901855469, 0.05854511260986328, 0.06004834175109863, 0.06404232978820801, 0.06216001510620117, 0.058770179748535156, 0.058008670806884766, 0.06010890007019043, 0.061095237731933594, 0.06026768684387207, 0.05809926986694336, 0.06148171424865723, 0.060827016830444336, 0.060285329818725586, 0.06374168395996094, 0.061304330825805664, 0.06731700897216797, 0.06122541427612305, 0.0596919059753418, 0.060633182525634766, 0.05931258201599121, 0.06046700477600098, 0.06057119369506836, 0.05922651290893555, 0.0596463680267334, 0.05864453315734863, 0.06057572364807129, 0.05861973762512207, 0.05899357795715332, 0.060057878494262695, 0.058429718017578125, 0.060938119888305664, 0.05978274345397949, 0.05924057960510254, 0.06050682067871094, 0.05862569808959961, 0.05917477607727051, 0.05892324447631836, 0.0596163272857666]
    BPNN_runtimes_seq_cpu = [0.10863208770751953, 0.08238554000854492, 0.09173440933227539, 0.0853269100189209, 0.08945107460021973, 0.09090781211853027, 0.09998345375061035, 0.09675192832946777, 0.08996844291687012, 0.08359599113464355, 0.08504152297973633, 0.09459900856018066, 0.09918832778930664, 0.08538317680358887, 0.09831380844116211, 0.09071493148803711, 0.09154057502746582, 0.0883016586303711, 0.08319497108459473, 0.08014965057373047, 0.07815337181091309, 0.10342788696289062, 0.0851905345916748, 0.08782553672790527, 0.07578802108764648, 0.07806539535522461, 0.0774688720703125, 0.0783236026763916, 0.07511210441589355, 0.08007693290710449, 0.11082720756530762, 0.08535027503967285, 0.0774846076965332, 0.0760958194732666, 0.08675217628479004, 0.0793604850769043, 0.07645344734191895, 0.07773232460021973, 0.09214377403259277, 0.08632969856262207, 0.0864725112915039, 0.08308815956115723, 0.07702803611755371, 0.09874510765075684, 0.07765340805053711, 0.08294534683227539, 0.08363842964172363, 0.07770514488220215, 0.07666540145874023, 0.07872462272644043, 0.07764172554016113, 0.0809934139251709, 0.0753176212310791, 0.07770633697509766, 0.0775141716003418, 0.08794689178466797, 0.07906389236450195, 0.07759380340576172, 0.07718896865844727, 0.07984185218811035, 0.0792841911315918, 0.07827448844909668, 0.07629871368408203, 0.07602643966674805, 0.08040928840637207, 0.07563924789428711, 0.07411479949951172, 0.10200667381286621, 0.0789496898651123, 0.07280421257019043, 0.09984064102172852, 0.08788609504699707, 0.0825803279876709, 0.07758307456970215, 0.08542346954345703, 0.07487273216247559, 0.07709240913391113, 0.0747215747833252, 0.08442449569702148, 0.07541346549987793, 0.07544732093811035, 0.07341313362121582, 0.07744002342224121, 0.07303905487060547, 0.0758812427520752, 0.0822751522064209, 0.08058404922485352, 0.07228231430053711, 0.07515335083007812, 0.07942485809326172, 0.09194588661193848, 0.07770371437072754, 0.07254290580749512, 0.08791661262512207, 0.07408833503723145, 0.07698822021484375, 0.07510852813720703, 0.08027982711791992, 0.07554292678833008, 0.08650732040405273, 0.07317972183227539, 0.08393049240112305, 0.0718679428100586, 0.07615876197814941, 0.07975363731384277]

    ###### MAKE A HISTOGRAM ######
    bins = np.logspace(np.log10(.001),np.log10(6000), 25)
    plt.xscale('log')

    plt.hist(BPNN_runtimes, bins, alpha=0.5, label='BPNN-P')
    plt.hist(BPNN_runtimes_seq_cpu, bins, alpha=0.5, label='BPNN-S')
    plt.hist(F2_runtimes, bins, alpha=0.5, ls='dashed', lw=1, edgeColor = 'black', label='F2')
    plt.hist(approxMC_runtimes, bins, alpha=0.5, label='ApproxMC3')

    plt.title('Runtimes or_50 Benchmarks', fontsize=20)
    plt.xlabel("Runtime (Seconds)", fontsize=14)
    plt.ylabel("Benchmark Count", fontsize=14)

    lgd = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13),
            fancybox=True, ncol=4, fontsize=12, prop={'size': 11})

    matplotlib.rcParams.update({'font.size': 10})        

    #eps doesn't work with transparency
    plt.savefig('./runtime_histogram.png', bbox_extra_artists=(lgd,), bbox_inches='tight', format='png')   
    plt.clf()
    ###### MAKE A CACTUS PLOT ######
    bins = np.logspace(np.log10(.001),np.log10(6000), 25)
    plt.yscale('log')

    plt.scatter([i for i in range(len(BPNN_runtimes))], sorted(BPNN_runtimes), marker='x',linewidth=.75, color='blue', label='BPNN-P')
    plt.scatter([i for i in range(len(BPNN_runtimes_seq_cpu))], sorted(BPNN_runtimes_seq_cpu), marker='2',linewidth=.75, color='blue', label='BPNN-S')
    plt.scatter([i for i in range(len(approxMC_runtimes))], sorted(approxMC_runtimes), marker='+',linewidth=.75, color='tab:orange', label='ApproxMC3')
    plt.scatter([i for i in range(len(F2_runtimes))], sorted(F2_runtimes), marker='1',linewidth=.75, color='green', label='F2')

    # plt.title('Runtimes For Solved or_50 Benchmarks', fontsize=20)
    plt.title('Runtimes On OR_50 Benchmarks', fontsize=20)
    plt.xlabel("Benchmark", fontsize=14)
    plt.ylabel("Runtime (Seconds)", fontsize=14)

    lgd = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13),
            fancybox=True, ncol=4, fontsize=12, prop={'size': 11})

    matplotlib.rcParams.update({'font.size': 10})        

    plt.savefig('./runtime_cactus_plot.eps', bbox_extra_artists=(lgd,), bbox_inches='tight', format='eps')   
    plt.clf()

    #### Try 3d plot ####
    # get absolute errors
    assert(len(results['runtimes']) == len(results['problem_names']))
    approxMC_errors = []    
    F2_errors = []    
    BPNN_errors = []
    for idx, BPNN_est in enumerate(results['BPNN_estimated_ln_counts']):
        problem_name = results['problem_names'][idx]
        # if not (problem_name in [problem['problem'] for problem in  ALL_TRAIN_PROBLEMS[problem_category]]):
        #     continue
        BPNN_errors.append(np.abs(BPNN_est - results['exact_ln_solution_counts'][idx]))
        
        approxMC_lnZ_est, approxMC_time = get_approxMC_estAndTime(problem_name)
        if approxMC_lnZ_est is not None:
            approxMC_errors.append(np.abs(approxMC_lnZ_est - results['exact_ln_solution_counts'][idx]))
        F2_lnZ_est, F2_varDeg3_time = get_F2_estAndTime(problem_name)
        if F2_lnZ_est is not None:
            F2_errors.append(np.abs(F2_lnZ_est - results['exact_ln_solution_counts'][idx]))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # plt.title('Runtimes For Solved or_50 Benchmarks', fontsize=20)
    # ax.set_title('Runtimes And Estimates On OR_50 Benchmarks', fontsize=20)
    ax.set_xlabel("Benchmark", fontsize=14)
    ax.set_ylabel("Runtime (Seconds)", fontsize=14)
    ax.set_zlabel("Error", fontsize=14)

    # lgd = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13),
    #         fancybox=True, ncol=4, fontsize=12, prop={'size': 11})

    matplotlib.rcParams.update({'font.size': 10})        


    BPNN_runtimes_seq_cpu_sorted, BPNN_errors_sorted = zip(*sorted(zip(BPNN_runtimes_seq_cpu, BPNN_errors)))
    approxMC_runtimes_sorted, approxMC_errors_sorted = zip(*sorted(zip(approxMC_runtimes, approxMC_errors)))
    F2_runtimes_sorted, F2_errors_sorted = zip(*sorted(zip(F2_runtimes, F2_errors)))

    ax.scatter([i for i in range(len(BPNN_runtimes_seq_cpu))], BPNN_runtimes_seq_cpu_sorted, BPNN_errors_sorted)#, zdir='z', s=20, c=None, depthshade=True)#, *args, **kwargs)
    ax.scatter([i for i in range(len(approxMC_runtimes))], approxMC_runtimes_sorted, approxMC_errors_sorted)#, zdir='z', s=20, c=None, depthshade=True)#, *args, **kwargs)
    ax.scatter([i for i in range(len(F2_runtimes))], F2_runtimes_sorted, F2_errors_sorted)#, zdir='z', s=20, c=None, depthshade=True)#, *args, **kwargs)
    plt.savefig('./runtime_and_error_3d.png')#, bbox_extra_artists=(lgd,), bbox_inches='tight', format='png')   



    return approxMC_over_BPNN_runtime_ratios, F2_over_BPNN_runtime_ratios, approxMC_over_F2_runtime_ratios


def print_BPNN_runtime_ratio_stats():
    approxMC_over_BPNN_runtime_ratios, F2_over_BPNN_runtime_ratios, approxMC_over_F2_runtime_ratios = get_BPNN_runtime_ratios()
    print("np.min(approxMC_over_BPNN_runtime_ratios):", np.min(approxMC_over_BPNN_runtime_ratios))
    print("np.percentile(approxMC_over_BPNN_runtime_ratios, 10):", np.percentile(approxMC_over_BPNN_runtime_ratios, 10))        
    print("np.mean(approxMC_over_BPNN_runtime_ratios):", np.mean(approxMC_over_BPNN_runtime_ratios))
    print("np.median(approxMC_over_BPNN_runtime_ratios):", np.median(approxMC_over_BPNN_runtime_ratios))
    print("np.max(approxMC_over_BPNN_runtime_ratios):", np.max(approxMC_over_BPNN_runtime_ratios))
    print()
    print("np.min(approxMC_over_F2_runtime_ratios):", np.min(approxMC_over_F2_runtime_ratios))
    print("np.percentile(approxMC_over_F2_runtime_ratios, 10):", np.percentile(approxMC_over_F2_runtime_ratios, 10))    
    print("np.mean(approxMC_over_F2_runtime_ratios):", np.mean(approxMC_over_F2_runtime_ratios))
    print("np.median(approxMC_over_F2_runtime_ratios):", np.median(approxMC_over_F2_runtime_ratios))
    print("np.max(approxMC_over_F2_runtime_ratios):", np.max(approxMC_over_F2_runtime_ratios))    
    print()
    print("np.min(F2_over_BPNN_runtime_ratios):", np.min(F2_over_BPNN_runtime_ratios))
    print("np.percentile(F2_over_BPNN_runtime_ratios, 10):", np.percentile(F2_over_BPNN_runtime_ratios, 10))    
    print("np.mean(F2_over_BPNN_runtime_ratios):", np.mean(F2_over_BPNN_runtime_ratios))
    print("np.median(F2_over_BPNN_runtime_ratios):", np.median(F2_over_BPNN_runtime_ratios))
    print("np.max(F2_over_BPNN_runtime_ratios):", np.max(F2_over_BPNN_runtime_ratios))


if __name__ == "__main__":
    print_BPNN_runtime_ratio_stats()
    # make_SAT_estimate_figure()
