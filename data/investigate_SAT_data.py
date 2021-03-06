import os
from decimal import Decimal
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from collections import defaultdict
import operator
import random

################################################################################
##### Current Testing SAT Problems
################################################################################
blasted_problems_test =["blasted_case_0_b11_1", "blasted_case_0_b12_1", "blasted_case_0_b12_2", "blasted_case_0_b12_even1", "blasted_case_0_b12_even2", "blasted_case_0_b12_even3", "blasted_case_0_b14_1", "blasted_case_0_ptb_1", "blasted_case_0_ptb_2", "blasted_case100", "blasted_case101", "blasted_case102", "blasted_case103", "blasted_case104", "blasted_case105", "blasted_case106", "blasted_case107", "blasted_case108", "blasted_case109", "blasted_case10", "blasted_case110", "blasted_case111", "blasted_case112", "blasted_case113", "blasted_case114", "blasted_case115", "blasted_case116", "blasted_case117", "blasted_case118", "blasted_case119", "blasted_case11", "blasted_case120", "blasted_case121", "blasted_case122", "blasted_case123", "blasted_case124", "blasted_case125", "blasted_case126", "blasted_case127", "blasted_case128", "blasted_case12", "blasted_case130", "blasted_case131", "blasted_case132", "blasted_case133", "blasted_case134", "blasted_case135", "blasted_case136", "blasted_case137", "blasted_case138", "blasted_case139", "blasted_case140", "blasted_case141", "blasted_case142", "blasted_case143", "blasted_case144", "blasted_case145", "blasted_case146", "blasted_case_1_4_b14_even", "blasted_case14", "blasted_case15", "blasted_case17", "blasted_case18", "blasted_case19", "blasted_case_1_b11_1", "blasted_case_1_b12_1", "blasted_case_1_b12_2", "blasted_case_1_b12_even1", "blasted_case_1_b12_even2", "blasted_case_1_b12_even3", "blasted_case_1_b14_1", "blasted_case_1_b14_2", "blasted_case_1_b14_3", "blasted_case1_b14_even3", "blasted_case_1_b14_even", "blasted_case1", "blasted_case_1_ptb_1", "blasted_case_1_ptb_2", "blasted_case200", "blasted_case201", "blasted_case202", "blasted_case203", "blasted_case204", "blasted_case205", "blasted_case206", "blasted_case207", "blasted_case208", "blasted_case209", "blasted_case20", "blasted_case210", "blasted_case211", "blasted_case212", "blasted_case213", "blasted_case214", "blasted_case21", "blasted_case22", "blasted_case23", "blasted_case24", "blasted_case25", "blasted_case26", "blasted_case27", "blasted_case28", "blasted_case29", "blasted_case_2_b12_1", "blasted_case_2_b12_2", "blasted_case_2_b12_even1", "blasted_case_2_b12_even2", "blasted_case_2_b12_even3", "blasted_case_2_b14_1", "blasted_case_2_b14_2", "blasted_case_2_b14_3", "blasted_case_2_b14_even", "blasted_case2", "blasted_case_2_ptb_1", "blasted_case_2_ptb_2", "blasted_case30", "blasted_case31", "blasted_case32", "blasted_case33", "blasted_case_3_4_b14_even", "blasted_case34", "blasted_case35", "blasted_case36", "blasted_case37", "blasted_case38", "blasted_case39", "blasted_case_3_b14_1", "blasted_case_3_b14_2", "blasted_case_3_b14_3", "blasted_case3_b14_even3", "blasted_case3", "blasted_case40", "blasted_case41", "blasted_case42", "blasted_case43", "blasted_case44", "blasted_case45", "blasted_case46", "blasted_case47", "blasted_case49", "blasted_case4", "blasted_case50", "blasted_case51", "blasted_case52", "blasted_case53", "blasted_case54", "blasted_case55", "blasted_case56", "blasted_case57", "blasted_case58", "blasted_case59_1", "blasted_case59", "blasted_case5", "blasted_case60", "blasted_case61", "blasted_case62", "blasted_case63", "blasted_case64", "blasted_case68", "blasted_case6", "blasted_case7", "blasted_case8", "blasted_case9", "blasted_squaring10", "blasted_squaring11", "blasted_squaring12", "blasted_squaring14", "blasted_squaring16", "blasted_squaring1", "blasted_squaring20", "blasted_squaring21", "blasted_squaring22", "blasted_squaring23", "blasted_squaring24", "blasted_squaring25", "blasted_squaring26", "blasted_squaring27", "blasted_squaring28", "blasted_squaring29", "blasted_squaring2", "blasted_squaring30", "blasted_squaring3", "blasted_squaring40", "blasted_squaring41", "blasted_squaring42", "blasted_squaring4", "blasted_squaring50", "blasted_squaring51", "blasted_squaring5", "blasted_squaring60", "blasted_squaring6", "blasted_squaring70", "blasted_squaring7", "blasted_squaring8", "blasted_squaring9", "blasted_TR_b12_1_linear", "blasted_TR_b12_2_linear", "blasted_TR_b12_even2_linear", "blasted_TR_b12_even3_linear", "blasted_TR_b12_even7_linear", "blasted_TR_b14_1_linear", "blasted_TR_b14_2_linear", "blasted_TR_b14_3_linear", "blasted_TR_b14_even2_linear", "blasted_TR_b14_even3_linear", "blasted_TR_b14_even_linear", "blasted_TR_device_1_even_linear", "blasted_TR_device_1_linear", "blasted_TR_ptb_1_linear", "blasted_TR_ptb_2_linear"]

################################################################################

brp_problems_test = ["brp.pm_14steps_10int_8fract_p1_N=200_MAX=4over", "brp.pm_14steps_10int_8fract_p1_N=200_MAX=4under", "brp.pm_14steps_12int_8fract_p1_N=1000_MAX=4over", "brp.pm_14steps_12int_8fract_p1_N=1000_MAX=4under", "brp.pm_14steps_12int_8fract_p1_N=400_MAX=4over", "brp.pm_14steps_12int_8fract_p1_N=400_MAX=4under", "brp.pm_14steps_12int_8fract_p1_N=600_MAX=4over", "brp.pm_14steps_12int_8fract_p1_N=600_MAX=4under", "brp.pm_14steps_12int_8fract_p1_N=800_MAX=4over", "brp.pm_14steps_12int_8fract_p1_N=800_MAX=4under", "brp.pm_14steps_13int_8fract_p1_N=2000_MAX=4over", "brp.pm_14steps_13int_8fract_p1_N=2000_MAX=4under", "brp.pm_14steps_14int_8fract_p1_N=3000_MAX=4over", "brp.pm_14steps_14int_8fract_p1_N=3000_MAX=4under", "brp.pm_14steps_14int_8fract_p1_N=4000_MAX=4over", "brp.pm_14steps_14int_8fract_p1_N=4000_MAX=4under", "brp.pm_14steps_14int_8fract_p1_N=5000_MAX=4over", "brp.pm_14steps_14int_8fract_p1_N=5000_MAX=4under", "brp.pm_14steps_22int_8fract_p1_N=1000000_MAX=4over", "brp.pm_14steps_22int_8fract_p1_N=1000000_MAX=4under", "brp.pm_14steps_22int_8fract_p1_N=100000_MAX=4over", "brp.pm_14steps_22int_8fract_p1_N=100000_MAX=4under", "brp.pm_14steps_22int_8fract_p1_N=10000_MAX=4over", "brp.pm_14steps_22int_8fract_p1_N=10000_MAX=4under", "brp.pm_14steps_26int_8fract_p1_N=10000000_MAX=4over", "brp.pm_14steps_26int_8fract_p1_N=10000000_MAX=4under", "brp.pm_14steps_30int_8fract_p1_N=100000000_MAX=4over", "brp.pm_14steps_30int_8fract_p1_N=100000000_MAX=4under", "brp.pm_14steps_8int_8fract_p1_N=16_MAX=2over", "brp.pm_14steps_8int_8fract_p1_N=16_MAX=2under", "brp.pm_14steps_8int_8fract_p1_N=32_MAX=4over", "brp.pm_14steps_8int_8fract_p1_N=32_MAX=4under", "brp.pm_14steps_8int_8fract_p1_N=64_MAX=5over", "brp.pm_14steps_8int_8fract_p1_N=64_MAX=5under"]

################################################################################

sk_problems_test = ["compress.sk_17_291", "ConcreteActivityService.sk_13_28", "ConcreteRoleAffectationService.sk_119_273", "diagStencilClean.sk_41_36", "diagStencil.sk_35_36", "doublyLinkedList.sk_8_37", "enqueueSeqSK.sk_10_42", "GuidanceService2.sk_2_27", "GuidanceService.sk_4_27", "isolateRightmost.sk_7_481", "IssueServiceImpl.sk_8_30", "IterationService.sk_12_27", "jburnim_morton.sk_13_530", "karatsuba.sk_7_41", "listReverse.sk_11_43", "logcount.sk_16_86", "LoginService2.sk_23_36", "LoginService.sk_20_34", "log2.sk_72_391", "lss.sk_6_7", "NotificationServiceImpl2.sk_10_36", "parity.sk_11_11", "partition.sk_22_155", "PhaseService.sk_14_27", "Pollard.sk_1_10", "polynomial.sk_7_25", "ProcessBean.sk_8_64", "ProjectService3.sk_12_55", "registerlesSwap.sk_3_10", "reverse.sk_11_258", "SetTest.sk_9_21", "signedAvg.sk_8_1020", "sort.sk_8_52", "tableBasedAddition.sk_240_1024", "tutorial1.sk_1_1", "tutorial2.sk_3_4", "tutorial3.sk_4_31", "UserServiceImpl.sk_8_32", "xpose.sk_6_134"]
################################################################################

crowds_problems_test = ["crowds_big.pm_15steps_8int_7fract_PCTL_TotalRuns=10_CrowdSize=40over", "crowds_big.pm_15steps_8int_7fract_PCTL_TotalRuns=10_CrowdSize=40under", "crowds_big.pm_15steps_8int_7fract_PCTL_TotalRuns=20_CrowdSize=40over", "crowds_big.pm_15steps_8int_7fract_PCTL_TotalRuns=20_CrowdSize=40under", "crowds_big.pm_15steps_9int_7fract_PCTL_TotalRuns=40_CrowdSize=128over", "crowds_big.pm_15steps_9int_7fract_PCTL_TotalRuns=40_CrowdSize=128under", "crowds_big.pm_15steps_9int_7fract_PCTL_TotalRuns=60_CrowdSize=128over", "crowds_big.pm_15steps_9int_7fract_PCTL_TotalRuns=60_CrowdSize=128under", "crowds.pm_15steps_8int_7fract_PCTL_TotalRuns=10_CrowdSize=20over", "crowds.pm_15steps_8int_7fract_PCTL_TotalRuns=10_CrowdSize=20under", "crowds.pm_15steps_8int_7fract_PCTL_TotalRuns=3_CrowdSize=5over", "crowds.pm_15steps_8int_7fract_PCTL_TotalRuns=3_CrowdSize=5under", "crowds.pm_15steps_8int_7fract_PCTL_TotalRuns=6_CrowdSize=10over", "crowds.pm_15steps_8int_7fract_PCTL_TotalRuns=6_CrowdSize=10under"]

egl_problems_test = ["egl.pm_100steps_6int_1fract_unfairA_N=10_L=2over", "egl.pm_100steps_6int_1fract_unfairA_N=10_L=2under", "egl.pm_100steps_6int_1fract_unfairA_N=10_L=4over", "egl.pm_100steps_6int_1fract_unfairA_N=10_L=4under", "egl.pm_100steps_6int_1fract_unfairA_N=7_L=2over", "egl.pm_100steps_6int_1fract_unfairA_N=7_L=2under", "egl.pm_100steps_6int_1fract_unfairA_N=7_L=4over", "egl.pm_100steps_6int_1fract_unfairA_N=7_L=4under", "egl.pm_31steps_6int_1fract_unfairA_N=5_L=2over", "egl.pm_31steps_6int_1fract_unfairA_N=5_L=2under", "egl.pm_60steps_6int_1fract_unfairA_N=5_L=4over", "egl.pm_60steps_6int_1fract_unfairA_N=5_L=4under"]

herman_problems_test = ["herman15.pm_20steps_6int_1fract_stable_over", "herman15.pm_20steps_6int_1fract_stable_under", "herman21.pm_20steps_6int_1fract_stable_over", "herman21.pm_20steps_6int_1fract_stable_under", "herman31.pm_20steps_6int_1fract_stable_over", "herman31.pm_20steps_6int_1fract_stable_under", "herman3.pm_20steps_6int_1fract_stable_over", "herman3.pm_20steps_6int_1fract_stable_under", "herman41.pm_20steps_6int_1fract_stable_over", "herman41.pm_20steps_6int_1fract_stable_under", "herman9.pm_20steps_6int_1fract_stable_over", "herman9.pm_20steps_6int_1fract_stable_under"]

################################################################################

hash_problems_test = ["hash-10-1", "hash-10-2", "hash-10-3", "hash-10-4", "hash-10-5", "hash-10-6", "hash-10-7", "hash-10-8", "hash-10", "hash-11-1", "hash-11-2", "hash-11-3", "hash-11-4", "hash-11-5", "hash-11-6", "hash-11-7", "hash-11-8", "hash-11", "hash-12-1", "hash-12-2", "hash-12-3", "hash-12-4", "hash-12-5", "hash-12-6", "hash-12-7", "hash-12-8", "hash-12", "hash-13-1", "hash-13-2", "hash-13-3", "hash-13-4", "hash-13-5", "hash-13-6", "hash-13-7", "hash-13-8", "hash-14", "hash16-12", "hash16-4", "hash16-8", "hash-16", "hash-2", "hash-4", "hash-6", "hash-8-1", "hash-8-2", "hash-8-3", "hash-8-4", "hash-8-5", "hash-8-6", "hash-8-7", "hash-8-8", "hash-8"]

################################################################################

leader_sync_problems_test = ["leader_sync3_2.pm_4steps_7int_1fract_elected_over", "leader_sync3_2.pm_4steps_7int_1fract_elected_under", "leader_sync3_32.pm_4steps_7int_5fract_elected_over", "leader_sync3_32.pm_4steps_7int_5fract_elected_under", "leader_sync3_64.pm_4steps_7int_6fract_elected_over", "leader_sync3_64.pm_4steps_7int_6fract_elected_under", "leader_sync3_8.pm_4steps_7int_3fract_elected_over", "leader_sync3_8.pm_4steps_7int_3fract_elected_under", "leader_sync4_11.pm_5steps_7int_10fract_elected_neg_over", "leader_sync4_11.pm_5steps_7int_10fract_elected_neg_under", "leader_sync4_11.pm_5steps_7int_10fract_elected_over", "leader_sync4_11.pm_5steps_7int_10fract_elected_under", "leader_sync4_11.pm_5steps_7int_11fract_elected_neg_over", "leader_sync4_11.pm_5steps_7int_11fract_elected_neg_under", "leader_sync4_11.pm_5steps_7int_11fract_elected_over", "leader_sync4_11.pm_5steps_7int_11fract_elected_under", "leader_sync4_11.pm_5steps_7int_12fract_elected_neg_over", "leader_sync4_11.pm_5steps_7int_12fract_elected_neg_under", "leader_sync4_11.pm_5steps_7int_12fract_elected_over", "leader_sync4_11.pm_5steps_7int_12fract_elected_under", "leader_sync4_11.pm_5steps_7int_13fract_elected_neg_over", "leader_sync4_11.pm_5steps_7int_13fract_elected_neg_under", "leader_sync4_11.pm_5steps_7int_13fract_elected_over", "leader_sync4_11.pm_5steps_7int_13fract_elected_under", "leader_sync4_11.pm_5steps_7int_14fract_elected_neg_over", "leader_sync4_11.pm_5steps_7int_14fract_elected_neg_under", "leader_sync4_11.pm_5steps_7int_14fract_elected_over", "leader_sync4_11.pm_5steps_7int_14fract_elected_under", "leader_sync4_11.pm_5steps_7int_15fract_elected_neg_over", "leader_sync4_11.pm_5steps_7int_15fract_elected_neg_under", "leader_sync4_11.pm_5steps_7int_15fract_elected_over", "leader_sync4_11.pm_5steps_7int_15fract_elected_under", "leader_sync4_11.pm_5steps_7int_16fract_elected_neg_over", "leader_sync4_11.pm_5steps_7int_16fract_elected_neg_under", "leader_sync4_11.pm_5steps_7int_16fract_elected_over", "leader_sync4_11.pm_5steps_7int_16fract_elected_under", "leader_sync4_11.pm_5steps_7int_17fract_elected_neg_over", "leader_sync4_11.pm_5steps_7int_17fract_elected_neg_under", "leader_sync4_11.pm_5steps_7int_17fract_elected_over", "leader_sync4_11.pm_5steps_7int_17fract_elected_under", "leader_sync4_11.pm_5steps_7int_18fract_elected_neg_over", "leader_sync4_11.pm_5steps_7int_18fract_elected_neg_under", "leader_sync4_11.pm_5steps_7int_18fract_elected_over", "leader_sync4_11.pm_5steps_7int_18fract_elected_under", "leader_sync4_11.pm_5steps_7int_19fract_elected_neg_over", "leader_sync4_11.pm_5steps_7int_19fract_elected_neg_under", "leader_sync4_11.pm_5steps_7int_19fract_elected_over", "leader_sync4_11.pm_5steps_7int_19fract_elected_under", "leader_sync4_11.pm_5steps_7int_20fract_elected_neg_over", "leader_sync4_11.pm_5steps_7int_20fract_elected_neg_under", "leader_sync4_11.pm_5steps_7int_20fract_elected_over", "leader_sync4_11.pm_5steps_7int_20fract_elected_under", "leader_sync4_11.pm_5steps_7int_4fract_elected_neg_over", "leader_sync4_11.pm_5steps_7int_4fract_elected_neg_under", "leader_sync4_11.pm_5steps_7int_4fract_elected_over", "leader_sync4_11.pm_5steps_7int_4fract_elected_under", "leader_sync4_11.pm_5steps_7int_5fract_elected_neg_over", "leader_sync4_11.pm_5steps_7int_5fract_elected_neg_under", "leader_sync4_11.pm_5steps_7int_5fract_elected_over", "leader_sync4_11.pm_5steps_7int_5fract_elected_under", "leader_sync4_11.pm_5steps_7int_6fract_elected_neg_over", "leader_sync4_11.pm_5steps_7int_6fract_elected_neg_under", "leader_sync4_11.pm_5steps_7int_6fract_elected_over", "leader_sync4_11.pm_5steps_7int_6fract_elected_under", "leader_sync4_11.pm_5steps_7int_7fract_elected_neg_over", "leader_sync4_11.pm_5steps_7int_7fract_elected_neg_under", "leader_sync4_11.pm_5steps_7int_7fract_elected_over", "leader_sync4_11.pm_5steps_7int_7fract_elected_under", "leader_sync4_11.pm_5steps_7int_8fract_elected_neg_over", "leader_sync4_11.pm_5steps_7int_8fract_elected_neg_under", "leader_sync4_11.pm_5steps_7int_8fract_elected_over", "leader_sync4_11.pm_5steps_7int_8fract_elected_under", "leader_sync4_11.pm_5steps_7int_9fract_elected_neg_over", "leader_sync4_11.pm_5steps_7int_9fract_elected_neg_under", "leader_sync4_11.pm_5steps_7int_9fract_elected_over", "leader_sync4_11.pm_5steps_7int_9fract_elected_under", "leader_sync4_2.pm_5steps_7int_1fract_elected_over", "leader_sync4_2.pm_5steps_7int_1fract_elected_under", "leader_sync4_32.pm_5steps_7int_5fract_elected_over", "leader_sync4_32.pm_5steps_7int_5fract_elected_under", "leader_sync4_64.pm_5steps_7int_6fract_elected_over", "leader_sync4_64.pm_5steps_7int_6fract_elected_under", "leader_sync4_8.pm_5steps_7int_3fract_elected_over", "leader_sync4_8.pm_5steps_7int_3fract_elected_under", "leader_sync6_2.pm_7steps_7int_1fract_elected_over", "leader_sync6_2.pm_7steps_7int_1fract_elected_under", "leader_sync6_32.pm_7steps_7int_5fract_elected_over", "leader_sync6_32.pm_7steps_7int_5fract_elected_under", "leader_sync6_64.pm_7steps_7int_6fract_elected_over", "leader_sync6_64.pm_7steps_7int_6fract_elected_under", "leader_sync6_8.pm_7steps_7int_3fract_elected_over", "leader_sync6_8.pm_7steps_7int_3fract_elected_under"]

################################################################################

log_problems_test = ["log-1", "log-2", "log-3", "log-4", "log-5"]

################################################################################

min_problems_test = ["min-12", "min-12s", "min-16", "min-16s", "min-1s", "min-20", "min-20s", "min-24", "min-24s", "min-28", "min-28s", "min-2s", "min-32", "min-32s", "min-3s", "min-4", "min-4s", "min-6s", "min-8", "min-8s"]

################################################################################

modexp_problems_test = ["modexp16-2", "modexp16-4", "modexp8-4-1", "modexp8-4-2", "modexp8-4-3", "modexp8-4-4", "modexp8-4-5", "modexp8-4-6", "modexp8-4-7", "modexp8-4-8", "modexp8-5-1", "modexp8-5-2", "modexp8-5-3", "modexp8-5-4", "modexp8-5-5", "modexp8-5-6", "modexp8-5-7", "modexp8-5-8", "modexp8-6-1", "modexp8-6-2", "modexp8-6-3", "modexp8-6-4", "modexp8-6-5", "modexp8-6-6", "modexp8-6-7", "modexp8-6-8", "modexp8-7-1", "modexp8-7-2", "modexp8-7-3", "modexp8-7-4", "modexp8-7-5", "modexp8-7-6", "modexp8-7-7", "modexp8-7-8", "modexp8-8-1", "modexp8-8-2", "modexp8-8-3", "modexp8-8-4", "modexp8-8-5", "modexp8-8-6", "modexp8-8-7", "modexp8-8-8"]

################################################################################

nand_problems_test = ["nand.pm_100steps_15int_8fract_lessThan10PercentAreErroneous_N=20_K=4over", "nand.pm_100steps_15int_8fract_lessThan10PercentAreErroneous_N=20_K=4under", "nand.pm_80steps_15int_8fract_lessThan10PercentAreErroneous_N=20_K=2over", "nand.pm_80steps_15int_8fract_lessThan10PercentAreErroneous_N=20_K=2under", "nand.pm_80steps_15int_8fract_lessThan10PercentAreErroneous_N=20_K=3over", "nand.pm_80steps_15int_8fract_lessThan10PercentAreErroneous_N=20_K=3under"]

################################################################################

or_100_problems_test =  ["or-100-10-10", "or-100-10-10-UC-10", "or-100-10-10-UC-20", "or-100-10-10-UC-30", "or-100-10-10-UC-40", "or-100-10-10-UC-50", "or-100-10-10-UC-60", "or-100-10-1", "or-100-10-1-UC-10", "or-100-10-1-UC-20", "or-100-10-1-UC-30", "or-100-10-1-UC-40", "or-100-10-1-UC-50", "or-100-10-1-UC-60", "or-100-10-2", "or-100-10-2-UC-10", "or-100-10-2-UC-20", "or-100-10-2-UC-30", "or-100-10-2-UC-40", "or-100-10-2-UC-50", "or-100-10-2-UC-60", "or-100-10-3", "or-100-10-3-UC-10", "or-100-10-3-UC-20", "or-100-10-3-UC-30", "or-100-10-3-UC-40", "or-100-10-3-UC-50", "or-100-10-3-UC-60", "or-100-10-4", "or-100-10-4-UC-10", "or-100-10-4-UC-20", "or-100-10-4-UC-30", "or-100-10-4-UC-40", "or-100-10-4-UC-50", "or-100-10-4-UC-60", "or-100-10-5", "or-100-10-5-UC-10", "or-100-10-5-UC-20", "or-100-10-5-UC-30", "or-100-10-5-UC-40", "or-100-10-5-UC-50", "or-100-10-5-UC-60", "or-100-10-6", "or-100-10-6-UC-10", "or-100-10-6-UC-20", "or-100-10-6-UC-30", "or-100-10-6-UC-40", "or-100-10-6-UC-50", "or-100-10-6-UC-60", "or-100-10-7", "or-100-10-7-UC-10", "or-100-10-7-UC-20", "or-100-10-7-UC-30", "or-100-10-7-UC-40", "or-100-10-7-UC-50", "or-100-10-7-UC-60", "or-100-10-8", "or-100-10-8-UC-10", "or-100-10-8-UC-20", "or-100-10-8-UC-30", "or-100-10-8-UC-40", "or-100-10-8-UC-50", "or-100-10-8-UC-60", "or-100-10-9", "or-100-10-9-UC-10", "or-100-10-9-UC-20", "or-100-10-9-UC-30", "or-100-10-9-UC-40", "or-100-10-9-UC-50", "or-100-10-9-UC-60", "or-100-20-10", "or-100-20-10-UC-10", "or-100-20-10-UC-20", "or-100-20-10-UC-30", "or-100-20-10-UC-40", "or-100-20-10-UC-50", "or-100-20-10-UC-60", "or-100-20-1", "or-100-20-1-UC-10", "or-100-20-1-UC-20", "or-100-20-1-UC-30", "or-100-20-1-UC-40", "or-100-20-1-UC-50", "or-100-20-1-UC-60", "or-100-20-2", "or-100-20-2-UC-10", "or-100-20-2-UC-20", "or-100-20-2-UC-30", "or-100-20-2-UC-40", "or-100-20-2-UC-50", "or-100-20-2-UC-60", "or-100-20-3", "or-100-20-3-UC-10", "or-100-20-3-UC-20", "or-100-20-3-UC-30", "or-100-20-3-UC-40", "or-100-20-3-UC-50", "or-100-20-3-UC-60", "or-100-20-4", "or-100-20-4-UC-10", "or-100-20-4-UC-20", "or-100-20-4-UC-30", "or-100-20-4-UC-40", "or-100-20-4-UC-50", "or-100-20-4-UC-60", "or-100-20-5", "or-100-20-5-UC-10", "or-100-20-5-UC-20", "or-100-20-5-UC-30", "or-100-20-5-UC-40", "or-100-20-5-UC-50", "or-100-20-5-UC-60", "or-100-20-6", "or-100-20-6-UC-10", "or-100-20-6-UC-20", "or-100-20-6-UC-30", "or-100-20-6-UC-40", "or-100-20-6-UC-50", "or-100-20-6-UC-60", "or-100-20-7", "or-100-20-7-UC-10", "or-100-20-7-UC-20", "or-100-20-7-UC-30", "or-100-20-7-UC-40", "or-100-20-7-UC-50", "or-100-20-7-UC-60", "or-100-20-8", "or-100-20-8-UC-10", "or-100-20-8-UC-20", "or-100-20-8-UC-30", "or-100-20-8-UC-40", "or-100-20-8-UC-50", "or-100-20-8-UC-60", "or-100-20-9", "or-100-20-9-UC-10", "or-100-20-9-UC-20", "or-100-20-9-UC-30", "or-100-20-9-UC-40", "or-100-20-9-UC-50", "or-100-20-9-UC-60", "or-100-5-10", "or-100-5-10-UC-10", "or-100-5-10-UC-20", "or-100-5-10-UC-30", "or-100-5-10-UC-40", "or-100-5-10-UC-50", "or-100-5-10-UC-60", "or-100-5-1", "or-100-5-1-UC-10", "or-100-5-1-UC-20", "or-100-5-1-UC-30", "or-100-5-1-UC-40", "or-100-5-1-UC-50", "or-100-5-1-UC-60", "or-100-5-2", "or-100-5-2-UC-10", "or-100-5-2-UC-20", "or-100-5-2-UC-30", "or-100-5-2-UC-40", "or-100-5-2-UC-50", "or-100-5-2-UC-60", "or-100-5-3", "or-100-5-3-UC-10", "or-100-5-3-UC-20", "or-100-5-3-UC-30", "or-100-5-3-UC-40", "or-100-5-3-UC-50", "or-100-5-3-UC-60", "or-100-5-4", "or-100-5-4-UC-10", "or-100-5-4-UC-20", "or-100-5-4-UC-30", "or-100-5-4-UC-40", "or-100-5-4-UC-50", "or-100-5-4-UC-60", "or-100-5-5", "or-100-5-5-UC-10", "or-100-5-5-UC-20", "or-100-5-5-UC-30", "or-100-5-5-UC-40", "or-100-5-5-UC-50", "or-100-5-5-UC-60", "or-100-5-6", "or-100-5-6-UC-10", "or-100-5-6-UC-20", "or-100-5-6-UC-30", "or-100-5-6-UC-40", "or-100-5-6-UC-50", "or-100-5-6-UC-60", "or-100-5-7", "or-100-5-7-UC-10", "or-100-5-7-UC-20", "or-100-5-7-UC-30", "or-100-5-7-UC-40", "or-100-5-7-UC-50", "or-100-5-7-UC-60", "or-100-5-8", "or-100-5-8-UC-10", "or-100-5-8-UC-20", "or-100-5-8-UC-30", "or-100-5-8-UC-40", "or-100-5-8-UC-50", "or-100-5-8-UC-60", "or-100-5-9", "or-100-5-9-UC-10", "or-100-5-9-UC-20", "or-100-5-9-UC-30", "or-100-5-9-UC-40", "or-100-5-9-UC-50", "or-100-5-9-UC-60"]

or_50_problems_test = ["or-50-10-10", "or-50-10-10-UC-10", "or-50-10-10-UC-20", "or-50-10-10-UC-30", "or-50-10-10-UC-40", "or-50-10-1", "or-50-10-1-UC-10", "or-50-10-1-UC-20", "or-50-10-1-UC-30", "or-50-10-1-UC-40", "or-50-10-2", "or-50-10-2-UC-10", "or-50-10-2-UC-20", "or-50-10-2-UC-30", "or-50-10-2-UC-40", "or-50-10-3", "or-50-10-3-UC-10", "or-50-10-3-UC-20", "or-50-10-3-UC-30", "or-50-10-3-UC-40", "or-50-10-4", "or-50-10-4-UC-10", "or-50-10-4-UC-20", "or-50-10-4-UC-30", "or-50-10-4-UC-40", "or-50-10-5", "or-50-10-5-UC-10", "or-50-10-5-UC-20", "or-50-10-5-UC-30", "or-50-10-5-UC-40", "or-50-10-6", "or-50-10-6-UC-10", "or-50-10-6-UC-20", "or-50-10-6-UC-30", "or-50-10-6-UC-40", "or-50-10-7", "or-50-10-7-UC-10", "or-50-10-7-UC-20", "or-50-10-7-UC-30", "or-50-10-7-UC-40", "or-50-10-8", "or-50-10-8-UC-10", "or-50-10-8-UC-20", "or-50-10-8-UC-30", "or-50-10-8-UC-40", "or-50-10-9", "or-50-10-9-UC-10", "or-50-10-9-UC-20", "or-50-10-9-UC-30", "or-50-10-9-UC-40", "or-50-20-10", "or-50-20-10-UC-10", "or-50-20-10-UC-20", "or-50-20-10-UC-30", "or-50-20-10-UC-40", "or-50-20-1", "or-50-20-1-UC-10", "or-50-20-1-UC-20", "or-50-20-1-UC-30", "or-50-20-1-UC-40", "or-50-20-2", "or-50-20-2-UC-10", "or-50-20-2-UC-20", "or-50-20-2-UC-30", "or-50-20-2-UC-40", "or-50-20-3", "or-50-20-3-UC-10", "or-50-20-3-UC-20", "or-50-20-3-UC-30", "or-50-20-3-UC-40", "or-50-20-4", "or-50-20-4-UC-10", "or-50-20-4-UC-20", "or-50-20-4-UC-30", "or-50-20-4-UC-40", "or-50-20-5", "or-50-20-5-UC-10", "or-50-20-5-UC-20", "or-50-20-5-UC-30", "or-50-20-5-UC-40", "or-50-20-6", "or-50-20-6-UC-10", "or-50-20-6-UC-20", "or-50-20-6-UC-30", "or-50-20-6-UC-40", "or-50-20-7", "or-50-20-7-UC-10", "or-50-20-7-UC-20", "or-50-20-7-UC-30", "or-50-20-7-UC-40", "or-50-20-8", "or-50-20-8-UC-10", "or-50-20-8-UC-20", "or-50-20-8-UC-30", "or-50-20-8-UC-40", "or-50-20-9", "or-50-20-9-UC-10", "or-50-20-9-UC-20", "or-50-20-9-UC-30", "or-50-20-9-UC-40", "or-50-5-10", "or-50-5-10-UC-10", "or-50-5-10-UC-20", "or-50-5-10-UC-30", "or-50-5-10-UC-40", "or-50-5-1", "or-50-5-1-UC-10", "or-50-5-1-UC-20", "or-50-5-1-UC-30", "or-50-5-1-UC-40", "or-50-5-2", "or-50-5-2-UC-10", "or-50-5-2-UC-20", "or-50-5-2-UC-30", "or-50-5-2-UC-40", "or-50-5-3", "or-50-5-3-UC-10", "or-50-5-3-UC-20", "or-50-5-3-UC-30", "or-50-5-3-UC-40", "or-50-5-4", "or-50-5-4-UC-10", "or-50-5-4-UC-20", "or-50-5-4-UC-30", "or-50-5-4-UC-40", "or-50-5-5", "or-50-5-5-UC-10", "or-50-5-5-UC-20", "or-50-5-5-UC-30", "or-50-5-5-UC-40", "or-50-5-6", "or-50-5-6-UC-10", "or-50-5-6-UC-20", "or-50-5-6-UC-30", "or-50-5-6-UC-40", "or-50-5-7", "or-50-5-7-UC-10", "or-50-5-7-UC-20", "or-50-5-7-UC-30", "or-50-5-7-UC-40", "or-50-5-8", "or-50-5-8-UC-10", "or-50-5-8-UC-20", "or-50-5-8-UC-30", "or-50-5-8-UC-40", "or-50-5-9", "or-50-5-9-UC-10", "or-50-5-9-UC-20", "or-50-5-9-UC-30", "or-50-5-9-UC-40"]

or_60_problems_test =  ["or-60-10-10", "or-60-10-10-UC-10", "or-60-10-10-UC-20", "or-60-10-10-UC-30", "or-60-10-10-UC-40", "or-60-10-1", "or-60-10-1-UC-10", "or-60-10-1-UC-20", "or-60-10-1-UC-30", "or-60-10-1-UC-40", "or-60-10-2", "or-60-10-2-UC-10", "or-60-10-2-UC-20", "or-60-10-2-UC-30", "or-60-10-2-UC-40", "or-60-10-3", "or-60-10-3-UC-10", "or-60-10-3-UC-20", "or-60-10-3-UC-30", "or-60-10-3-UC-40", "or-60-10-4", "or-60-10-4-UC-10", "or-60-10-4-UC-20", "or-60-10-4-UC-30", "or-60-10-4-UC-40", "or-60-10-5", "or-60-10-5-UC-10", "or-60-10-5-UC-20", "or-60-10-5-UC-30", "or-60-10-5-UC-40", "or-60-10-6", "or-60-10-6-UC-10", "or-60-10-6-UC-20", "or-60-10-6-UC-30", "or-60-10-6-UC-40", "or-60-10-7", "or-60-10-7-UC-10", "or-60-10-7-UC-20", "or-60-10-7-UC-30", "or-60-10-7-UC-40", "or-60-10-8", "or-60-10-8-UC-10", "or-60-10-8-UC-20", "or-60-10-8-UC-30", "or-60-10-8-UC-40", "or-60-10-9", "or-60-10-9-UC-10", "or-60-10-9-UC-20", "or-60-10-9-UC-30", "or-60-10-9-UC-40", "or-60-20-10", "or-60-20-10-UC-10", "or-60-20-10-UC-20", "or-60-20-10-UC-30", "or-60-20-10-UC-40", "or-60-20-1", "or-60-20-1-UC-10", "or-60-20-1-UC-20", "or-60-20-1-UC-30", "or-60-20-1-UC-40", "or-60-20-2", "or-60-20-2-UC-10", "or-60-20-2-UC-20", "or-60-20-2-UC-30", "or-60-20-2-UC-40", "or-60-20-3", "or-60-20-3-UC-10", "or-60-20-3-UC-20", "or-60-20-3-UC-30", "or-60-20-3-UC-40", "or-60-20-4", "or-60-20-4-UC-10", "or-60-20-4-UC-20", "or-60-20-4-UC-30", "or-60-20-4-UC-40", "or-60-20-5", "or-60-20-5-UC-10", "or-60-20-5-UC-20", "or-60-20-5-UC-30", "or-60-20-5-UC-40", "or-60-20-6", "or-60-20-6-UC-10", "or-60-20-6-UC-20", "or-60-20-6-UC-30", "or-60-20-6-UC-40", "or-60-20-7", "or-60-20-7-UC-10", "or-60-20-7-UC-20", "or-60-20-7-UC-30", "or-60-20-7-UC-40", "or-60-20-8", "or-60-20-8-UC-10", "or-60-20-8-UC-20", "or-60-20-8-UC-30", "or-60-20-8-UC-40", "or-60-20-9", "or-60-20-9-UC-10", "or-60-20-9-UC-20", "or-60-20-9-UC-30", "or-60-20-9-UC-40", "or-60-5-10", "or-60-5-10-UC-10", "or-60-5-10-UC-20", "or-60-5-10-UC-30", "or-60-5-10-UC-40", "or-60-5-1", "or-60-5-1-UC-10", "or-60-5-1-UC-20", "or-60-5-1-UC-30", "or-60-5-1-UC-40", "or-60-5-2", "or-60-5-2-UC-10", "or-60-5-2-UC-20", "or-60-5-2-UC-30", "or-60-5-2-UC-40", "or-60-5-3", "or-60-5-3-UC-10", "or-60-5-3-UC-20", "or-60-5-3-UC-30", "or-60-5-3-UC-40", "or-60-5-4", "or-60-5-4-UC-10", "or-60-5-4-UC-20", "or-60-5-4-UC-30", "or-60-5-4-UC-40", "or-60-5-5", "or-60-5-5-UC-10", "or-60-5-5-UC-20", "or-60-5-5-UC-30", "or-60-5-5-UC-40", "or-60-5-6", "or-60-5-6-UC-10", "or-60-5-6-UC-20", "or-60-5-6-UC-30", "or-60-5-6-UC-40", "or-60-5-7", "or-60-5-7-UC-10", "or-60-5-7-UC-20", "or-60-5-7-UC-30", "or-60-5-7-UC-40", "or-60-5-8", "or-60-5-8-UC-10", "or-60-5-8-UC-20", "or-60-5-8-UC-30", "or-60-5-8-UC-40", "or-60-5-9", "or-60-5-9-UC-10", "or-60-5-9-UC-20", "or-60-5-9-UC-30", "or-60-5-9-UC-40"]

or_70_problems_test = ["or-70-10-10", "or-70-10-10-UC-10", "or-70-10-10-UC-20", "or-70-10-10-UC-30", "or-70-10-10-UC-40", "or-70-10-1", "or-70-10-1-UC-10", "or-70-10-1-UC-20", "or-70-10-1-UC-30", "or-70-10-1-UC-40", "or-70-10-2", "or-70-10-2-UC-10", "or-70-10-2-UC-20", "or-70-10-2-UC-30", "or-70-10-2-UC-40", "or-70-10-3", "or-70-10-3-UC-10", "or-70-10-3-UC-20", "or-70-10-3-UC-30", "or-70-10-3-UC-40", "or-70-10-4", "or-70-10-4-UC-10", "or-70-10-4-UC-20", "or-70-10-4-UC-30", "or-70-10-4-UC-40", "or-70-10-5", "or-70-10-5-UC-10", "or-70-10-5-UC-20", "or-70-10-5-UC-30", "or-70-10-5-UC-40", "or-70-10-6", "or-70-10-6-UC-10", "or-70-10-6-UC-20", "or-70-10-6-UC-30", "or-70-10-6-UC-40", "or-70-10-7", "or-70-10-7-UC-10", "or-70-10-7-UC-20", "or-70-10-7-UC-30", "or-70-10-7-UC-40", "or-70-10-8", "or-70-10-8-UC-10", "or-70-10-8-UC-20", "or-70-10-8-UC-30", "or-70-10-8-UC-40", "or-70-10-9", "or-70-10-9-UC-10", "or-70-10-9-UC-20", "or-70-10-9-UC-30", "or-70-10-9-UC-40", "or-70-20-10", "or-70-20-10-UC-10", "or-70-20-10-UC-20", "or-70-20-10-UC-30", "or-70-20-10-UC-40", "or-70-20-1", "or-70-20-1-UC-10", "or-70-20-1-UC-20", "or-70-20-1-UC-30", "or-70-20-1-UC-40", "or-70-20-2", "or-70-20-2-UC-10", "or-70-20-2-UC-20", "or-70-20-2-UC-30", "or-70-20-2-UC-40", "or-70-20-3", "or-70-20-3-UC-10", "or-70-20-3-UC-20", "or-70-20-3-UC-30", "or-70-20-3-UC-40", "or-70-20-4", "or-70-20-4-UC-10", "or-70-20-4-UC-20", "or-70-20-4-UC-30", "or-70-20-4-UC-40", "or-70-20-5", "or-70-20-5-UC-10", "or-70-20-5-UC-20", "or-70-20-5-UC-30", "or-70-20-5-UC-40", "or-70-20-6", "or-70-20-6-UC-10", "or-70-20-6-UC-20", "or-70-20-6-UC-30", "or-70-20-6-UC-40", "or-70-20-7", "or-70-20-7-UC-10", "or-70-20-7-UC-20", "or-70-20-7-UC-30", "or-70-20-7-UC-40", "or-70-20-8", "or-70-20-8-UC-10", "or-70-20-8-UC-20", "or-70-20-8-UC-30", "or-70-20-8-UC-40", "or-70-20-9", "or-70-20-9-UC-10", "or-70-20-9-UC-20", "or-70-20-9-UC-30", "or-70-20-9-UC-40", "or-70-5-10", "or-70-5-10-UC-10", "or-70-5-10-UC-20", "or-70-5-10-UC-30", "or-70-5-10-UC-40", "or-70-5-1", "or-70-5-1-UC-10", "or-70-5-1-UC-20", "or-70-5-1-UC-30", "or-70-5-1-UC-40", "or-70-5-2", "or-70-5-2-UC-10", "or-70-5-2-UC-20", "or-70-5-2-UC-30", "or-70-5-2-UC-40", "or-70-5-3", "or-70-5-3-UC-10", "or-70-5-3-UC-20", "or-70-5-3-UC-30", "or-70-5-3-UC-40", "or-70-5-4", "or-70-5-4-UC-10", "or-70-5-4-UC-20", "or-70-5-4-UC-30", "or-70-5-4-UC-40", "or-70-5-5", "or-70-5-5-UC-10", "or-70-5-5-UC-20", "or-70-5-5-UC-30", "or-70-5-5-UC-40", "or-70-5-6", "or-70-5-6-UC-10", "or-70-5-6-UC-20", "or-70-5-6-UC-30", "or-70-5-6-UC-40", "or-70-5-7", "or-70-5-7-UC-10", "or-70-5-7-UC-20", "or-70-5-7-UC-30", "or-70-5-7-UC-40", "or-70-5-8", "or-70-5-8-UC-10", "or-70-5-8-UC-20", "or-70-5-8-UC-30", "or-70-5-8-UC-40", "or-70-5-9", "or-70-5-9-UC-10", "or-70-5-9-UC-20", "or-70-5-9-UC-30", "or-70-5-9-UC-40"]

################################################################################

prod_problems_test = ["prod-16", "prod-1s", "prod-20", "prod-24", "prod-28", "prod-2", "prod-2s", "prod-32", "prod-3s", "prod-4", "prod-4s", "prod-8", "prod-8s"]

################################################################################

s_problems_test = ["s1196a_15_7", "s1196a_3_2", "s1196a_7_4", "s1238a_15_7", "s1238a_3_2", "s1238a_7_4", "s13207a_15_7", "s13207a_3_2", "s13207a_7_4", "s1423a_15_7", "s1423a_3_2", "s1423a_7_4", "s1488_15_7", "s1488_3_2", "s1488_7_4", "s15850a_15_7", "s15850a_3_2", "s15850a_7_4", "s27_15_7", "s27_3_2", "s27_7_4", "s27_new_15_7", "s27_new_3_2", "s27_new_7_4", "s298_15_7", "s298_3_2", "s298_7_4", "s344_15_7", "s344_3_2", "s344_7_4", "s349_15_7", "s349_3_2", "s349_7_4", "s35932_15_7", "s35932_3_2", "s35932_7_4", "s382_15_7", "s382_3_2", "s382_7_4", "s38417_15_7", "s38417_3_2", "s38417_7_4", "s38584_15_7", "s38584_3_2", "s38584_7_4", "s420_15_7", "s420_3_2", "s420_7_4", "s420_new1_15_7", "s420_new1_3_2", "s420_new_15_7", "s420_new1_7_4", "s420_new_3_2", "s420_new_7_4", "s444_15_7", "s444_3_2", "s444_7_4", "s510_15_7", "s510_3_2", "s510_7_4", "s526_15_7", "s526_3_2", "s526_7_4", "s526a_15_7", "s526a_3_2", "s526a_7_4", "s5378a_15_7", "s5378a_3_2", "s5378a_7_4", "s641_15_7", "s641_3_2", "s641_7_4", "s713_15_7", "s713_3_2", "s713_7_4", "s820a_15_7", "s820a_3_2", "s820a_7_4", "s832a_15_7", "s832a_3_2", "s832a_7_4", "s838_15_7", "s838_3_2", "s838_7_4", "s9234a_15_7", "s9234a_3_2", "s9234a_7_4", "s953a_15_7", "s953a_3_2", "s953a_7_4"]

################################################################################

tire_problems_test = ["tire-1", "tire-2", "tire-3", "tire-4"]

################################################################################
##### Current Training SAT Problems
################################################################################
number_letter_number_problems_train = ["01A-1","01B-1","01B-2","01B-3","01B-4","01B-5","02A-1","02A-2","02A-3","02B-1","02B-2","02B-3","02B-4","02B-5","03A-1","03A-2","03B-1","03B-2","03B-3","03B-4","04A-1","04A-2","04A-3","04B-1","04B-2","04B-3","04B-4","05A-1","05A-2","05B-1","05B-2","05B-3","06A-1","06A-2","06A-3","06A-4","06B-1","06B-2","06B-3","06B-4","07A-1","07A-2","07A-3","07A-4","07A-5","07B-1","07B-2","07B-3","07B-4","07B-5","07B-6","08A-1","08A-2","08A-3","08A-4","08B-1","08B-2","08B-3","08B-4","09A-1","09A-2","09A-3","09B-1","09B-2","09B-3","09B-4","09B-5","09B-6","10A-1","10A-2","10A-3","10A-4","10B-10","10B-11","10B-1","10B-2","10B-3","10B-4","10B-5","10B-6","10B-7","10B-8","10B-9","11A-1","11A-2","11A-3","11A-4","11B-1","11B-2","11B-3","11B-4","11B-5","12A-1","12A-2","12A-3","12A-4","12B-1","12B-2","12B-3","12B-4","12B-5","12B-6","13A-1","13A-2","13A-3","13A-4","13B-1","13B-2","13B-3","13B-4","13B-5","14A-1","14A-2","14A-3","15A-1","15A-2","15A-3","15A-4","15B-1","15B-2","15B-3","15B-4","15B-5","17A-1","17A-2","17A-3","17A-4","17A-5","17A-6","17B-1","17B-2","17B-3","17B-4","17B-5", "18A-1","18A-2","18A-3","18A-4"]

################################################################################

sk_problems_train = ["107.sk_3_90","109.sk_4_36","17.sk_3_45","19.sk_3_48","20.sk_1_51","27.sk_3_32","29.sk_3_45","30.sk_5_76","32.sk_4_38","35.sk_3_52","36.sk_3_77",
"51.sk_4_38","53.sk_4_32","55.sk_3_46","56.sk_6_38","57.sk_4_64","5step","63.sk_3_64","70.sk_3_40","71.sk_3_65",
"77.sk_3_44","79.sk_4_40","7.sk_4_50","80.sk_2_48","81.sk_5_51","84.sk_4_77",
"ActivityService2.sk_10_27","ActivityService.sk_11_27","10.sk_1_46","110.sk_3_88","111.sk_2_36"]

################################################################################

problem_4step_train = ["4step"]

################################################################################

problems_50_train = ["50-10-10-q","50-10-1-q","50-10-2-q","50-10-3-q","50-10-4-q","50-10-5-q","50-10-6-q","50-10-7-q","50-10-8-q","50-10-9-q","50-12-10-q","50-12-1-q","50-12-2-q","50-12-3-q","50-12-4-q","50-12-5-q","50-12-6-q","50-12-7-q","50-12-8-q","50-12-9-q","50-14-10-q","50-14-1-q","50-14-2-q","50-14-3-q","50-14-4-q","50-14-5-q","50-14-6-q","50-14-7-q","50-14-8-q","50-14-9-q","50-16-10-q","50-16-1-q","50-16-2-q","50-16-3-q","50-16-4-q","50-16-5-q","50-16-6-q","50-16-7-q","50-16-8-q","50-16-9-q","50-18-10-q","50-18-1-q","50-18-2-q","50-18-3-q","50-18-4-q","50-18-5-q","50-18-6-q","50-18-7-q","50-18-8-q","50-18-9-q","50-20-10-q","50-20-1-q","50-20-2-q","50-20-3-q","50-20-4-q","50-20-5-q","50-20-6-q","50-20-7-q","50-20-8-q","50-20-9-q"]

problems_75_train = ["75-10-10-q","75-10-1-q","75-10-2-q","75-10-3-q","75-10-4-q","75-10-5-q","75-10-6-q","75-10-7-q","75-10-8-q","75-10-9-q","75-12-10-q","75-12-1-q","75-12-2-q","75-12-3-q","75-12-4-q","75-12-5-q","75-12-6-q","75-12-7-q","75-12-8-q","75-12-9-q","75-14-10-q","75-14-1-q","75-14-2-q","75-14-3-q","75-14-4-q","75-14-5-q","75-14-6-q","75-14-7-q","75-14-8-q","75-14-9-q","75-15-10-q","75-15-1-q","75-15-2-q","75-15-3-q","75-15-4-q","75-15-5-q","75-15-6-q","75-15-7-q","75-15-8-q","75-15-9-q","75-16-10-q","75-16-1-q","75-16-2-q","75-16-3-q","75-16-4-q","75-16-5-q","75-16-6-q","75-16-7-q","75-16-8-q","75-16-9-q","75-17-10-q","75-17-1-q","75-17-2-q","75-17-3-q","75-17-4-q","75-17-5-q","75-17-6-q","75-17-7-q","75-17-8-q","75-17-9-q","75-18-10-q","75-18-1-q","75-18-2-q","75-18-3-q","75-18-4-q","75-18-5-q","75-18-6-q","75-18-7-q","75-18-8-q","75-18-9-q","75-19-10-q","75-19-1-q","75-19-2-q","75-19-3-q","75-19-4-q","75-19-5-q","75-19-6-q","75-19-7-q","75-19-8-q","75-19-9-q","75-20-10-q","75-20-1-q","75-20-2-q","75-20-3-q","75-20-4-q","75-20-5-q","75-20-6-q","75-20-7-q","75-20-8-q","75-20-9-q","75-21-10-q","75-21-1-q","75-21-2-q","75-21-3-q","75-21-4-q","75-21-5-q","75-21-6-q","75-21-7-q","75-21-8-q","75-21-9-q","75-22-10-q","75-22-1-q","75-22-2-q","75-22-3-q","75-22-4-q","75-22-5-q","75-22-6-q","75-22-7-q","75-22-8-q","75-22-9-q","75-23-10-q","75-23-1-q","75-23-2-q","75-23-3-q","75-23-4-q","75-23-5-q","75-23-6-q","75-23-7-q","75-23-8-q","75-23-9-q","75-24-10-q","75-24-1-q","75-24-2-q","75-24-3-q","75-24-4-q","75-24-5-q","75-24-6-q","75-24-7-q","75-24-8-q","75-24-9-q","75-25-10-q","75-25-1-q","75-25-2-q","75-25-3-q","75-25-4-q","75-25-5-q","75-25-6-q","75-25-7-q","75-25-8-q","75-25-9-q","75-26-10-q","75-26-1-q","75-26-2-q","75-26-3-q","75-26-4-q","75-26-5-q","75-26-6-q","75-26-7-q","75-26-8-q","75-26-9-q"]

problems_90_train = ["90-10-10-q","90-10-1-q","90-10-2-q","90-10-3-q","90-10-4-q","90-10-5-q","90-10-6-q","90-10-7-q","90-10-8-q","90-10-9-q","90-12-10-q","90-12-1-q","90-12-2-q","90-12-3-q","90-12-4-q","90-12-5-q","90-12-6-q","90-12-7-q","90-12-8-q","90-12-9-q","90-14-10-q","90-14-1-q","90-14-2-q","90-14-3-q","90-14-4-q","90-14-5-q","90-14-6-q","90-14-7-q","90-14-8-q","90-14-9-q","90-15-10-q","90-15-1-q","90-15-2-q","90-15-3-q","90-15-4-q","90-15-5-q","90-15-6-q","90-15-7-q","90-15-8-q","90-15-9-q","90-16-10-q","90-16-1-q","90-16-2-q","90-16-3-q","90-16-4-q","90-16-5-q","90-16-6-q","90-16-7-q","90-16-8-q","90-16-9-q","90-17-10-q","90-17-1-q","90-17-2-q","90-17-3-q","90-17-4-q","90-17-5-q","90-17-6-q","90-17-7-q","90-17-8-q","90-17-9-q","90-18-10-q","90-18-1-q","90-18-2-q","90-18-3-q","90-18-4-q","90-18-5-q","90-18-6-q","90-18-7-q","90-18-8-q","90-18-9-q","90-19-10-q","90-19-1-q","90-19-2-q","90-19-3-q","90-19-4-q","90-19-5-q","90-19-6-q","90-19-7-q","90-19-8-q","90-19-9-q","90-20-10-q","90-20-1-q","90-20-2-q","90-20-3-q","90-20-4-q","90-20-5-q","90-20-6-q","90-20-7-q","90-20-8-q","90-20-9-q","90-21-10-q","90-21-1-q","90-21-2-q","90-21-3-q","90-21-4-q","90-21-5-q","90-21-6-q","90-21-7-q","90-21-8-q","90-21-9-q","90-22-10-q","90-22-1-q","90-22-2-q","90-22-3-q","90-22-4-q","90-22-5-q","90-22-6-q","90-22-7-q","90-22-8-q","90-22-9-q","90-23-10-q","90-23-1-q","90-23-2-q","90-23-3-q","90-23-4-q","90-23-5-q","90-23-6-q","90-23-7-q","90-23-8-q","90-23-9-q","90-24-10-q","90-24-1-q","90-24-2-q","90-24-3-q","90-24-4-q","90-24-5-q","90-24-6-q","90-24-7-q","90-24-8-q","90-24-9-q","90-25-10-q","90-25-1-q","90-25-2-q","90-25-3-q","90-25-4-q","90-25-5-q","90-25-6-q","90-25-7-q","90-25-8-q","90-25-9-q","90-26-10-q","90-26-1-q","90-26-2-q","90-26-3-q","90-26-4-q","90-26-5-q","90-26-6-q","90-26-7-q","90-26-8-q","90-26-9-q","90-30-10-q","90-30-1-q","90-30-2-q","90-30-3-q","90-30-4-q","90-30-5-q","90-30-6-q","90-30-7-q","90-30-8-q","90-30-9-q","90-34-10-q","90-34-1-q","90-34-2-q","90-34-3-q","90-34-4-q","90-34-5-q","90-34-6-q","90-34-7-q","90-34-8-q","90-34-9-q","90-38-10-q","90-38-1-q","90-38-2-q","90-38-3-q","90-38-4-q","90-38-5-q","90-38-6-q","90-38-7-q","90-38-8-q","90-38-9-q","90-42-10-q","90-42-1-q","90-42-2-q","90-42-3-q","90-42-4-q","90-42-5-q","90-42-6-q","90-42-7-q","90-42-8-q","90-42-9-q","90-46-10-q","90-46-1-q","90-46-2-q","90-46-3-q","90-46-4-q","90-46-5-q","90-46-6-q","90-46-7-q","90-46-8-q","90-46-9-q","90-50-10-q","90-50-1-q","90-50-2-q","90-50-3-q","90-50-4-q","90-50-5-q","90-50-6-q","90-50-7-q","90-50-8-q","90-50-9-q"]


dict_of_test_problems = {"blasted_problems": blasted_problems_test, "brp_problems": brp_problems_test, "sk_problems": sk_problems_test, "crowds_problems": crowds_problems_test, "egl_problems": egl_problems_test, "herman_problems": herman_problems_test, "hash_problems": hash_problems_test, "leader_sync_problems": leader_sync_problems_test, "log_problems": log_problems_test, "min_problems": min_problems_test, "modexp_problems": modexp_problems_test, "nand_problems": nand_problems_test, "or_100_problems": or_100_problems_test, "or_50_problems": or_50_problems_test, "or_60_problems": or_60_problems_test, "or_70_problems": or_70_problems_test, "prod_problems": prod_problems_test, "s_problems": s_problems_test, "tire_problems": tire_problems_test}



dict_of_training_problems = {"number_letter_number_problems": number_letter_number_problems_train, "sk_problems": sk_problems_train, "problem_4step": problem_4step_train, "problems_50": problems_50_train, "problems_75": problems_75_train, "problems_90": problems_90_train}

dict_of_all_problems = {"blasted_problems": blasted_problems_test, "brp_problems": brp_problems_test, "crowds_problems": crowds_problems_test, "egl_problems": egl_problems_test, "herman_problems": herman_problems_test, "hash_problems": hash_problems_test, "leader_sync_problems": leader_sync_problems_test, "log_problems": log_problems_test, "min_problems": min_problems_test, "modexp_problems": modexp_problems_test, "nand_problems": nand_problems_test, "or_100_problems": or_100_problems_test, "or_50_problems": or_50_problems_test, "or_60_problems": or_60_problems_test, "or_70_problems": or_70_problems_test, "prod_problems": prod_problems_test, "s_problems": s_problems_test, "tire_problems": tire_problems_test, "number_letter_number_problems": number_letter_number_problems_train, "sk_problems": sk_problems_train + sk_problems_test, "problem_4step": problem_4step_train, "problems_50": problems_50_train, "problems_75": problems_75_train, "problems_90": problems_90_train}

def get_max_vars_in_clause(filename):
    '''
    Return the maximum clause length (number of variables in the largest clause)
    for the SAT problem specified by filename (in cnf form)
    '''
    max_len = 0
    with open(filename, 'r') as f:    
        for line in f:
            line_as_list = line.strip().split()
            if len(line_as_list) == 0:
                continue
            # print("line_as_list[:-1]:")
            # print(line_as_list[:-1])
            # print()            
            if line_as_list[0] == "p":
                continue
            elif line_as_list[0] == "c":
                continue
            else:
                cur_clause = [int(s) for s in line_as_list[:-1]]
                clause_len = len(cur_clause)
                if clause_len > max_len:
                    max_len = clause_len
    return max_len


if __name__ == "__main__":

    #testing problems
#     exactSolver_completed_problems = {"blasted_problems": 0,"brp_problems": 0,"sk_problems": 0,"crowds_problems": 0,"egl_problems": 0,"herman_problems": 0,"hash_problems": 0,"leader_sync_problems": 0,"log_problems": 0,"min_problems": 0,"modexp_problems": 0,"nand_problems": 0,"or_100_problems": 0,"or_50_problems": 0,"or_60_problems": 0,"or_70_problems": 0,"prod_problems": 0,"s_problems": 0,"tire_problems": 0}

    #training problems
#     exactSolver_completed_problems = {"number_letter_number_problems": 0, "sk_problems": 0, "problem_4step": 0, "problems_50": 0, "problems_75": 0, "problems_90": 0}
    
    #all problems
    exactSolver_completed_problems = {"blasted_problems": 0,"brp_problems": 0,"crowds_problems": 0,"egl_problems": 0,"herman_problems": 0,"hash_problems": 0,"leader_sync_problems": 0,"log_problems": 0,"min_problems": 0,"modexp_problems": 0,"nand_problems": 0,"or_100_problems": 0,"or_50_problems": 0,"or_60_problems": 0,"or_70_problems": 0,"prod_problems": 0,"s_problems": 0,"tire_problems": 0, "number_letter_number_problems": 0, "sk_problems": 0, "problem_4step": 0, "problems_50": 0, "problems_75": 0, "problems_90": 0}


    problemsSolvedExactly_byCategory = defaultdict(list)
    
    exactSolver_timeouts = 0
    exactSolver_completions = 0
#     dsharp_timeouts = 0
#     dsharp_completions = 0
    F2_varDeg3_timeouts = 0
    F2_varDeg3_completions = 0
    completions_by_F2_and_exactSolver = 0
    
    exact_counts_directory = './exact_SAT_counts_noIndSets/'
    approxMC_estimates_directory = './SATestimates_approxMC/'
#     approxMC_estimates_directory = './SATestimates_approxMC_origData/'


    exact_sat_counts = []
    exact_sat_counts_1 = [] #for plotting varDeg6
    exact_sat_counts_2 = [] #for plotting approxMC
    
    F2_varDeg3_lower_bounds_errors = []
    F2_varDeg6_lower_bounds_errors = []
    approxMC_solCountEstimate_errors = []
    
    exactSolver_times_by_category = defaultdict(list)
    F2_varDeg3_times_by_category = defaultdict(list)

    new_test_problems = {}
    new_train_problems = {}
    
    approxMC_completed_count = 0
    for filename in os.listdir(exact_counts_directory):
        if os.path.isfile(approxMC_estimates_directory + filename):
#             print("found file:", approxMC_estimates_directory + filename, ":):):):):):):)")
            approxMC_solCountEstimate = None
            with open(approxMC_estimates_directory + filename, 'r') as f_approxMC:
#                 print("opening file314")
                for line in f_approxMC:
                    if line.startswith('approxMC time_out:'):
                        print("found line 1 :)")
                        if line.split()[2] == "True":
                            timeout_noIndSet = True
                        else:
                            timeout_noIndSet = False
                            assert(line.split()[2] == "False")
                            
#                         print("timeout:", timeout_noIndSet)
#                         print("line:", line)
#                         print()
                        if not timeout_noIndSet and line.split()[6] != 'None':
                            approxMC_time = float(line.split()[6])
                            approxMC_solCountEstimate = float(line.split()[4])
                        elif not timeout_noIndSet:
                            print("approxMC did not timeout, but solution count is None??")
                    if line.startswith('approxMC_runWithIndependentSet time_out:'):
                        print("found line 2 :)")                        
                        if line.split()[2] == "True":
                            timeout_withIndSet = True
                        else:
                            timeout_withIndSet = False
                            assert(line.split()[2] == "False")                        
                        if not timeout_withIndSet:                        
                            approxMC_time_withIndSet = float(line.split()[6]) + float(line.split()[10])
                            approxMC_solCountEstimate_withIndSet = float(line.split()[4])
                            if approxMC_solCountEstimate is None:
                                approxMC_time = approxMC_time_withIndSet
                                approxMC_solCountEstimate = approxMC_solCountEstimate_withIndSet
                            elif approxMC_time_withIndSet < approxMC_time:
                                approxMC_time = approxMC_time_withIndSet
                                approxMC_solCountEstimate = approxMC_solCountEstimate_withIndSet
                                assert(np.abs(approxMC_solCountEstimate - approxMC_solCountEstimate_withIndSet) < 8), ("approxMC with and without independent sets differs by more than expected:", approxMC_solCountEstimate, approxMC_solCountEstimate_withIndSet)
                                print("approxMC with independed set is faster than without")
                            else:
                                print("approxMC with independed set is SLOWER than without")
                print("approxMC_solCountEstimate =", approxMC_solCountEstimate)
            if approxMC_solCountEstimate is not None:
                approxMC_completed_count += 1
        else:
            print("could not find file:", approxMC_estimates_directory + filename)
            
        if filename.endswith(".txt"): 
            with open(exact_counts_directory + filename, 'r') as f_solution_count:
#                 print("filename:", filename)
                cur_problem_category = None
#                 for problem_category, problem_names in dict_of_test_problems.items():
#                 for problem_category, problem_names in dict_of_training_problems.items():   
                for problem_category, problem_names in dict_of_all_problems.items():   
                    if filename[:-4] in problem_names:
                        cur_problem_category = problem_category
                        break
                if cur_problem_category is None:
                    continue
#                 assert(cur_problem_category is not None), "Error: problem not found!!!"
                
#                 sleep(sld)
                sharpSAT_solution_count = None
                dsharp_solution_count = None
                F2_varDeg3_log_lowerBound = None
                F2_varDeg6_log_lowerBound = None
            
                for line in f_solution_count:
                    if line.strip().split(" ")[0] == 'sharpSAT':
                        sharpSAT_solution_count = Decimal(line.strip().split(" ")[4])
                        if Decimal.is_nan(sharpSAT_solution_count):
                            sharpSAT_solution_count = None
                        sharpSAT_time = float(line.strip().split(" ")[6])
                    if line.strip().split(" ")[0] == 'dsharp':
                        dsharp_solution_count = Decimal(line.strip().split(" ")[4])
                        if Decimal.is_nan(dsharp_solution_count):
                            dsharp_solution_count = None 
                        dsharp_time = float(line.strip().split(" ")[6])
                    if line.strip().split(" ")[0] == 'biregular_variable_degree_3_Tsol_1' and line.strip().split(" ")[2] == 'False': #not a timeout
                        F2_varDeg3_log_lowerBound = float(line.strip().split(" ")[4])
                        F2_varDeg3_time = float(line.strip().split(" ")[6])

                    if line.strip().split(" ")[0] == 'biregular_variable_degree_6_Tsol_1' and line.strip().split(" ")[2] == 'False': #not a timeout
                        F2_varDeg6_log_lowerBound = float(line.strip().split(" ")[4])
                    
                    
                USE_DSHARP_ONLY = True
                if USE_DSHARP_ONLY:
                    if (dsharp_solution_count is not None):
                        exact_solution_count = dsharp_solution_count
                    else:
                        exact_solution_count = None                    

                    if exact_solution_count is None:
                        exactSolver_timeouts += 1
                    else:
                        exactSolver_completions += 1 
                        exactSolver_completed_problems[cur_problem_category] += 1
                        exactSolver_time = dsharp_time
                        exactSolver_times_by_category[cur_problem_category].append((exactSolver_time, filename))
                        problemsSolvedExactly_byCategory[cur_problem_category].append(filename)

                    
                else:
                    if (sharpSAT_solution_count is not None) and (dsharp_solution_count is not None):
    #                     assert(sharpSAT_solution_count == dsharp_solution_count), (sharpSAT_solution_count, dsharp_solution_count)
                        if(sharpSAT_solution_count != dsharp_solution_count):
                            difference = float(sharpSAT_solution_count.ln())/np.log(2) - float(dsharp_solution_count.ln())/np.log(2)
                            print("sharpSAT and dsharp disagree on", filename, "by:", difference)
                        exact_solution_count = sharpSAT_solution_count
                    elif (sharpSAT_solution_count is not None):
                        exact_solution_count = sharpSAT_solution_count
                    elif (dsharp_solution_count is not None):
                        exact_solution_count = dsharp_solution_count
                    else:
                        exact_solution_count = None                    

                    if exact_solution_count is None:
                        exactSolver_timeouts += 1
                    else:
                        exactSolver_completions += 1 
                        exactSolver_completed_problems[cur_problem_category] += 1
                        exactSolver_time = min(dsharp_time, sharpSAT_time)
                        if dsharp_time < sharpSAT_time:
    #                         print("dsharp faster")
                            pass
                        else:
    #                         print("sharpSAT faster") 
                            pass
                        exactSolver_times_by_category[cur_problem_category].append((exactSolver_time, filename))
                        problemsSolvedExactly_byCategory[cur_problem_category].append(filename)

                if F2_varDeg3_log_lowerBound is None:
                    F2_varDeg3_timeouts += 1
                else:
                    F2_varDeg3_completions += 1
                    
                if (exact_solution_count is not None) and (F2_varDeg3_log_lowerBound is not None):
                    completions_by_F2_and_exactSolver += 1
                    exact_sat_counts.append(float(exact_solution_count.ln())/np.log(2))
                    F2_varDeg3_lower_bounds_errors.append(F2_varDeg3_log_lowerBound - float(exact_solution_count.ln())/np.log(2))
                    F2_varDeg3_times_by_category[cur_problem_category].append((F2_varDeg3_time, filename))
             
                if (exact_solution_count is not None) and (F2_varDeg6_log_lowerBound is not None):
                    exact_sat_counts_1.append(float(exact_solution_count.ln())/np.log(2))
                    F2_varDeg6_lower_bounds_errors.append(F2_varDeg6_log_lowerBound - float(exact_solution_count.ln())/np.log(2))
                
                if (exact_solution_count is not None) and (approxMC_solCountEstimate is not None):
                    exact_sat_counts_2.append(float(exact_solution_count.ln())/np.log(2))
                    approxMC_solCountEstimate_errors.append(approxMC_solCountEstimate - float(exact_solution_count.ln())/np.log(2))
                                    
                
                if (F2_varDeg3_log_lowerBound is not None) and (F2_varDeg6_log_lowerBound is not None) and (F2_varDeg6_log_lowerBound < F2_varDeg3_log_lowerBound - 5):
                    print("deg6 LB < deg3 LB:", F2_varDeg6_log_lowerBound, F2_varDeg3_log_lowerBound, filename)
    print("exactSolver_timeouts:", exactSolver_timeouts)
    print("exactSolver_completions:", exactSolver_completions)
    print("exactSolver_timeouts + exactSolver_completions =", exactSolver_timeouts + exactSolver_completions)
    print("F2_varDeg3_timeouts:", F2_varDeg3_timeouts)
    print("F2_varDeg3_completions:", F2_varDeg3_completions)
    print("F2_varDeg3_timeouts + F2_varDeg3_completions =", F2_varDeg3_timeouts + F2_varDeg3_completions)
    print("completions_by_F2_and_exactSolver:", completions_by_F2_and_exactSolver)

    
    
#     print("problemsSolvedExactly_byCategory:")
#     print(problemsSolvedExactly_byCategory)
    
    print("exactSolver_times_by_category:")
    check_problems_solved_by_exactSolver = 0
    problems_solved_under2 = []
    problems_solved_over2 = []    
    #only consider problems with <= MAX_CLAUSE_DEGREE variables appearing in every clause
    MAX_CLAUSE_DEGREE = 5
    test_problems = {}
    for problem_category, list_of_times in exactSolver_times_by_category.items():
        times_only = [time for (time, file) in list_of_times]
            
        print(len(list_of_times), "problems in category", problem_category, "min time:", np.min(times_only), "max time:", np.max(times_only), "mean time:", np.mean(times_only))
        for time, file in list_of_times:
            if time < 2:
                problems_solved_under2.append(file)
            else:
                problems_solved_over2.append(file)                
        list_of_times.sort(key = operator.itemgetter(0))
#         print(list_of_times)
#         print()
        check_problems_solved_by_exactSolver += len(list_of_times)
    
        pruned_list_of_times = []
        for (time, file) in list_of_times:
            sat_formula_file = "./sat_problems_noIndSets/" + file.split('.txt')[0] + ".cnf.gz.no_w.cnf"
            cur_problem_max_clause_degree = get_max_vars_in_clause(sat_formula_file)
            if cur_problem_max_clause_degree <= MAX_CLAUSE_DEGREE:
                pruned_list_of_times.append({'dsharp_time':time, 'problem':file.split('.txt')[0]})
        list_of_times = pruned_list_of_times
        times_only = [problem['dsharp_time'] for problem in pruned_list_of_times]
        if len(list_of_times) > 0:
            print(len(list_of_times), "problems in category", problem_category, "with <=", MAX_CLAUSE_DEGREE, "variables in the largest clause, min time:", np.min(times_only), "max time:", np.max(times_only), "mean time:", np.mean(times_only))
        else:
            print("no problems with <=", MAX_CLAUSE_DEGREE, "variables in the largest clause")
            
        random.shuffle(list_of_times)
        if len(list_of_times) > 10:
            new_test_problems[problem_category] = list_of_times[:len(list_of_times)*3//10]
            new_test_problems[problem_category].sort(key = lambda i: i['dsharp_time'])
            new_train_problems[problem_category] = list_of_times[len(list_of_times)*3//10:]
            new_train_problems[problem_category].sort(key = lambda i: i['dsharp_time'])

            
        elif len(list_of_times) > 0:
            new_test_problems[problem_category] = list_of_times
            new_test_problems[problem_category].sort(key = lambda i: i['dsharp_time'])

        print()

        
    
    print("check_problems_solved_by_exactSolver =", check_problems_solved_by_exactSolver)
    print()
#     print("problems_solved_under2:", problems_solved_under2)
#     print("problems_solved_over2:", problems_solved_over2)    
    print('#'*80)
    
    print("F2_varDeg3_times_by_category:")
    check_problems_solved_by_F2 = 0
    for problem_category, list_of_times in F2_varDeg3_times_by_category.items():
        times_only = [time for (time, file) in list_of_times]        
        print(len(list_of_times), "problems in category", problem_category, "min time:", np.min(times_only), "max time:", np.max(times_only), "mean time:", np.mean(times_only))
        check_problems_solved_by_F2 += len(list_of_times)
        list_of_times.sort(key = operator.itemgetter(0))
#         print(list_of_times)
#         print()
        
    print("check_problems_solved_by_F2 =", check_problems_solved_by_F2)
    print()

    plt.plot([min(exact_sat_counts), max(exact_sat_counts)], [0, 0], '-', c='g', label='Exact')

    plt.plot(exact_sat_counts, F2_varDeg3_lower_bounds_errors, 'x', c='g', label='F2 lower, var_deg=3')
    plt.plot(exact_sat_counts_1, F2_varDeg6_lower_bounds_errors, '+', c='b', label='F2 lower, var_deg=6')
    plt.plot(exact_sat_counts_2, approxMC_solCountEstimate_errors, '2', c='r', label='ApproxMC')
    print("approxMC completed", len(approxMC_solCountEstimate_errors), "problems that the exact solver also completed")
    print("approxMC completed", approxMC_completed_count, "problems")

    



    # plt.axhline(y=math.log(2)*log_2_Z[PROBLEM_NAME], color='y', label='Ground Truth ln(Set Size)') 
    plt.xlabel('log_2(exact solution count)', fontsize=14)
    plt.ylabel('log_2(F2 lower bound) - log_2(exact solution count)', fontsize=14)
    plt.yscale('symlog')
    plt.title('Exact Solution Counts vs. F2 lower bounds', fontsize=20)
    # plt.legend(fontsize=8, loc=2, prop={'size': 6})    
#     plt.legend(fontsize=12, prop={'size': 12})    
    # Put a legend below current axis
    lgd = plt.legend(loc='upper center', bbox_to_anchor=(0.45, -0.12),
          fancybox=True, ncol=2, fontsize=12, prop={'size': 12})

    #make the font bigger
    matplotlib.rcParams.update({'font.size': 10})        

    plt.grid(True)

#     if not os.path.exists(results_directory + 'plots/'):
#         os.makedirs(results_directory + 'plots/')

    plot_name = 'SATcounts_vs_F2lowerBound.png'
    plt.savefig('./'+plot_name, bbox_extra_artists=(lgd,), bbox_inches='tight')
    matplotlib.pyplot.clf()
    # plt.show()    
    
    print("new training problems:")
    print(new_train_problems)
    print("new testing problems:")
    print(new_test_problems)   

