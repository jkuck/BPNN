import subprocess


def launch_set_of_experiments():
    for use_MLP1 in [True, False]:
        use_MLP3 = not use_MLP1
        for SHARE_WEIGHTS in [True, False]:
            for subtract_prv_messages in [True, False]:
                for bethe_mlp in ['shifted','standard','linear','none']:
                    subprocess.run(["sbatch", "launch_bpnnSpinGlass_sbatch.sh", "%s" % use_MLP1, "%s" % use_MLP3, "%s" % SHARE_WEIGHTS, "%s" % subtract_prv_messages, bethe_mlp])

def launch_batch_size_experiments():
    max_factor_state_dimensions = 3
    msg_passing_iters = 2
    belief_repeats = 1
    random_seed = 1
    problem_category_train = 'blasted_problems'
    train_val_split = 'random_shuffle'
    for batch_size in [1,2,5,10,20,50]:
        subprocess.run(["sbatch", "launch_bpSAT_sbatch.sh", "%d" % max_factor_state_dimensions, "%d" % msg_passing_iters, "%d" % belief_repeats, "%d" % random_seed, problem_category_train, train_val_split, "%d" % batch_size])
        
        
# launch_batch_size_experiments()        
launch_set_of_experiments()