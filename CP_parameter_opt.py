import sys
import warnings
warnings.filterwarnings(
    "ignore",
    message="The value of the smallest subnormal for <class 'numpy.float64'> type is zero."
)
sys.path.append("/Users/sisimiao/CLionProjects/Chase-Pyndiah/cmake-build-debug")
sys.path.append("build")
# sys.path.append("cmake-build-release-remote-host")
# sys.path.append("/Users/sisimiao/CLionProjects/GPC_cleanup/cmake-build-debug")



import os

from optuna import Trial

print(os.getcwd())

import numpy as np
print(sys.path)
import optuna
optuna.logging.set_verbosity(optuna.logging.INFO)

# "/usr/stud/miao/anaconda3/bin:/usr/stud/miao/anaconda3/condabin:/usr/stud/miao/.local/bin:/usr/stud/miao/bin:/usr/local/bin:/usr/bin:/bin"
import GPC_simulator_python
# print(dir(GPC_simulator_python))
# print(GPC_simulator_python.__file__)
# You can use Matplotlib instead of Plotly for visualization by simply replacing `optuna.visualization` with
# `optuna.visualization.matplotlib` in the following examples.
# print(pc_simulator_python.simulate_PC(0.1))
from optuna.visualization import plot_optimization_history
def write_best_to_file(filename, n, t, even, extend, shorten, opt_EbNo, state):
    best_params = state["best_params_since_reset"]
    if (best_params is None):
        print("no best params to write for n =", n, "t =", t, "even =", even, "extend =", extend, "shorten =", shorten)
        return
    print(best_params)
    with open(filename, "a") as f:   # "a" = append
        f.write(f"n = {n}\n")
        f.write(f"t = {t}\n")
        f.write(f"even = {even}\n")
        f.write(f"extend = {extend}\n")
        f.write(f"shorten = {shorten}\n")
        f.write("best_params_since_reset:\n")
        for k, v in best_params.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"Optimized at Eb/No = {opt_EbNo['value']:.3f} dB with BER = {state['best_since_reset']}\n")
        f.write("-" * 60 + "\n")

def adapt_EbNo_leq(study, trial):
    if trial.state != optuna.trial.TrialState.COMPLETE or trial.value is None:
        return
    print(f"[{trial.datetime_start}] Trial {trial.number} finished with value: {trial.value} "
          f"and parameters: {trial.params}. Best is trial {study.best_trial.number} with value: {study.best_value}.\n")
    v = trial.value

    # update best since reset
    if v <= state["best_since_reset"]:
        state["best_since_reset"] = v
        state["best_trial_since_reset"] = trial.number
        state["best_params_since_reset"] = dict(trial.params)


    print(
        f"Trial {trial.number} value={v} | "
        f"best_since_reset={state['best_since_reset']} (trial {state['best_trial_since_reset']}) | "
        f"global_best={study.best_value} (trial {study.best_trial.number})", flush=True
    )

    # condition to change Eb/No
    if state["best_since_reset"] < 1e-7:
        opt_EbNo["value"] = max(opt_EbNo["value"] - 0.05, 0.5)

        # reset local best
        state["best_since_reset"] = float("inf")
        state["best_trial_since_reset"] = None
        state["best_params_since_reset"] = None
        state["last_EbNo"] = opt_EbNo["value"]

        print(f"Changed Eb/No to {opt_EbNo['value']:.3f} → reset local best", flush=True)

def quantize(x, low, step):
    return round((x - low) / step) * step + low

def suggest_increasing(trial, name_prefix, n, low, high, step):
    values = []
    prev = low
    for i in range(n):
        prev_q = quantize(prev, low, step)
        prev_q = round(prev_q, 10)

        low_b = min(prev_q, high)
        high_b = max(prev_q, high)

        v = trial.suggest_float(
            f"{name_prefix}{i+1}",
            low_b,
            high_b,
            step=step
        )
        values.append(v)
        prev = v
    return values



def suggest_decreasing(trial, name_prefix, n, low, high, step):
    values = []
    prev = high
    for i in range(n):
        prev_q = quantize(prev, low, step)
        prev_q = round(prev_q, 10)

        low_b = min(low, prev_q)
        high_b = max(low, prev_q)

        v = trial.suggest_float(
            f"{name_prefix}{i+1}",
            low_b,
            high_b,
            step=step
        )
        values.append(v)
        prev = v
    return values

def objective(trial, opt_EbNo, n, t, even, extend, shorten, codeType, decodeMode, decIter, p_chase,useChaseII,  useTop2 , useNN4MD):
    EbNo_dB= opt_EbNo["value"]
    # alpha
    # alpha = [
    #     trial.suggest_float(f"a{i+1}",
    #                         0.1,
    #                         1.0,
    #                         step=0.02
    #                         )
    #     for i in range(2*decIter)
    # ]
    #
    # # beta
    # beta = [
    #     trial.suggest_float(f"b{i+1}", 0.1, 1.0, step=0.02)
    #     for i in range(2*decIter)
    # ]
    #
    # # MDscale
    # MDscale = [
    #     trial.suggest_float(f"t{i+1}", 0.05, 1.0, step=0.1)
    #     for i in range(2*decIter)
    # ]
    # # MDscale = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    #

    # top2Threshold= [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    # MDscale =  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    alpha = suggest_increasing(trial, "a", 2*decIter, 0.1, 0.7, 0.005)
    beta  = suggest_increasing(trial, "b", 2*decIter, 0.4, 1.0, 0.005)
    top2Threshold= [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    MDscale =  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    NotMDscale = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    if (decodeMode>1):
        if (useTop2):
            top2Threshold= suggest_decreasing(trial, "t", 2*decIter, 0.0, 8.0, 0.1)
        else:
            if (not useNN4MD):
                NotMDscale = suggest_increasing(trial, "ns", 2*decIter, 0.0, 0.6, 0.005)
        MDscale  = suggest_increasing(trial, "scale", 2*decIter, 0.0, 0.5, 0.01)




    return GPC_simulator_python.simOnePointSISO(
        EbNo_dB, n, t, even, extend, shorten,
        codeType, decodeMode, decIter, p_chase, useChaseII, useTop2 , useNN4MD,
        alpha, beta, MDscale, top2Threshold, NotMDscale
    )
def print_only_new_best_trial(study, trial):
    if trial.value == study.best_value:
        print(f"[{trial.datetime_start}] Trial {trial.number} finished with value: {trial.value} "
              f"and parameters: {trial.params}. Best is trial {study.best_trial.number} with value: {study.best_value}.\n")

def print_only_new_best_trial_current(study, trial):
    if trial.state != optuna.trial.TrialState.COMPLETE or trial.value is None:
        return

    # print only when this trial is best SINCE LAST EbNo RESET
    if trial.number == state["best_trial_since_reset"]:
        print(
            f"[{trial.datetime_start}] Trial {trial.number} finished with value: {trial.value}\n"
            f"  Best since EbNo reset (EbNo={opt_EbNo['value']:.3f}):\n"
            f"  value = {state['best_since_reset']}\n"
            f"  params = {state['best_params_since_reset']}\n", flush=True
        )
###################################################################################################
data = [
    # (127, 2, True, 3.5),
    (255, 2, True, float(sys.argv[6])),
    # (511, 2, True, 4.7),
    # (255, 3, True, 3.6),
    # (511, 3, True, 4.4),
    # (1023, 3, True, 5.2)
]

for n, t, extend, EbNo in data:
    print(n, t, extend, EbNo)
    opt_EbNo = {"value": EbNo}
    # n = 255
    # t = 3
    even = False
    # extend = True
    shorten = 0
    maxSoftLevel = 15
    window_len = 8
    decodeMode = 2
    codeType = 0
    useTop2 = False
    useNN4MD = True
    useChaseII = True

    if (decodeMode==1):
        print("original CP decoder")
    if (decodeMode==2):
        print("optimized CP decoder")
        if (useTop2):
            print("use Top2 miscorrection detection")
        else:
            if (useNN4MD):
                print("use NN miscorrection detection")
            else:
                print("use NN message scaling")

    decIter = int(sys.argv[1])
    p_chase = int(sys.argv[2])
    trail_run = int(sys.argv[3])

    # for T in np.arange(0, 0.14 + 0.001, 0.02):
    # opt_EbNo = 3.6
    print("Simulation parameters:", flush=True)
    print(f"n = {n}", flush=True)
    print(f"t = {t}", flush=True)
    print(f"even = {even}", flush=True)
    print(f"extend = {extend}", flush=True)
    print(f"shorten = {shorten}", flush=True)
    print(f"trail_run = {trail_run}", flush=True)
    print(f"optimizing at EbNo = {opt_EbNo} dB", flush=True)


    state = {
        "best_since_reset": float("inf"),
        "best_trial_since_reset": None,
        "best_params_since_reset": None,
        "last_EbNo": opt_EbNo["value"],
    }


    study = optuna.create_study()
    warnings.filterwarnings("ignore")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study.optimize(
        lambda trial: objective(trial, opt_EbNo, n, t, even, extend, shorten,
                                0, decodeMode, decIter, p_chase, useChaseII, useTop2, useNN4MD),
        n_trials=trail_run,
        callbacks=[print_only_new_best_trial_current, adapt_EbNo_leq]
    )

    best_params = state["best_params_since_reset"]
    print(f"n = {n}", flush=True)
    print(f"t = {t}", flush=True)
    print(f"even = {even}", flush=True)
    print(f"extend = {extend}", flush=True)
    print(f"shorten = {shorten}", flush=True)
    print(best_params)
    print(f"Optimized at Eb/No = {opt_EbNo['value']:.3f} dB with BER = {state['best_since_reset']}\n", flush=True)


    EbNo_start = float(sys.argv[4])
    EbNo_end   = float(sys.argv[5])

    alpha = [best_params[f"a{i}"] for i in range(1, 2*decIter+1)]
    beta  = [best_params[f"b{i}"] for i in range(1, 2*decIter+1)]

    top2Threshold= [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    MDscale = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    NotMDscale = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    if (decodeMode!=1):
        if useTop2:
            top2Threshold= [best_params[f"t{i}"] for i in range(1, 2*decIter+1)]
        else:
            if (not useNN4MD):
                NotMDscale =[best_params[f"ns{i}"] for i in range(1, 2*decIter+1)]
        MDscale = [best_params[f"scale{i}"] for i in range(1, 2*decIter+1)]

    GPC_simulator_python.simBERcurveSISO(
        EbNo_start,  EbNo_end, n, t, even, extend, shorten,
        codeType, decodeMode, decIter, p_chase, useChaseII, useTop2, useNN4MD,
        alpha, beta, MDscale, top2Threshold,NotMDscale
    )

