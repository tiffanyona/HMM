using PyPlot
using Statistics
using Optim
using ForwardDiff
using JLD
using DataStructures
using LineSearches
using Pandas
using JSON

auxpath=pwd()
# if occursin("Users",auxpath)
#     path_functions="/Users/genis/wm_mice/HMM_wm_mice/functions/"
#     path_figures="/Users/genis/wm_mice/figures/"
#     path_data="/Users/genis/wm_mice/synthetic_data/"
#     path_results="/Users/genis/wm_mice/results/synthetic_data/"
#
# else
#     path_functions="/home/genis/wm_mice/scripts/functions/"
#     path_figures="/home/genis/wm_mice/figures/"
#     path_data="/home/genis/wm_mice/synthetic_data/"
#     path_results="/home/genis/wm_mice/results/synthetic_data/"
#
# end

if occursin("Users",auxpath)
    path_functions="C:\\Users\\Tiffany\\Google Drive\\WORKING_MEMORY\\MODEL\\HMM_wm_mice-main\\functions\\"
    path_figures="C:\\Users\\Tiffany\\Google Drive\\WORKING_MEMORY\\MODEL\\HMM_wm_mice-main\\figures\\"
    # path_data="C:\\Users\\Tiffany\\Google Drive\\WORKING_MEMORY\\MODEL\\HMM_wm_mice-main\\synthetic\\"
    path_data="C:\\Users\\Tiffany\\Google Drive\\WORKING_MEMORY\\MODEL\\HMM_wm_mice-main\\real\\"
    path_results="C:\\Users\\Tiffany\\Google Drive\\WORKING_MEMORY\\MODEL\\HMM_wm_mice-main\\results\\"
else
    path_functions="/home/tiffany/HMM_wm_mice-main/functions/"
    path_figures="/home/tiffany/HMM_wm_mice-main/figures/"
    # path_data="/home/tiffany/HMM_wm_mice-main/synthetic/" # when using synthetic data
    path_data="/home/tiffany/HMM_wm_mice-main/real/" # when using synthetic data
    path_results="/home/tiffany/HMM_wm_mice-main/results/"
end

include(path_functions*"functions_wm_mice.jl")
include(path_functions*"function_simulations.jl")
include(path_functions*"functions_mle.jl")
include(path_functions*"behaviour_analysis.jl")

function analysis_minimize(name,extra)
    if extra == "classic"
        model="pit11t22Mukc2MubBeta_wBeta_lBeta_bias_"
        labels_param=["WM init.","WM stay","HB stay","Muk","c2","Mub","After c.","After inc.","b"]
        Nparamfit=9

    elseif extra == "withoutx0beta_l"
        model="pit11t22Mukc2MubBeta_wBeta_bias_"
        labels_param=["WM init.","WM stay","HB stay","Muk","c2","Mub","After c.","b"]
        Nparamfit=8

    elseif extra == "all"
        model="pit11t22Mukc2x0MubBeta_wBeta_bias_"
        labels_param=["WM init.","WM stay","HB stay","Muk","c2","x0","Mub","After c.","b"]
        Nparamfit=9

    elseif extra == "withoutbeta"
        model="pit11t22Mukc2x0MubBeta_w"
        labels_param=["WM init.","WM stay","HB stay","Muk","c2","x0","Mub","After c."]
        Nparamfit=8
    end

    PossibleOutputs=[1,2]

    data_filename=path_data*name*".jld"
    data=load(data_filename)

    path_results_final=path_results*model*'_'*name*"\\"
    files=readdir(path_results_final)

    Ncondition=length(files)
    LL_final_all=[]
    ParamFit_all=[]
    PiFit_all=[]
    files_valid=[]
    LL_all=[]
    ll2=[]
    Files_valid=[]
    Pstate = []
    for file in files
        #
        try
            results=load(path_results_final*file)
            if length(results["ParamFit"])==Nparamfit && results["LL"]==results["LL"]
                # println("goodfile")
                # llaux=NegativeLoglikelihood_Nsessions(results["PFit"],results["TFit"],data["choices"],results["PiFit"])
                # push!(ll2,llaux)
                push!(ll2,results["LL"])
                push!(ParamFit_all,results["ParamFit"])
                push!(PiFit_all,results["PiFit"])
                push!(files_valid,file)
                push!(LL_all,results["LL_all"])
                push!(Pstate,results["Pstate"])
            end
        catch
            println(file)
        end
    end

    println("files_valid: ",length(files_valid))
    results=load(path_results_final*files_valid[1])
    args=results["args"]

    jitter(n::Real, factor=0.5) = n + (0.5 - rand()) * factor

    PyPlot.figure()
    Nparam=length(ParamFit_all[1])
    for iparam in 4:Nparamfit
        aux=zeros(length(ParamFit_all))
        for icondition in 1:length(aux)
            #println(icondition)
            aux[icondition]=ParamFit_all[icondition][iparam]
        end
        #println(aux)
        x = iparam.*ones(length(aux))
        PyPlot.plot(jitter.(x),aux,".k", markersize=4)
    end

    ll_min,indexMin=findmin(ll2) # retorna quin es el valor que dona el minim ll2
    println(ll_min)
    #indexMin=2
    PyPlot.plot(4:Nparam,ParamFit_all[indexMin][4:Nparamfit],".b", markersize=8)
    PyPlot.plot(4:Nparam,ParamFit_all[1][4:Nparamfit],".r", markersize=8)

    # param_original=data["param"]
    # println("param_original",param_original)

    println("good fit file: ",files_valid[indexMin])
    println("param_fit",ParamFit_all[indexMin]) #parametres que et donen el millor fit

    # This is for when we have the original values of the fit
    # for i in 1:Nparamfit
    #     PyPlot.plot([i],param_original[args[i]],"r|")
    # end

    PyPlot.xticks(4:Nparamfit, labels=labels_param[4:Nparamfit])
    fontsize=8
    save_path = path_figures*name*".png"
    PyPlot.suptitle(name)
    PyPlot.savefig(save_path)

    # Plot the transition matrix ------------------
    jitter(n::Real, factor=0.5) = n + (0.5 - rand()) * factor

    PyPlot.figure()
    Nparam=length(ParamFit_all[1])
    for iparam in 1:Nparamfit-6
        aux=zeros(length(ParamFit_all))
        for icondition in 1:length(aux)
            #println(icondition)
            aux[icondition]=ParamFit_all[icondition][iparam]
        end
        #println(aux)
        x = iparam.*ones(length(aux))
        PyPlot.plot(jitter.(x),aux,".k", markersize=4)
    end

    PyPlot.plot(1:3,ParamFit_all[indexMin][1:3],".b", markersize=8)
    # PyPlot.plot(1:3,ParamFit_all[1][1:3],".r", markersize=8)
    PyPlot.xticks(1:3, labels=labels_param[1:3])
    PyPlot.suptitle(name)
    save_path = path_figures*name*"_TT.png"
    PyPlot.savefig(save_path)

    # ----------------------------------

    results_min=files_valid[indexMin]
    # results_min=files_valid[1]
    # ll_min=ll2[1]

    for i in 1:length(ll2)
        println(ll2[i])
    end

    fraction = count(i->(round(i,digits=2)== round(ll_min, digits=2)), ll2)/length(ll2)*100
    println("The fraction of similiar minima is: "*string(fraction))

    # ll_min = ll2[5]
    # fraction = count(i->(round(i,digits=2)== round(ll_min, digits=2)), ll2)/length(ll2)*100
    # println("The fraction of second similiar minima is: "*string(fraction))

    return results_min
end

function JLDtoJSON(name,results_min,extra)
    println("Passing from JLD to JSON")
    if extra=="withoutx0"
        model="pit11t22Mukc2MubBeta_wBeta_bias_"
    elseif extra=="all"
        model="pit11t22Mukc2x0MubBeta_wBeta_bias_"
    elseif extra=="withoutbias"
        model="pit11t22Mukc2x0MubBeta_w"
    else
        model= "pit11t22c2c4SigmaX0Beta_wBeta_l"
    end

    println(model)
    #File with all the data
    path_results_final=path_results*model*'_'*name*"\\"
    filename=path_results_final*"initial_condition00.jld"
    data=load(filename)

    # File with the final parameters of the model
    println(results_min)
    filename=path_results_final*results_min
    data2=load(filename)

    # println("Reassign the values to a dataframe")
    # Saving the data for the model
    dict=Dict(:args=> data2["args"], :ParamFit=> data2["ParamFit"])
    df=Pandas.DataFrame(dict)
    filename_save=path_data*name*"_fit_"*extra*".json"
    Pandas.to_json(df,filename_save)

    # # pass data as a json string (how it shall be displayed in a file)
    # stringdata = JSON.json(dict)
    #
    # # write the file with the stringdata variable information
    # filename_save=path_real_data*name*"_fit.json"
    # open(filename_save , "w") do f
    #         write(f, stringdata)
    #      end

    # Saving data for the useful variables
    dict=Dict(:Pstate=> data2["Pstate"], :stim => data["stim"],:idelays => data["idelays"], :choices=>
    data["choices"],:past_choices=>data["past_choices"],:past_rewards=> data["past_rewards"],:day=> data["day"])

    df=Pandas.DataFrame(dict)
    filename_save=path_data*name*"_behavior_"*extra*".json"
    Pandas.to_json(df,filename_save)
end

function synthetic_data(name,tau,extra)
    println("Creating synthetic data")
    # model= "pit11t22c2c4SigmaX0Beta_wBeta_l"

    path_final=path_data*name*"_fit_"*extra*".json"
    println(path_final)
    data = JSON.parsefile(path_final)

    if extra=="all"
        mydict=Dict(data["args"]["1"]=> data["ParamFit"]["1"],data["args"]["2"]=> data["ParamFit"]["2"],
        data["args"]["3"]=> data["ParamFit"]["3"],data["args"]["4"]=> data["ParamFit"]["4"],
        data["args"]["5"]=> data["ParamFit"]["5"],data["args"]["6"]=> data["ParamFit"]["6"],
        data["args"]["7"]=> data["ParamFit"]["7"],data["args"]["8"]=> data["ParamFit"]["8"],data["args"]["0"]=> data["ParamFit"]["0"])

    else
        mydict=Dict(data["args"]["1"]=> data["ParamFit"]["1"],data["args"]["2"]=> data["ParamFit"]["2"],
        data["args"]["3"]=> data["ParamFit"]["3"],data["args"]["4"]=> data["ParamFit"]["4"],
        data["args"]["5"]=> data["ParamFit"]["5"],data["args"]["6"]=> data["ParamFit"]["6"],
        data["args"]["7"]=> data["ParamFit"]["7"],data["args"]["0"]=> data["ParamFit"]["0"])
    end

    PDwDw=mydict["t11"]
    PBiasBias=mydict["t22"]

    if extra == "classic"
        # Real data_set
        consts=["sigma","c4","x0","tau_w","tau_l","lambda"]
        y = [1.00, 1.00, 0, tau, tau, 0]
        args=["pi","t11","t22","mu_k","c2","mu_b","beta_w","beta_l","beta_bias"]
        x=[ mydict["pi"], PDwDw, PBiasBias, mydict["mu_k"], mydict["c2"], mydict["mu_b"], mydict["beta_w"], mydict["beta_l"], mydict["beta_bias"]]

    elseif extra == "withoutx0"
        consts=["sigma","c4","x0","tau_w","tau_l","lambda","beta_l"]
        y = [1.00, 1.00, 0, tau, tau, 0, 0]
        args=["pi","t11","t22","mu_k","c2","mu_b","beta_w","beta_bias"]
        x=[ mydict["pi"], PDwDw, PBiasBias, mydict["mu_k"], mydict["c2"], mydict["mu_b"], mydict["beta_w"], mydict["beta_bias"]]

    elseif extra == "all"
        consts=["sigma","c4","tau_w","tau_l","lambda","beta_l"]
        y = [1.00, 1.00,  tau, tau, 0, 0]
        args=["pi","t11","t22","mu_k","c2","x0","mu_b","beta_w","beta_bias"]
        x=[ mydict["pi"], PDwDw, PBiasBias, mydict["mu_k"], mydict["c2"],  mydict["x0"], mydict["mu_b"], mydict["beta_w"], mydict["beta_bias"]]

    elseif extra == "withoutbias"
        consts=["sigma","c4","tau_w","tau_l","lambda","beta_l","beta_bias"]
        y = [1.00, 1.00,  tau, tau, 0, 0, 0]
        args=["pi","t11","t22","mu_k","c2","x0","mu_b","beta_w"]
        x=[ mydict["pi"], PDwDw, PBiasBias, mydict["mu_k"], mydict["c2"],  mydict["x0"], mydict["mu_b"], mydict["beta_w"]]
    end
    param=make_dict(args,x,consts,y)
    # delays=[0.0,100,200,300,500,800,1000,10000]

    delays=[0.0,1000,3000,10000]

    #Ntrials=300
    Nsessions=100
    #choices,state,stim,past_choices,past_rewards,idelays=create_data(Ntrials,delays,args,x)

    PiInitialOriginal=[1 0]
    initial_state=1
    PossibleOutputs=[1,2]

    choices,state,stim,past_choices,past_rewards,idelays=create_data_Nsessions(Nsessions,delays,param,initial_state)

    global choices2=choices[1]
    global state2=state[1]
    global stim2=stim[1]
    global past_choices2=past_choices[1]
    global past_rewards2=past_rewards[1]
    global idelays2=idelays[1]
    for isession in 2:Nsessions
        global choices2=vcat(choices2,choices[isession])
        global state2=vcat(state2,state[isession])
        global stim2=vcat(stim2,stim[isession])
        global idelays2=vcat(idelays2,idelays[isession])
        global past_choices2=vcat(past_choices2,past_choices[isession])
        global past_rewards2=vcat(past_rewards2,past_rewards[isession])
    end

    results=standard_analysis(choices2,stim2,state2,delays,idelays2,past_choices2)
    POriginal_Nsession=ComputeEmissionProb_Nsessions(stim,delays,idelays,choices,past_choices,past_rewards,args,x,consts,y)
    POriginal=ComputeEmissionProb(stim2,delays,idelays2,choices2,past_choices2,past_rewards2,args,x,consts,y)

    # PrOriginalDw=ProbRightDw(delays,args,x,consts,y)
    T=[param["t11"] 1-param["t11"] ;
     1-param["t22"] param["t22"]]
    InitialP=[1,0]
    LlOriginal=NegativeLoglikelihood_Nsessions(POriginal_Nsession,T,choices,InitialP)

    # PyPlot.figure()
    # PyPlot.title("Synthetic data vs model")
    # PyPlot.plot(delays,PrOriginalDw[2,:],"r-")
    # PyPlot.plot(delays,results["PcDwDelay"],"o-")

    filename_save=path_data*name*"_synthetic_"*extra*".jld"
    JLD.save(filename_save,"param",param,"consts",consts,"LlOriginal",LlOriginal,
    "PiInitialOriginal",PiInitialOriginal,"TOriginal",T,"results",results,
    "choices",choices,"state",state,"stim",stim,"past_choices",past_choices,
    "past_rewards",past_rewards,"idelays",idelays,"POriginal",POriginal,
    "POriginal_Nsession",POriginal_Nsession,"delays",delays, "day", "day")

    # ------------- Save in json as well

    # Saving the data for the model
    dict=Dict(:param=> param, :consts=> consts, :LlOriginal=> LlOriginal, :PiInitialOriginal=> PiInitialOriginal, :TOriginal=> T,
    :results=> results,:POriginal=> POriginal, :POriginal_Nsession=> POriginal_Nsession, :delays=> delays)

    # pass data as a json string (how it shall be displayed in a file)
    stringdata = JSON.json(dict)

    # write the file with the stringdata variable information
    # filename_save=path_data*name*"_synthetic_params_.json"
    # open(filename_save , "w") do f
    #         write(f, stringdata)
    #      end

    # Saving data for the useful variables
    dict=Dict(:choices=> choices,:state=> state,:stim=> stim, :past_choices=> past_choices,:past_rewards=> past_rewards,
    :idelays=> idelays)

    df=Pandas.DataFrame(dict)
    filename_save=path_data*name*"_synthetic_behavior_"*extra*".json"
    Pandas.to_json(df,filename_save)
end

pygui(false)
# tau_list=[2.28, 0.6, 1.62, 0.94, 3.5, 1.55, 3.9, 2.1, 3.3, 2.3, 2.6, 1, 2, 2.6, 1.1, 0.8, 1, 1.3, 3.1, 1.8]
# list_name=["E03", "E04", "E05", "E06", "E07", "E08", "E09", "E10", "E11", "E12", "E13", "E14", "E15", "E16", "E17", "E18", "E19", "E20", "E21", "E22"]

# tau_list=[ 2.28 0.6 1.54 2.2 0.94 2.8 4.6 1.55 3.9 2.1 3.3 1.2 4.4 2.6 1 2 2.8 3 1.8 1 1.4 0.8 1 1.2 1.8 3.1 1.8]
# list_name=["E03", "E04", "E05_3", "E05_10", "E06", "E07_3", "E07_10", "E08" ,"E09", "E10", "E11", "E12_3", "E12_10", "E13" ,"E14", "E15_3", "E15_10", "E16_3", "E16_10", "E17_3", "E17_10", "E18" ,"E19", "E20_3", "E20_10", "E21", "E22"]

# tau_list=[2.3 2.96 1.78 2.78 5.8 1.48 4.2 2.38 1.22 2.17 0.05 1.33 1.92 1.62 1.78 1.44 1.93 2.5 1.56 2.4 2.69 1.83 3 2.4 1.4]
# list_name=["N02" "N03" "N04" "N05_3" "N05_10" "N07_3" "N07_10" "N08" "N09" "N11_3" "N11_10" "N13" "N19" "N20" "N21" "N22" "N24_3" "N24_10" "N25_3" "N25_10" "N26" "N27_3" "N27_10" "N28_3" "N28_10"]

tau_list=[ 1.06 3 2.88 1.1 3 1.34 2.1 3.98 2.96 1.75 1.06 2.01 0.72 2.58 2.4 2.39 3.05]
list_name=["C10b" "C12" "C13" "C15" "C18" "C19" "C20" "C22" "C28_3" "C28_10"  "C32" "C34" "C36" "C37_3" "C37_10" "C38" "C39" ]

extra = "all"
global j= 0
for name in list_name
    global j+=1
    println(name)
    println(tau_list[j])
    results_min = analysis_minimize(name,extra)
    JLDtoJSON(name,results_min,extra)
    synthetic_data(name,tau_list[j],extra)
end

# name = "E19"
# tau =  1
# results_min = analysis_minimize(name)
# JLDtoJSON(name,results_min)
# synthetic_data(name,tau)
