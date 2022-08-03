using PyPlot
using Statistics
using Optim
using ForwardDiff
using JLD
using DataStructures
using LineSearches
using Pandas
using JSON

pygui(true)
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

name= "E17"

model_list =
# ["pit11t22Mukc2x0Beta_wBeta_l",
# "pit11t22Mukc2x0MubBeta_wBeta_l",
# "pit11t22Mukc2MubBeta_wBeta_l",
# "pit11t22Mukc2Beta_wBeta_lBeta_bias_",
# "pit11t22Mukc2x0Beta_wBeta_lBeta_bias_",
# "pit11t22Mukc2x0MubBeta_wBeta_lBeta_bias_",
# "pit11t22Mukc2MubBeta_wBeta_lBeta_bias_"]

["pit11t22Mukc2x0MubBeta_wBeta_bias_",
"pit11t22Mukc2x0MubBeta_w"]

# label_list = [["WM init.","WM stay","HB stay","Muk","c2","x0","After c.","After inc."],
# ["WM init.","WM stay","HB stay","Muk","c2","x0","Mu_b","After c.","After inc."],
# ["WM init.","WM stay","HB stay","Muk","c2","Mu_b","After c.","After inc."],
# ["WM init.","WM stay","HB stay","Muk","c2","After c.","After inc.","b"],
# ["WM init.","WM stay","HB stay","Muk","c2","x0","After c.","After inc.","b"],
# ["WM init.","WM stay","HB stay","Muk","c2","x0","Mu_b","After c.","After inc.","b"],
# ["WM init.","WM stay","HB stay","Muk","c2","Mu_b","After c.","After inc.","b"]]

label_list = [["WM init.","WM stay","HB stay","Muk","c2","x0","Mu_b","After c.","b"],
["WM init.","WM stay","HB stay","Muk","c2","x0","Mu_b","After c."]]

variable_list = ["with" "without bias" ]
# variable_list = ["mub_bias" "bias"  "x0_bias" "mub_x0"  "mub" "full" "classic"]
# variable_list = ["mub_bias" "bias"  "mub_x0"  "mub" "full" "classic"]

function analysis_minimize(name, model, labels_param)
    Nparamfit=length(labels_param)
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

        try

            results=load(path_results_final*file)
            if length(results["ParamFit"])==Nparamfit && results["LL"]==results["LL"]
                # println("goodfile")
                llaux=NegativeLoglikelihood_Nsessions(results["PFit"],results["TFit"],data["choices"],results["PiFit"])
                push!(ll2,llaux)
                push!(LL_final_all,results["LL"])
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
    #indexMin=2
    PyPlot.suptitle(model)
    PyPlot.plot(4:Nparam,ParamFit_all[indexMin][4:Nparamfit],".b", markersize=8)
    # PyPlot.plot(4:Nparam,ParamFit_all[5][4:Nparamfit],".r", markersize=8)

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
    for iparam in 1:3
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
    # PyPlot.plot(1:3,ParamFit_all[5][1:3],".r", markersize=8)
    PyPlot.xticks(1:3, labels=labels_param[1:3])
    PyPlot.suptitle(model)
    save_path = path_figures*name*"_TT.png"
    PyPlot.savefig(save_path)

    # ----------------------------------

    results_min=load(path_results_final*files_valid[indexMin])

    # for i in 1:length(ll2)
    #     println(ll2[i])
    # end

    fraction = count(i->(round(i,digits=2)== round(ll_min, digits=2)), ll2)/length(ll2)*100
    println("The fraction of similiar minima is: "*string(fraction))

    # ll_min = ll2[5]
    # fraction = count(i->(round(i,digits=2)== round(ll_min, digits=2)), ll2)/length(ll2)*100
    # println("The fraction of second similiar minima is: "*string(fraction))

    return results_min
end

list_results =[]
for index in 1:length(model_list)
    println(model_list[index])
    results_min=analysis_minimize(name,model_list[index],label_list[index])
    push!(list_results,results_min)
end

name="E15_3"
function plot_comparision(name,list_results)
    data_filename=path_data*name*".jld"
    data=load(data_filename)

    jitter(n::Real, factor=0.2) = n + (0.5 - rand()) * factor

    PyPlot.figure()

    LL_final_all=[]
    ParamFit_all=[]
    PiFit_all=[]
    files_valid=[]
    LL_all=[]
    ll2=[]
    args=[]
    Pstate = []
    for value in list_results
        push!(LL_final_all,value["LL"])
        push!(ParamFit_all,value["ParamFit"])
        push!(PiFit_all,value["PiFit"])
        push!(files_valid,value["args"])
        push!(LL_all,value["LL_all"])
        push!(Pstate,value["Pstate"])
        push!(args,value["args"])
    end
    ll_min,indexMin=findmin(LL_final_all) # retorna quin es el valor que dona el minim ll2
    println(args[indexMin])
    print(LL_final_all)
    for index in 1:length(LL_final_all)
        println(label_list[index])
        println(2*length(args[index])+2*LL_final_all[index])
    end

    labels_param=["WM init.","WM stay","HB stay","Muk","c2","x0","Mub","After c.","After inc.","b"]
    PyPlot.xticks(1:10, labels=labels_param)
    for index in 1:length(list_results)
        if label_list[index] == ["WM init.","WM stay","HB stay","Muk","c2","x0","After c.","After inc."]
            x = [1,2,3,4,5,6,8,9]
            PyPlot.plot(jitter.(x),list_results[index]["ParamFit"],".k", markersize=6)
        end
        if label_list[index] == ["WM init.","WM stay","HB stay","Muk","c2","Mub","After c.","After inc."]
            x = [1,2,3,4,5,7,8,9]
            PyPlot.plot(jitter.(x),list_results[index]["ParamFit"],".k", markersize=6)
        end
        if list_results[index]["args"] == ["pi","t11","t22","mu_k","c2","x0","mu_b","beta_w","beta_l"]
            x = [1,2,3,4,5,6,7,8,9]
            PyPlot.plot(jitter.(x),list_results[index]["ParamFit"],".k", markersize=6)
        end
        if list_results[index]["args"] == ["pi","t11","t22","mu_k","c2","x0","beta_w","beta_l"]
            x = [1,2,3,4,5,6,8,9]
            PyPlot.plot(jitter.(x),list_results[index]["ParamFit"],".r", markersize=6)
        end
        if list_results[index]["args"] == ["pi","t11","t22","mu_k","c2","beta_w","beta_l","beta_bias"]
            x = [1,2,3,4,5,8,9,10]
            PyPlot.plot(jitter.(x),list_results[index]["ParamFit"],".y", markersize=6)
        end
        if list_results[index]["args"] == ["pi","t11","t22","mu_k","c2","x0","mu_b","beta_w","beta_l","beta_bias"]
            x = [1,2,3,4,5,6,7,8,9,10]
            PyPlot.plot(jitter.(x),list_results[index]["ParamFit"],".b", markersize=6)
        end
    end
end
plot_comparision(name,list_results)

function JLDtoJSON(name,results_min,variable)
    println("Passing from JLD to JSON")
    model="pit11t22Mukc2MubBeta_wBeta_lBeta_bias_"

    #File with all the data
    path_results_final=path_results*model*'_'*name*"\\"
    filename=path_results_final*"initial_condition00.jld"
    data=load(filename)

    # File with the final parameters of the model
    data2=results_min

    # println("Reassign the values to a dataframe")
    # Saving the data for the model
    dict=Dict(:args=> data2["args"], :ParamFit=> data2["ParamFit"])
    df=Pandas.DataFrame(dict)
    filename_save=path_data*name*"_fit_"*variable*".json"
    println(filename_save)
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
    filename_save=path_data*name*"_behavior_"*variable*".json"
    Pandas.to_json(df,filename_save)
end

function synthetic_data(name,variable,args,consts,y)
    println("Creating synthetic data")
    # model= "pit11t22c2c4SigmaX0Beta_wBeta_l"

    path_final=path_data*name*"_fit_"*variable*".json"
    println(path_final)
    data = JSON.parsefile(path_final)

    if length(data["args"]) == 9
        println(length(data["args"]))
        mydict=Dict(data["args"]["1"]=> data["ParamFit"]["1"],data["args"]["2"]=> data["ParamFit"]["2"],
        data["args"]["3"]=> data["ParamFit"]["3"],data["args"]["4"]=> data["ParamFit"]["4"],
        data["args"]["5"]=> data["ParamFit"]["5"],data["args"]["6"]=> data["ParamFit"]["6"],
        data["args"]["7"]=> data["ParamFit"]["7"],data["args"]["8"]=> data["ParamFit"]["8"],data["args"]["0"]=> data["ParamFit"]["0"])
        x=[ mydict[args[1]],  mydict[args[2]],  mydict[args[3]], mydict[args[4]], mydict[args[5]], mydict[args[6]], mydict[args[7]], mydict[args[8]], mydict[args[9]]]

    end

    if length(data["args"]) == 8
        mydict=Dict(data["args"]["1"]=> data["ParamFit"]["1"],data["args"]["2"]=> data["ParamFit"]["2"],
        data["args"]["3"]=> data["ParamFit"]["3"],data["args"]["4"]=> data["ParamFit"]["4"],
        data["args"]["5"]=> data["ParamFit"]["5"],data["args"]["6"]=> data["ParamFit"]["6"],
        data["args"]["7"]=> data["ParamFit"]["7"],data["args"]["0"]=> data["ParamFit"]["0"])
        x=[ mydict[args[1]],  mydict[args[2]],  mydict[args[3]], mydict[args[4]], mydict[args[5]], mydict[args[6]], mydict[args[7]], mydict[args[8]]]

    end

    if length(data["args"]) == 7
        mydict=Dict(data["args"]["1"]=> data["ParamFit"]["1"],data["args"]["2"]=> data["ParamFit"]["2"],
        data["args"]["3"]=> data["ParamFit"]["3"],data["args"]["4"]=> data["ParamFit"]["4"],
        data["args"]["5"]=> data["ParamFit"]["5"],data["args"]["6"]=> data["ParamFit"]["6"],
        data["args"]["0"]=> data["ParamFit"]["0"])
        x=[ mydict[args[1]],  mydict[args[2]],  mydict[args[3]], mydict[args[4]], mydict[args[5]], mydict[args[6]], mydict[args[7]]]

    end

    if length(data["args"]) == 10
        mydict=Dict(data["args"]["1"]=> data["ParamFit"]["1"],data["args"]["2"]=> data["ParamFit"]["2"],
        data["args"]["3"]=> data["ParamFit"]["3"],data["args"]["4"]=> data["ParamFit"]["4"],
        data["args"]["5"]=> data["ParamFit"]["5"],data["args"]["6"]=> data["ParamFit"]["6"],data["args"]["9"]=> data["ParamFit"]["9"],
        data["args"]["7"]=> data["ParamFit"]["7"],data["args"]["8"]=> data["ParamFit"]["8"],data["args"]["0"]=> data["ParamFit"]["0"])
        x=[ mydict[args[1]],  mydict[args[2]],  mydict[args[3]], mydict[args[4]], mydict[args[5]], mydict[args[6]], mydict[args[7]], mydict[args[8]], mydict[args[9]], mydict[args[10]]]

    end

    #Real data_set

    param=make_dict(args,x,consts,y)
    # delays=[0.0,100,200,300,500,800,1000,10000]

    delays=[0.0,1000,3000,10000]

    #Ntrials=300
    Nsessions=200
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

    # results=standard_analysis(choices2,stim2,state2,delays,idelays2,past_choices2)


    # POriginal_Nsession=ComputeEmissionProb_Nsessions(stim,delays,idelays,choices,past_choices,past_rewards,args,x,consts,y)
    # POriginal=ComputeEmissionProb(stim2,delays,idelays2,choices2,past_choices2,past_rewards2,args,x,consts,y)

    # PrOriginalDw=ProbRightDw(delays,args,x,consts,y)
    # T=[param["t11"] 1-param["t11"] ;
    #  1-param["t22"] param["t22"]]
    # InitialP=[1,0]
    # LlOriginal=NegativeLoglikelihood_Nsessions(POriginal_Nsession,T,choices,InitialP)

    # PyPlot.figure()
    # PyPlot.title("Synthetic data vs model")
    # PyPlot.plot(delays,PrOriginalDw[2,:],"r-")
    # PyPlot.plot(delays,results["PcDwDelay"],"o-")
    #
    # filename_save=path_data*name*"_synthetic_all.jld"
    # JLD.save(filename_save,"param",param,"consts",consts,"LlOriginal",LlOriginal,
    # "PiInitialOriginal",PiInitialOriginal,"TOriginal",T,"results",results,
    # "choices",choices,"state",state,"stim",stim,"past_choices",past_choices,
    # "past_rewards",past_rewards,"idelays",idelays,"POriginal",POriginal,
    # "POriginal_Nsession",POriginal_Nsession,"delays",delays, "day", "day")

    # ------------- Save in json as well

    # Saving the data for the model
    # dict=Dict(:param=> param, :consts=> consts, :LlOriginal=> LlOriginal, :PiInitialOriginal=> PiInitialOriginal, :TOriginal=> T,
    # :results=> results,:POriginal=> POriginal, :POriginal_Nsession=> POriginal_Nsession, :delays=> delays)

    # pass data as a json string (how it shall be displayed in a file)
    # stringdata = JSON.json(dict)

    # write the file with the stringdata variable information
    # filename_save=path_data*name*"_synthetic_params_.json"
    # open(filename_save , "w") do f
    #         write(f, stringdata)
    #      end

    # Saving data for the useful variables
    dict=Dict(:choices=> choices,:state=> state,:stim=> stim, :past_choices=> past_choices,:past_rewards=> past_rewards,
    :idelays=> idelays)

    df=Pandas.DataFrame(dict)
    filename_save=path_data*name*"_synthetic_behavior_"*variable*".json"
    Pandas.to_json(df,filename_save)
end

for index in 1:length(list_results)
    JLDtoJSON(name,list_results[index],variable_list[index])
    synthetic_data(name,variable_list[index], list_results[index]["args"],list_results[index]["consts"],list_results[index]["y"])
end
