using PyPlot
using Statistics
using Optim
using ForwardDiff
using JLD
using LineSearches
println("hola")

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
    path_data="C:\\Users\\Tiffany\\Google Drive\\WORKING_MEMORY\\MODEL\\HMM_wm_mice-main\\synthetic\\"
    # path_data="C:\\Users\\Tiffany\\Google Drive\\WORKING_MEMORY\\MODEL\\HMM_wm_mice-main\\real\\"
    path_results="C:\\Users\\Tiffany\\Google Drive\\WORKING_MEMORY\\MODEL\\HMM_wm_mice-main\\results\\"
else
    path_functions="/home/tiffany/HMM_wm_mice-main/functions/"
    path_figures="/home/tiffany/HMM_wm_mice-main/figures/"
    path_data="/home/tiffany/HMM_wm_mice-main/synthetic/" # when using synthetic data
    path_data="/home/tiffany/HMM_wm_mice-main/real/" # when using synthetic data
    path_results="/home/tiffany/HMM_wm_mice-main/results/"
end
include(path_functions*"functions_wm_mice.jl")
include(path_functions*"function_simulations.jl")
include(path_functions*"functions_mle.jl")
include(path_functions*"behaviour_analysis.jl")



#being a chicken
# consts=["mu_k","c4","mu_b","tau_w","tau_l","lambda"]
# args=["c2","sigma","x0","beta_w","beta_l"]
# lower=[ 0.0, 0.05,-1.0,-10.0, -10.0]
# upper=[10.0, 10.0,  1.0,10.0,10.0]
# model="pit11t22c2SigmaX0Beta_wBeta_l"
# labels_param=["pi","t11","t22","c2","s","x0","w","l"]

#
# consts=["mu_k","c4","mu_b","tau_w","tau_l","lambda"]
# args=["c2","sigma","x0","beta_w","beta_l"]
# lower=[ 0.0, 0.05,-1.0,-10.0, -10.0]
# upper=[10.0, 10.0,  1.0,10.0,10.0]
# model="pit11t22c2SigmaX0Beta_wBeta_l_Muk1"
# labels_param=["pi","t11","t22","c2","s","x0","w","l"]


#Full model
# consts=["c4","mu_b","tau_w","tau_l","lambda"]
# args=["mu_k","c2","sigma","x0","beta_w","beta_l"]
# lower=[ 0.0,0.0, 0.05,-1.0,-10.0, -10.0]
# upper=[10.0,10.0, 10.0,  1.0,10.0,10.0]
# model="pit11t22Mukc2SigmaX0Beta_wBeta_l"
# labels_param=["pi","t11","t22","k","c2","s","x0","w","l"]


model="pit11t22Mukc2MubBeta_wBeta_lBeta_bias"
labels_param=["pi","t11","t22","Muk","c2","Mub","w","l","b"]
name = "data_set3_50Sessions"


Nparamfit=9

PossibleOutputs=[1,2]
Nconditions=30


data_set_num=1
fig,axes=subplots(1,5,figsize=(20/2.54,5/2.54))
fig2,axes2=subplots(1,5,figsize=(20/2.54,5/2.54))


for data_set_num in 1:5
    println(data_set_num)
    # data_filename=path_data*"data_set"*string(data_set_num)*"_50Sessions.jld"
    # data=load(data_filename)

    data_filename=path_data*name*".jld"
    data=load(data_filename)

    path_results="/Users/genis/wm_mice/results/synthetic_data/"*model*"Nsessions_dataset"*string(data_set_num)*"/"
    #path_results="/Users/genis/wm_mice/results/synthetic_data/"*model*"Ntrials"*string(Ntrials)*"dataset"*string(data_set_num)*"/"
    files=readdir(path_results)

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

        #try

            results=load(path_results*file)
            if length(results["ParamFit"])==Nparamfit && results["LL"]==results["LL"]
                #println("goodfile")
                llaux=NegativeLoglikelihood_Nsessions(results["PFit"],results["TFit"],data["choices"],results["PiFit"])
                push!(ll2,llaux)
                push!(LL_final_all,results["LL"])
                push!(ParamFit_all,results["ParamFit"])
                push!(PiFit_all,results["PiFit"])
                push!(files_valid,file)
                push!(LL_all,results["LL_all"])
                push!(Pstate,results["Pstate"])
            end

        #     end
        # catch
        #     println(file)
        # end

    end

    println("files_valid",length(files_valid))
    results=load(path_results*files_valid[1])
    args=results["args"]

    Nparam=length(ParamFit_all[1])
    for iparam in 1:Nparamfit
        aux=zeros(length(ParamFit_all))
        for icondition in 1:length(aux)
            #println(icondition)
            aux[icondition]=ParamFit_all[icondition][iparam]
        end
        #println(aux)
        axes[data_set_num].plot(iparam.*ones(length(aux)),aux,".k")
    end

    ll_min,indexMin=findmin(ll2) # retorna quin es el valor que dona el minim ll2
    #indexMin=2
    axes[data_set_num].plot(1:Nparam,ParamFit_all[indexMin],".b")


    param_original=data["param"]
    println("param_original",param_original)
    println("param_fit",ParamFit_all[indexMin]) #parametres que et donen el millor fit

    for i in 1:Nparamfit
        axes[data_set_num].plot([i],param_original[args[i]],"r|")
    end

    axes[data_set_num].set_xticks(1:Nparamfit)
    fontsize=8
    axes[data_set_num].set_xticklabels(labels_param,fontsize=fontsize)

    results_min=load(path_results*files_valid[indexMin])

    PROriginal_vector=data["POriginal"]
    PFit_vector=vcat_all(results_min["PFit"])

    axes2[data_set_num].plot(PROriginal_vector[:,2,2],PFit_vector[:,2,2],".",color="purple")
    axes2[data_set_num].plot(PROriginal_vector[:,1,2],PFit_vector[:,1,2],".",color="green")
    axes2[data_set_num].plot([0,1],[0,1],"--")


end


#
# data_filename=path_data*"data_set"*string(data_set_num)*".jld"
# data=load(data_filename)
#
#
# results=load(path_results*files_valid[indexMin])
# ECLL_original=ComputeECLL_full_aux(data["POriginal"],data["TOriginal"],data["choices"],data["PiInitialOriginal"])
#
# ECLL_fit=ComputeECLL_full_aux(results["PFit"],results["TFit"],data["choices"],results["PiFit"])
# results2=load(path_results*files_valid[2])
# ECLL_fit2=ComputeECLL_full_aux(results2["PFit"],results2["TFit"],data["choices"],results2["PiFit"])
#
#
# PFwdState,PBackState,Pstate,xi=ProbabilityState(data["POriginal"],data["TOriginal"],data["choices"],data["PiInitialOriginal"])
#
# PFwdStateFit,PBackStateFit,PstateFit,xiFit=ProbabilityState(results["PFit"],results["TFit"],data["choices"],results["PiFit"])



# XInitial=zeros(length(args))
# PiInitial=zeros(2)
# for icondition in 1:Nconditions
#     println(icondition)
#     for data_set_num in 1:5
#         println(data_set_num)
#         data_filename=path_data*"data_set"*string(data_set_num)*".jld"
#         data=load(data_filename)
#         #y=data["y"] #this might need to be updated when fitting the entire model
#         y=[data["y"][2],data["y"][3],data["y"][4],data["y"][5],data["y"][6]]
#         println(data["x"],data["y"],data["args"],data["consts"])
#         param_original=make_dict(data["args"],data["x"],data["consts"],data["y"])
#
#         for iparam in 1:length(lower)
#             XInitial[iparam]=lower[iparam]+ (upper[iparam]-lower[iparam])*rand()
#         end
#
#         pdwdw=rand()
#         pbiasbias=rand()
#         TInitial=[pdwdw 1-pdwdw ; 1-pbiasbias pbiasbias]
#         aux=rand()
#         PiInitial[1]=aux
#         PiInitial[2]=1-aux
#
#         #Fit using originial parameters as initial parameters
#         try
#             PFit,TFit,PiFit,Ll,ParamFit,xfit=fitBaumWelchAlgorithm(data["stim"],data["delays"],data["idelays"],
#             data["choices"],data["past_choices"],data["past_rewards"],args,XInitial,lower,upper,TInitial,PiInitial,PossibleOutputs,consts,y)
#             #PFit,TFit,PiFit,Ll,ParamFit,xfit=0,0,0,0,0,0
#             path_final=path_results*"dataset"*string(data_set_num)*"/"
#
#             if  isdir(path_final)==false
#                 mkdir(path_final)
#             end
#
#             ifile=0
#             filename=path_final*"initial_condition"*string(ifile)*".jld"
#             while isfile(filename)
#                 ifile=ifile+1
#                 filename=path_final*"initial_condition"*string(ifile)*".jld"
#             end
#             save(filename,"PFit",PFit,"TFit",TFit,"PiFit",PiFit,"LL",Ll,"ParamFit",ParamFit,"xfit",xfit)
#
#         catch
#             println("Something wrong during EM algorithm")
#         end
#
#
#     end
# end


#Fit random  parameters as initial parameters
# xini=[ 0.4,   0.8, 0.0,   4.0,   1.0]
# PDwDw2=0.5
# PBiasBias2=0.7
# Tx=[PDwDw2 1-PDwDw2; 1-PBiasBias2 PBiasBias2]
# PNew2,TNew2,PiNew2,Ll2,ParamFit2,xfit2=fitBaumWelchAlgorithm(stim,delays,idelays,choices,past_choices,past_rewards,args,xini,lower,upper,Tx,PiInitialOriginal,PossibleOutputs,consts,y)
# PiNew2,PFwdState2,PBackState2,Pstate2,xi=ProbabilityState(PNew,TNew,choices,PiNew)



# Nconditions=1
# Nstates=2
# XInitial=zeros(Nconditions,length(lower))
# TInitialAll=zeros(Nconditions,Nstates,Nstates)
# ConfideceIntervals=zeros(Nconditions,length(lower)+Nstates)
# Ll=zeros(Nconditions)
# ParamFit=zeros(Nconditions,length(lower)+Nstates)
#
# PiInitial=zeros(Nconditions,Nstates)
#
# PFit=zeros(Nconditions,Ntrials,Nstates,NPossibleOutputs)
#
# TFit=zeros(Nconditions,Nstates,Nstates)
# PiFit=zeros(Nconditions,Nstates)
#
# for icondition in 1:Nconditions
#     println("icondition:", icondition)
#     #random initial conditions
#     for iparam in 1:length(lower)
#         XInitial[icondition,iparam]=lower[iparam]+ (upper[iparam]-lower[iparam])*rand()
#     end
#
#     pdwdw=rand()
#     pbiasbias=rand()
#     TInitial=[pdwdw 1-pdwdw ; 1-pbiasbias pbiasbias]
#     TInitialAll[icondition,:,:]=TInitial
#     aux=rand()
#     PiInitial[icondition,1]=aux
#     PiInitial[icondition,2]=1-aux
#
#     #PNew,TNew,PiNew,Ll[icondition],ParamFit[icondition,:],xfit=fitBaumWelchAlgorithm(stim,delays,idelays,choices,past_choices,past_rewards,args,XInitial[icondition,:],lower,upper,TInitial,PiInitial[icondition,:],PossibleOutputs,consts,y)
#     PFit[icondition,:,:,:],TFit[icondition,:,:],PiFit[icondition,:],Ll[icondition],ParamFit[icondition,:],xfit=fitBaumWelchAlgorithm(stim,delays,idelays,choices,past_choices,past_rewards,args,XInitial[icondition,:],lower,upper,TInitial,PiInitial[icondition,:],PossibleOutputs,consts,y)
#
#     #ConfideceIntervals[icondition,:]=ComputeConfidenceIntervals(stim,delays,idelays,choices,past_choices,past_rewards,args,xfit,lower,upper,TNew,PiNew,PossibleOutputs,consts,y)
#
# end

#filename_save="/home/genis/wm_mice/synthetic_data/minimize_sigma_x0_betal_pdwdw_pbiasbias_Ntrials"*string(Ntrials)*".jld"

# LL=zeros(length(PDwVector),length(PBiasVector))
# for idw in 1:length(PDwVector)
#     for ibias in 1:length(PBiasVector)
#         LL[idw,ibias]=Compute_negative_LL_hmm_module(PDwVector[idw],PBiasVector[ibias],PrDw,PrBias,choices)
#         #LL[idw,ibias]=Compute_negative_LL_hmm_module(PDwDw,PBiasBias,PDwVector[idw],PBiasVector[ibias],choices)
#
#     end
# end
# figure()
# imshow(LL,origin="lower",extent=[PBiasVector[1],PBiasVector[end],PDwVector[1],PDwVector[end]],aspect="auto",cmap="hot")
# xlabel("PbiasBias")
# ylabel("PDwDw")
# plot([ PBiasBias],[PDwDw],"bo")
#
# #plot( [ PrBias],[PrDw],"bo")
#
#
# a=findall(x->x==minimum(LL),LL)
# plot([ PBiasVector[a[1][2]]],[PDwVector[a[1][1]]],"bs")
#
# colorbar()
# show()




# filename_save="/home/genis/wm_mice/synthetic_data/minimize_sigma_c2_Ntrials"*string(Ntrials)*".jld"
# filename_save="/home/genis/wm_mice/synthetic_data/minimize_sigma_c2_Ntrials"*string(Ntrials)*"_NDataSets"*string(NDataSets)*".jld"
# filename_save="/home/genis/wm_mice/synthetic_data/minimize_betaw_betal_only_history_bias_Ntrials"*string(Ntrials)*".jld"
# filename_save="/home/genis/wm_mice/synthetic_data/minimize_betaw_betal_only_history_bias_Ntrials"*string(Ntrials)*"_NDataSets"*string(NDataSets)*".jld"
# filename_save="/home/genis/wm_mice/synthetic_data/minimize_sigma_c2_wm_only_Ntrials"*string(Ntrials)*"_NDataSets"*string(NDataSets)*".jld"


# filename_save="/home/genis/wm_mice/synthetic_data/minimize_sigma_x0_betal_pdwdw_pbiasbias_Ntrials"*string(Ntrials)*".jld"
#
# save(filename_save,"x",x,"args",args,"y",y,"consts",consts,"XInitial",XInitial,"Ll",Ll,
# "PiInitial",PiInitial,"TInitialAll",TInitialAll,"ConfideceIntervals",ConfideceIntervals,
# "LlOriginal",LlOriginal)
