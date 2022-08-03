#using PyPlot
using Statistics
using Optim
using ForwardDiff
using JLD
using LineSearches
println("hola")

auxpath=pwd()
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
#include(path_functions*"behaviour_analysis.jl")

#being a chicken
# consts=["mu_k","c4","mu_b","tau_w","tau_l","lambda"]
# args=["pi","t11","t22","c2","sigma","x0","beta_w","beta_l"]
# lower=[0.0,0.0,0.0,0.0, 0.05,-1.0,-10.0, -10.0]
# upper=[1.0,1.0,1.0,10.0, 10.0,  1.0,10.0,10.0]
# model="pit11t22c2SigmaX0Beta_wBeta_l"


#being a chicken
# consts=["mu_k","c4","mu_b","tau_w","tau_l","lambda"]
# args=["pi","t11","t22","c2","sigma","x0","beta_w","beta_l"]
# lower=[0.0,0.0,0.0,0.0, 0.05,-1.0,-10.0, -10.0]
# upper=[1.0,1.0,1.0,10.0, 10.0,  1.0,10.0,10.0]
# model="pit11t22c2SigmaX0Beta_wBeta_l_Muk1"


# consts=["mu_k","c4","mu_b","tau_w","tau_l","lambda"]
# args=["pi","t11","t22","c2","sigma","x0","beta_w","beta_l"]
# lower=[0.0,0.0,0.0,0.0, 0.05,-1.0,-10.0, -10.0]
# upper=[1.0,1.0,1.0,10.0, 10.0,  1.0,10.0,10.0]
# model="pit11t22c2SigmaX0Beta_wBeta_l"

#being a chicken 2
#
# consts=["sigma","c4","x0","tau_w","tau_l","lambda"]
# args=["pi","t11","t22","mu_k","c2","mu_b","beta_w","beta_l","beta_bias"]
# lower=[0.0,0.0,   0.0,  0.0,  0.0, -10.0, -10.0,   -10.0,    -10.0]
# upper=[1.0,1.0,   1.0,  10.0,10.0, 10.0,  10.0,    10.0,     10.0]
# model="pit11t22Mukc2MubBeta_wBeta_lBeta_bias"
# name_of_file = "data_set"*string(data_set_num)*"_50Sessions.jld"

#Full model
# consts=["c4","mu_b","tau_w","tau_l","lambda"]
# args=["pi","t11","t22","mu_k","c2","sigma","x0","beta_w","beta_l"]
# lower=[0.0,0.0,0.0,0.0,0.0, 0.05,-1.0,-10.0, -10.0]
# upper=[1.0,1.0,1.0,1.0,10.0, 10.0,  1.0,10.0,10.0]
# model="pit11t22Mukc2SigmaX0Beta_wBeta_l"


# #Full model2
# consts=["mu_k","mu_b","tau_w","tau_l","lambda"]
# args=["pi","t11","t22","c2","c4","sigma","x0","beta_w","beta_l"]
# lower=[0.0,0.0,0.0,0.0,0.0, 0.05,   -1.0,-10.0, -10.0]
# upper=[1.0,1.0,1.0,10.0,10.0, 10.0,  1.0,10.0,10.0]
# model="pit11t22c2c4SigmaX0Beta_wBeta_l"


#Real data_set
consts=["sigma","c4","x0","tau_w","tau_l","lambda"]
args=["pi","t11","t22","mu_k","c2","mu_b","beta_w","beta_l","beta_bias"]
lower=[0.0,0.0,   0.0,  0.0,  0.0, -10.0, -10.0,   -10.0,    -10.0]
upper=[1.0,1.0,   1.0,  10.0,10.0, 10.0,  10.0,    10.0,     10.0]
model="N24"
name_of_file = "session_N24_test2.jld"

#Ntrials=10000
PossibleOutputs=[1,2]
Nstates=2
Nconditions=10
NDataSets=1
data_set_num=5
#icondition=30
ParamInitial=zeros(length(args))
PiInitial=zeros(2)
for icondition in 1:Nconditions
    println("Initial condition: ", icondition)
   for data_set_num in 1:NDataSets
        println("Dataset:  ",data_set_num)
        data_filename=path_data*name_of_file
        data=load(data_filename)
        #y=data["y"] #this might need to be updated when fitting the entire model
        y=zeros(length(consts))
        for i in 1:length(consts)
            y[i]=data["param"][consts[i]]
        end
        #y[1]=1.0 ##### REMOVE THIS #### only for testing that we can find good models with diferent parameters
        println(data["param"])

        for iparam in 1:length(lower)
            ParamInitial[iparam]=lower[iparam]+ (upper[iparam]-lower[iparam])*rand()
        end
        XInitial=ParamInitial[4:end] #i need to separate the parameters fitted in the different maximization phases

        #Fit using originial parameters as initial parameters
        #try
            # PFit,TFit,PiFit,Ll,ParamFit,xfit=fitBaumWelchAlgorithm(data["stim"][1:Ntrials],data["delays"],data["idelays"][1:Ntrials],
            # data["choices"][1:Ntrials],data["past_choices"][1:Ntrials,:],data["past_rewards"][1:Ntrials,:],args,ParamInitial,XInitial,lower,upper,
            # PossibleOutputs,Nstates,consts,y)

            PFit,TFit,PiFit,Ll,ParamFit,xfit=fitBaumWelchAlgorithm_Nsessions(data["stim"],data["delays"],data["idelays"],
            data["choices"],data["past_choices"],data["past_rewards"],args,ParamInitial,XInitial,lower,upper,
            PossibleOutputs,Nstates,consts,y)

            _,_,Pstate,_=ProbabilityState_Nsessions(PFit,TFit,data["choices"],PiFit)

            println("ParamFit",ParamFit)
            #PFit,TFit,PiFit,Ll,ParamFit,xfit=0,0,0,0,0,0
            path_final=path_results*model*"Nsessions_dataset"*string(data_set_num)*"/"

            if  isdir(path_final)==false
                mkdir(path_final)
            end

            ifile=0
            filename=path_final*"initial_condition"*string(ifile)*".jld"
            while isfile(filename)
                ifile=ifile+1
                filename=path_final*"initial_condition"*string(ifile)*".jld"
            end

            print(PFit)
            print(filename)
            JLD.save(filename,"PFit",PFit,"TFit",TFit,"PiFit",PiFit,"LL_all",Ll,"LL",Ll[end],"ParamFit",
            ParamFit,"xfit",xfit,"args",args,"consts",consts,"y",y,"lower",lower,"upper",upper,"PiInitial",PiInitial,
            "ParamInitial",ParamInitial,"XInitial",XInitial,"Pstate",Pstate)

        # catch
        #     println("Something wrong during EM algorithm")
        # end


    end
end


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
