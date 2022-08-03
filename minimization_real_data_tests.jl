#using PyPlot
using Statistics
using Optim
using ForwardDiff
using JLD
using LineSearches
using ArgParse

auxpath=pwd()
if occursin("Users",auxpath)
    path_functions="C:\\Users\\Tiffany\\Google Drive\\WORKING_MEMORY\\MODEL\\HMM_wm_mice-main\\functions\\"
    path_figures="C:\\Users\\Tiffany\\Google Drive\\WORKING_MEMORY\\MODEL\\HMM_wm_mice-main\\figures\\"
    path_data="C:\\Users\\Tiffany\\Google Drive\\WORKING_MEMORY\\MODEL\\HMM_wm_mice-main\\real\\"
    path_results="C:\\Users\\Tiffany\\Google Drive\\WORKING_MEMORY\\MODEL\\HMM_wm_mice-main\\results\\"
else
    path_functions="/home/tiffany/HMM_wm_mice-main/functions/"
    path_figures="/home/tiffany/HMM_wm_mice-main/figures/"
    path_data="/home/tiffany/HMM_wm_mice-main/real/" # when using synthetic data
    path_results="/home/tiffany/HMM_wm_mice-main/results/"
end

include(path_functions*"functions_wm_mice.jl")
include(path_functions*"function_simulations.jl")
include(path_functions*"functions_mle.jl")
#include(path_functions*"behaviour_analysis.jl")

s = ArgParseSettings()
@add_arg_table s begin
    "--save_choices"
        help = "Boolean to specify whether we want to save the behavioral values"
        arg_type = Bool
        default = false
    "--loops"
        help = "Number of loops that we want the code to do"
        arg_type = Int64
        default = 10
    "--tau"
        help = "tau for a given animal"
        arg_type = Float64
        default = 2.2
    "--test"
        help = "test which values to include in model or not"
        arg_type = String
        default = "classic"
    "subject"
        help = "specify which animal to run the minimize"
        required = true

end

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

parsed_args = parse_args(ARGS, s)
name = parsed_args["subject"]
save_choices_local = parsed_args["save_choices"]
Nconditions = parsed_args["loops"]
tau = parsed_args["tau"]
test = parsed_args["test"]

# name = "E12"
# save_choices_local = true
# Nconditions = 10
# tau=2.6

if test == "classic"
    #Real data_set
    consts=["sigma","c4","x0","tau_w","tau_l","lambda"]
    y = [1.00, 1.00, 0, tau, tau, 0]
    args=["pi","t11","t22","mu_k","c2","mu_b","beta_w","beta_l","beta_bias"]
    lower=[0.0,0.0,   0.0,  0.0,  0.0, -10.0, -10.0,   -10.0,    -10.0]
    upper=[1.0,1.0,   1.0,  10.0,10.0, 10.0,  10.0,    10.0,     10.0]
    model="pit11t22Mukc2MubBeta_wBeta_lBeta_bias_"
end

if test == "mub_x0"
    #without mu_b and x0
    consts=["sigma","c4","x0","tau_w","tau_l","lambda","mu_b"]
    y = [1.00, 1.00, 0, tau, tau, 0,0]
    args=["pi","t11","t22","mu_k","c2","beta_w","beta_l","beta_bias"]
    lower=[0.0,0.0,   0.0,  0.0,  0.0,  -10.0,   -10.0,    -10.0]
    upper=[1.0,1.0,   1.0,  10.0,10.0,  10.0,    10.0,     10.0]
    model="pit11t22Mukc2Beta_wBeta_lBeta_bias_"
end

if test == "mub"
    #without x0
    consts=["sigma","c4","tau_w","tau_l","lambda","mu_b"]
    y = [1.00, 1.00, tau, tau, 0,0]
    args=["pi","t11","t22","mu_k","c2","x0","beta_w","beta_l","beta_bias"]
    lower=[0.0,0.0,   0.0,  0.0,  0.0,  -10.0,-10.0,   -10.0,    -10.0]
    upper=[1.0,1.0,   1.0,  10.0,10.0,  10.0, 10.0,    10.0,     10.0]
    model="pit11t22Mukc2x0Beta_wBeta_lBeta_bias_"
end

if test == "full"
    #with x0, mub and bias
    consts=["sigma","c4","tau_w","tau_l","lambda"]
    y = [1.00, 1.00, tau, tau, 0]
    args=["pi","t11","t22","mu_k","c2","x0","mu_b","beta_w","beta_l","beta_bias"]
    lower=[0.0,0.0,   0.0,  0.0,  0.0, -10.0, -10.0, -10.0,   -10.0,    -10.0]
    upper=[1.0,1.0,   1.0,  10.0,10.0, 10.0, 10.0,  10.0,    10.0,     10.0]
    model="pit11t22Mukc2x0MubBeta_wBeta_lBeta_bias_"
end

if test == "bias"
    #with x0, mub and not bias
    consts=["sigma","c4","tau_w","tau_l","lambda", "beta_bias"]
    y = [1.00, 1.00, tau, tau, 0,0]
    args=["pi","t11","t22","mu_k","c2","x0","mu_b","beta_w","beta_l"]
    lower=[0.0,0.0,   0.0,  0.0,  0.0, -10.0, -10.0, -10.0,   -10.0]
    upper=[1.0,1.0,   1.0,  10.0,10.0, 10.0, 10.0,  10.0,    10.0]
    model="pit11t22Mukc2x0MubBeta_wBeta_l"
end

if test=="mub_bias"
    #with x0 and not bias or mub
    consts=["sigma","c4","tau_w","tau_l","lambda","beta_bias","mu_b"]
    y = [1.00, 1.00, tau, tau, 0,0,0]
    args=["pi","t11","t22","mu_k","c2","x0","beta_w","beta_l"]
    lower=[0.0,0.0,   0.0,  0.0,  0.0, -10.0, -10.0,   -10.0]
    upper=[1.0,1.0,   1.0,  10.0,10.0, 10.0, 10.0,    10.0]
    model="pit11t22Mukc2x0Beta_wBeta_l"
end

if test=="x0_bias"
    #with x0 and not bias or mub
    consts=["sigma","c4","tau_w","tau_l","lambda","beta_bias","x0"]
    y = [1.00, 1.00, tau, tau, 0,0,0]
    args=["pi","t11","t22","mu_k","c2","mu_b","beta_w","beta_l"]
    lower=[0.0,0.0,   0.0,  0.0,  0.0, -10.0, -10.0,   -10.0]
    upper=[1.0,1.0,   1.0,  10.0,10.0, 10.0, 10.0,    10.0]
    model="pit11t22Mukc2MubBeta_wBeta_l"
end
#Dataset without after correct and after error
# consts=["sigma","c4","x0","tau_w","tau_l","lambda"]
# y = [1.00, 1.00, 0, tau, tau, 0]
# args=["pi","t11","t22","mu_k","c2","mu_b","beta_w"]
# lower=[0.0,0.0,   0.0,  0.0,  0.0, -10.0, -10.0]
# upper=[1.0,1.0,   1.0,  10.0,10.0, 10.0,  10.0]
# # model="pit11t22c2c4SigmaX0Beta_wBeta_l_"
# model="pit11t22Mukc2MubBeta_wlBeta_bias_"

#Ntrials=10000
PossibleOutputs=[1,2]
Nstates=2
NDataSets=1
data_set_num=1
#icondition=30
ParamInitial=zeros(length(args))
PiInitial=zeros(2)

for icondition in 1:Nconditions
    println("Initial condition: ", icondition)
    flush(stdout)
   for data_set_num in 1:NDataSets
        data_filename=path_data*name*".jld"
        data=load(data_filename)
        println("Session number: ", length(data["stim"]))

        #y=data["y"] #this might need to be updated when fitting the entire model
        # y=zeros(length(consts))
        # for i in 1:length(consts)
        #     y[i]=data["param"][consts[i]]
        # end
        #y[1]=1.0 ##### REMOVE THIS #### only for testing that we can find good models with diferent parameters
        # println(data["param"])

        for iparam in 1:length(lower)
            ParamInitial[iparam]=lower[iparam]+ (upper[iparam]-lower[iparam])*rand()
        end
        XInitial=ParamInitial[4:end] #i need to separate the parameters fitted in the different maximization phases

        #Fit using originial parameters as initial parameters
        #try
            # PFit,TFit,PiFit,Ll,ParamFit,xfit=fitBaumWelchAlgorithm(data["stim"][1:Ntrials],data["delays"],data["idelays"][1:Ntrials],
            # data["choices"][1:Ntrials],data["past_choices"][1:Ntrials,:],data["past_rewards"][1:Ntrials,:],args,ParamInitial,XInitial,lower,upper,
            # PossibleOutputs,Nstates,consts,y)

            PFit,TFit,PiFit,Ll,ParamFit,xfit,iter=fitBaumWelchAlgorithm_Nsessions(data["stim"],data["delays"],data["idelays"],
            data["choices"],data["past_choices"],data["past_rewards"],args,ParamInitial,XInitial,lower,upper,
            PossibleOutputs,Nstates,consts,y)

            _,_,Pstate,_=ProbabilityState_Nsessions(PFit,TFit,data["choices"],PiFit)

            println("ParamFit",ParamFit)
            #PFit,TFit,PiFit,Ll,ParamFit,xfit=0,0,0,0,0,0
            path_final=path_results*model*"_"*string(name)*"/"

            if  isdir(path_final)==false
                mkdir(path_final)
            end

            ifile=0
            filename=path_final*"initial_condition"*string(ifile)*".jld"
            while isfile(filename)
                ifile=ifile+1
                filename=path_final*"initial_condition"*string(ifile)*".jld"
            end

            println(filename)
            if iter < 1000
                if save_choices_local
                    JLD.save(filename,"PFit",PFit,"TFit",TFit,"PiFit",PiFit,"LL_all",Ll,"LL",Ll[end],"ParamFit",
                    ParamFit,"xfit",xfit,"args",args,"consts",consts,"y",y,"lower",lower,"upper",upper,"PiInitial",PiInitial,
                    "ParamInitial",ParamInitial,"XInitial",XInitial,"Pstate",Pstate,"stim",data["stim"],"idelays", data["idelays"], "choices",
                    data["choices"],"past_choices",data["past_choices"],"past_rewards", data["past_rewards"], "day",data["day"])
                    println("Saved behavioral variables")
                    save_choices = false

                else
                    JLD.save(filename,"PFit",PFit,"TFit",TFit,"PiFit",PiFit,"LL_all",Ll,"LL",Ll[end],"ParamFit",
                    ParamFit,"xfit",xfit,"args",args,"consts",consts,"y",y,"lower",lower,"upper",upper,"PiInitial",PiInitial,
                    "ParamInitial",ParamInitial,"XInitial",XInitial,"Pstate",Pstate)
                end
            else
                println("Maximum iterations reached")
                flush(stdout)
            end
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
