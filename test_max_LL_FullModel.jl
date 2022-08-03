using PyPlot
using Statistics
using Optim
using ForwardDiff
using JLD
using LineSearches
println("hola")

auxpath=pwd()
if occursin("Users",auxpath)
    path_functions="/Users/genis/wm_mice/HMM_wm_mice/functions/"
    path_figures="/Users/genis/wm_mice/figures/"
else
    path_functions="/home/genis/wm_mice/scripts/functions/"
    path_figures="/home/genis/wm_mice/figures/"
end

include(path_functions*"functions_wm_mice.jl")
include(path_functions*"function_simulations.jl")
include(path_functions*"functions_mle.jl")
include(path_functions*"behaviour_analysis.jl")



PDwDw=0.9
PBiasBias=0.9
PrDw=0.9
PrBias=0.3


consts=["mu_k","c4","mu_b","tau_w","tau_l","lambda"]
y=[   0.2,   1.0, 0.00,    10,     10, 0.01]

args=["c2","sigma","x0","beta_w","beta_l"]
x=[   1.2,   0.3, 0.0,   3.0,   -1.0]

lower=[ 0.0, 0.05,-1.0,-10.0, -10.0]
upper=[10.0, 10.0,  1.0,10.0,10.0]

param=make_dict(args,x,consts,y)
delays=[0.0,100,200,300,500,800,1000,10000]
Ntrials=Int(1e4)
#choices,state,stim,past_choices,past_rewards,idelays=create_data(Ntrials,delays,args,x)


T=[PDwDw 1-PDwDw; 1-PBiasBias PBiasBias]
PiInitialOriginal=[1 0]
PossibleOutputs=[1,2]
NPossibleOutputs=length(PossibleOutputs)

#choices,state,stim,past_choices,past_rewards,idelays=create_data(Ntrials,delays,T,args,x,consts,y)
results=standard_analysis(choices,stim,state,delays,idelays)

############## sanity checks data ################################3
#pr,pstate=Compute_negative_LL_hmm_module(PDwDw,PBiasBias,PrDw,PrBias,choices)
#ll=Compute_negative_LL_hmm_module(PDwDw,PBiasBias,PrDw,PrBias,choices)

################ LL vs sigma ###########
# PDwVector=0.05:0.05:0.95
# PBiasVector=0.05:0.05:0.95
#
#
# P=ComputeEmissionProb(stim,delays,idelays,choices,past_choices,past_rewards,args,x,consts,y)
# ll=ComputeNegativeLogLikelihood(P,T,choices,PiInitial)
#
# SIGMA=0.05:0.01:1
# Ll=zeros(length(SIGMA))
# for isigma in 1:length(SIGMA)
#         x[1]=SIGMA[isigma]
#         P=ComputeEmissionProb(stim,delays,idelays,choices,past_choices,past_rewards,args,x,consts,y)
#         Ll[isigma]=ComputeNegativeLogLikelihood(P,T,choices,PiInitial)
# end
#
# figure()
# plot(SIGMA,Ll)



############## fitting ############

PossibleOutputs=[1,2]

#### compute ECLL Original ####
POriginal=ComputeEmissionProb(stim,delays,idelays,choices,past_choices,past_rewards,args,x,consts,y)
LlOriginal=ComputeECLL_aux(POriginal,T,choices,PiInitialOriginal)

#Fit using originial parameters as initial parameters
PNew,TNew,PiNew,Ll,ParamFit,xfit=fitBaumWelchAlgorithm(stim,delays,idelays,choices,past_choices,past_rewards,args,x,lower,upper,T,PiInitialOriginal,PossibleOutputs,consts,y)


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
