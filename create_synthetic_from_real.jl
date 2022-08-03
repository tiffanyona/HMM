
using PyPlot
using Statistics
using Optim
using ForwardDiff
using JLD
using LineSearches
using Pandas
using JSON

auxpath=pwd()
# if occursin("Users",auxpath)
#     path_functions="/Users/genis/wm_mice/HMM_wm_mice/functions/"
#     path_figures="/Users/genis/wm_mice/figures/"
#     path_synthetic_data="/Users/genis/wm_mice/synthetic_data/"
# else
#     path_functions="/home/genis/wm_mice/scripts/functions/"
#     path_figures="/home/genis/wm_mice/figures/"
#     path_synthetic_data="/home/genis/wm_mice/synthetic_data/"
#
# end
if occursin("Users",auxpath)
    path_functions="C:\\Users\\Tiffany\\Google Drive\\WORKING_MEMORY\\MODEL\\HMM_wm_mice-main\\functions\\"
    path_figures="C:\\Users\\Tiffany\\Google Drive\\WORKING_MEMORY\\MODEL\\HMM_wm_mice-main\\figures\\"
    path_data="C:\\Users\\Tiffany\\Google Drive\\WORKING_MEMORY\\MODEL\\HMM_wm_mice-main\\real\\"

else
    path_functions="/home/tiffany/HMM_wm_mice-main/functions/"
    path_figures="/home/tiffany/HMM_wm_mice-main/figure/"
    path_data="/home/tiffany/HMM_wm_mice-main/synthetic/"

end

include(path_functions*"functions_wm_mice.jl")
include(path_functions*"function_simulations.jl")
include(path_functions*"functions_mle.jl")
include(path_functions*"behaviour_analysis.jl")

#Param1
# PDwDw=0.95
# PBiasBias=0.95
# consts=["mu_k","c4","mu_b","tau_w","tau_l","lambda"]
# y=[   1.0,   1.0, 0.00,    2.5,     2.5, 0.01]
# args=["pi","t11","t22","c2","sigma","x0","beta_w","beta_l","beta_bias"]
# x=[ 1.0,PDwDw, PBiasBias,  4.5,   1.0, 0.0,   5.0,   1.0, -3]
# data_set_num=1

println("Loading")
model="pit11t22Mukc2MubBeta_wBeta_lBeta_bias_"
# model= "pit11t22c2c4SigmaX0Beta_wBeta_l"
name = "E11"

path_final=path_data*name*"_fit.json"
println(path_final)
data = JSON.parsefile(path_final)

mydict=Dict(data["args"]["1"]=> data["ParamFit"]["1"],data["args"]["2"]=> data["ParamFit"]["2"],
data["args"]["3"]=> data["ParamFit"]["3"],data["args"]["4"]=> data["ParamFit"]["4"],
data["args"]["5"]=> data["ParamFit"]["5"],data["args"]["6"]=> data["ParamFit"]["6"],
data["args"]["7"]=> data["ParamFit"]["7"],data["args"]["8"]=> data["ParamFit"]["8"],data["args"]["0"]=> data["ParamFit"]["0"])

tau = 1.3
PDwDw=mydict["t11"]
PBiasBias=mydict["t22"]

#Real data_set
consts=["sigma","c4","x0","tau_w","tau_l","lambda"]
y = [1.00, 1.00, 0, tau, tau, 0]
args=["pi","t11","t22","mu_k","c2","mu_b","beta_w","beta_l","beta_bias"]
x=[ mydict["pi"], PDwDw, PBiasBias, mydict["mu_k"], mydict["c2"], mydict["mu_b"], mydict["beta_w"], mydict["beta_l"], mydict["beta_bias"]]

# model="pit11t22c2c4SigmaX0Beta_wBeta_l_"
model="pit11t22Mukc2MubBeta_wBeta_lBeta_bias_"

param=make_dict(args,x,consts,y)
# delays=[0.0,100,200,300,500,800,1000,10000]
delays=[0.0,1000,3000,10000]

#Ntrials=300
Nsessions=50
#choices,state,stim,past_choices,past_rewards,idelays=create_data(Ntrials,delays,args,x)

PiInitialOriginal=[1 0]
initial_state=1
PossibleOutputs=[1,2]

choices,state,stim,past_choices,past_rewards,idelays=create_data_Nsessions(Nsessions,delays,param,initial_state)
choices2=choices[1]
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

# choices2=reshape(transpose(choices),Ntrials*Nsessions)
# state2=reshape(transpose(state),Ntrials*Nsessions)
# stim2=reshape(transpose(stim),Ntrials*Nsessions)
# idelays2=reshape(transpose(idelays),Ntrials*Nsessions)
# past_rewards2=reshape(permutedims(past_rewards,(2,1,3)),(Ntrials*Nsessions,10))
# past_choices2=reshape(permutedims(past_choices,(2,1,3)),(Ntrials*Nsessions,10))

results,repeat=standard_analysis(choices2,stim2,state2,delays,idelays2,past_choices2)
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

filename_save=path_data*name*"_synthetic_all.jld"
JLD.save(filename_save,"param",param,"consts",consts,"LlOriginal",LlOriginal,
"PiInitialOriginal",PiInitialOriginal,"TOriginal",T,"results",results,
"choices",choices,"state",state,"stim",stim,"past_choices",past_choices,
"past_rewards",past_rewards,"idelays",idelays,"POriginal",POriginal,
"POriginal_Nsession",POriginal_Nsession,"delays",delays, "day", "day")

# ------------- Save in json as well

# dict=Dict(:choices=>choices,:stim=>stim,:past_choices=>past_choices,:past_rewards=>past_rewards,:idelays=>idelays)

# PAST_CHOICES=[]
# PAST_REWARDS=[]
# Ntrials=1000
# for itrial in 1:Ntrials
#     push!(PAST_CHOICES,past_choices[itrial,:])
#     push!(PAST_REWARDS,past_rewards[itrial,:])
# end
#
# Saving the data for the model
dict=Dict(:param=> param, :consts=> consts, :LlOriginal=> LlOriginal, :PiInitialOriginal=> PiInitialOriginal, :TOriginal=> T,
:results=> results,:POriginal=> POriginal, :POriginal_Nsession=> POriginal_Nsession, :delays=> delays)

# pass data as a json string (how it shall be displayed in a file)
stringdata = JSON.json(dict)

# write the file with the stringdata variable information
filename_save=path_data*name*"_synthetic_params_.json"
open(filename_save , "w") do f
        write(f, stringdata)
     end

# Saving data for the useful variables
dict=Dict(:choices=> choices,:state=> state,:stim=> stim, :past_choices=> past_choices,:past_rewards=> past_rewards,
:idelays=> idelays)

df=Pandas.DataFrame(dict)
filename_save=path_data*name*"_synthetic_behavior.json"
Pandas.to_json(df,filename_save)

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

# PossibleOutputs=[1,2]

#### compute ECLL Original ####
#POriginal=ComputeEmissionProb(stim,delays,idelays,choices,past_choices,past_rewards,args,x,consts,y)
#LlOriginal=ComputeECLL_aux(POriginal,T,choices,PiInitialOriginal)

#Fit using originial parameters as initial parameters
#PNew,TNew,PiNew,Ll,ParamFit,xfit=fitBaumWelchAlgorithm(stim,delays,idelays,choices,past_choices,past_rewards,args,x,lower,upper,T,PiInitialOriginal,PossibleOutputs,consts,y)


#Fit random  parameters as initial parameters
# xini=[ 0.4,   0.8, 0.0,   4.0,   1.0]
# PDwDw2=0.5
# PBiasBias2=0.7
# Tx=[PDwDw2 1-PDwDw2; 1-PBiasBias2 PBiasBias2]
# PNew2,TNew2,PiNew2,Ll2,ParamFit2,xfit2=fitBaumWelchAlgorithm(stim,delays,idelays,choices,past_choices,past_rewards,args,xini,lower,upper,Tx,PiInitialOriginal,PossibleOutputs,consts,y)
# PiNew2,PFwdState2,PBackState2,Pstate2,xi=ProbabilityState(PNew,TNew,choices,PiNew)



#Fit using originial parameters as initial parameters other initial param
# #x=[0.8,0.9, -2]
# PDwDw=0.8
# PBiasBias=0.8
# Tx=[PDwDw 1-PDwDw; 1-PBiasBias PBiasBias]
#
# PNew,TNew,PiNew,Ll,ParamFit,xfit=fitBaumWelchAlgorithm(stim,delays,idelays,choices,past_choices,past_rewards,args,x,lower,upper,Tx,PiInitialOriginal,PossibleOutputs,consts,y)



# Nconditions=2
# Nstates=2
# XInitial=zeros(Nconditions,length(lower))
# TInitialAll=zeros(Nconditions,Nstates,Nstates)
# ConfideceIntervals=zeros(Nconditions,length(lower)+Nstates)
# Ll=zeros(Nconditions)
# ParamFit=zeros(Nconditions,length(lower)+Nstates)
# PiInitial=zeros(Nconditions,Nstates)
#
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
#     PNew,TNew,PiNew,Ll[icondition],ParamFit[icondition,:],xfit=fitBaumWelchAlgorithm(stim,delays,idelays,choices,past_choices,past_rewards,args,XInitial[icondition,:],lower,upper,TInitial,PiInitial[icondition,:],PossibleOutputs,consts,y)
#     ConfideceIntervals[icondition,:]=ComputeConfidenceIntervals(stim,delays,idelays,choices,past_choices,past_rewards,args,xfit,lower,upper,TNew,PiNew,PossibleOutputs,consts,y)
#
# end


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
