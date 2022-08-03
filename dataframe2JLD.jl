using PyPlot
using ArgParse
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
    # path_data="C:\\Users\\Tiffany\\Google Drive\\WORKING_MEMORY\\MODEL\\HMM_wm_mice-main\\synthetic\\"
    path_data="C:\\Users\\Tiffany\\Google Drive\\WORKING_MEMORY\\MODEL\\HMM_wm_mice-main\\real\\"
    path_results="C:\\Users\\Tiffany\\Google Drive\\WORKING_MEMORY\\MODEL\\HMM_wm_mice-main\\results\\"
else
    path_functions="/home/tiffany/HMM_wm_mice-main/HMM_wm_mice/functions/"
    path_figures="/home/tiffany/HMM_wm_mice-main/figures/"
    # path_data="/home/tiffany/HMM_wm_mice-main/synthetic/"
    path_data="/home/tiffany/HMM_wm_mice-main/real/"
    path_results="/home/tiffany/HMM_wm_mice-main/results/"
end

include(path_functions*"functions_wm_mice.jl")
include(path_functions*"function_simulations.jl")
include(path_functions*"functions_mle.jl")
include(path_functions*"behaviour_analysis.jl")

function fromdataframetoJLD(session_name)
    PDwDw = 0.9
    PBiasBias = 0.9
    consts=["mu_k","c4","mu_b","tau_w","tau_l","lambda"]
    y=[   0.8,   1.0, 0.00,    2.5,     2.5, 0.01]
    args=["pi","t11","t22","c2","sigma","x0","beta_w","beta_l","beta_bias"]
    x=[ 1.0,PDwDw, PBiasBias,  3.5,   1.0, 0.0,   5.0,   -3.0, -1]
    param=make_dict(args,x,consts,y)

    # println("Loading")
    # model="pit11t22Mukc2MubBeta_wBeta_lBeta_bias"
    # path_final=path_results*model*"Nsessions_dataset1/"
    # filename=path_final*"initial_condition2.jld"
    # data=load(filename)

    println(session_name)
    path_final=path_data*session_name*".json"
    data = JSON.parsefile(path_final)

    # create variable to write the information
    global dict2 = Dict()
    open(path_final, "r") do f
        global dict2
          # file information to string
        dict2=JSON.parse(data)  # parse and transform data
    end

    rewards = []
    for session in 1:length(dict2["past_rewards"])
       m = zeros(length(dict2["past_rewards"][session]),10)
       for index in 1:length(dict2["past_rewards"][session])
           m[index,:] = dict2["past_rewards"][session][index]
       end
       push!(rewards,m)
    end

    choices = []
    for session in 1:length(dict2["past_choices"])
       # println(session)
       m = zeros(length(dict2["past_choices"][session]),10)
       for index in 1:length(dict2["past_choices"][session])
           m[index,:] = dict2["past_choices"][session][index]
       end
       push!(choices,m)
    end

    filename_save=path_data*session_name*".jld"
    delays=[0,1000,3000,10000]
    JLD.save(filename_save,"param",param,"consts",consts,
    "choices",dict2["choice"],"stim",dict2["stim"],"past_choices",choices,
    "past_rewards",rewards,"delays",delays,"idelays",dict2["idelays"],"day",dict2["day"])

    # JLD.save(filename_save,"param",param,"consts",consts,"LlOriginal",LlOriginal,
    # "PiInitialOriginal",PiInitialOriginal,"TOriginal",T,"results",results,
    # "choices",choices,"state",state,"stim",stim,"past_choices",past_choices,
    # "past_rewards",past_rewards,"idelays",idelays,"POriginal",POriginal,
    # "POriginal_Nsession",POriginal_Nsession,"delays",delays)
end
#
# list_name=["E03", "E04", "E05", "E06", "E07", "E08", "E09", "E10", "E11", "E12", "E13", "E14", "E15", "E16", "E17", "E18", "E19", "E20", "E21", "E22",
# "N02", "N03", "N04", "N05", "N07", "N08", "N09", "N11", "N13", "N19", "N20", "N21", "N22", "N24", "N25", "N26", "N27", "N28", "C10b", "C12", "C13", "C15", "C18", "C19", "C20", "C22", "C28", "C32", "C34", "C36" ,"C37", "C38", "C39" ]

list_name=["E03", "E04", "E05_3", "E05_10", "E06", "E07_3","E07_10", "E08", "E09", "E10", "E11", "E12_3",  "E12_10", "E13", "E14", "E15_3", "E15_10", "E16_3", "E16_10", "E17_3", "E17_10", "E18", "E19", "E20_3", "E20_10", "E21", "E22",
"N02", "N03", "N04", "N05_3", "N05_10", "N07_3", "N07_10", "N08", "N09", "N11_3", "N11_10", "N13", "N19", "N20", "N21", "N22", "N24_3", "N24_10", "N25_3",  "N25_10","N26", "N27_3", "N27_10", "N28_3", "N28_10", "C10b", "C12", "C13", "C15", "C18", "C19", "C20", "C22", "C28_3", "C28_10", "C32", "C34", "C36" ,"C37_3","C37_10", "C38", "C39" ]

for session_name in list_name
    fromdataframetoJLD(session_name)
end
