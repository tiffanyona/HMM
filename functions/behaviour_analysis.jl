using PyPlot


function standard_analysis(choices,stim,state,delays,idelays,past_choices,plot=true)
    #change stim and choices from 1,2 to -1,1
    stim_aux=zeros(length(stim))
    choices_aux=zeros(length(stim))
    a=findall(x->x==1,stim)
    b=findall(x->x==2,stim)

    stim_aux[a].=-1
    stim_aux[b].=1

    a=findall(x->x==1,choices)
    b=findall(x->x==2,choices)
    choices_aux[a].=-1
    choices_aux[b].=1
    rewards=((choices_aux.*stim_aux).+1)./2.0
    ##############  P correct vs delay ############
    Pc_delay=zeros(length(delays))
    PcDwDelay=zeros(length(delays))
    PcBiasDelay=zeros(length(delays))

    output_data=Dict()

    for idelay in 1:length(delays)
        println(idelay)
        index=findall(x->x==idelay,idelays)
        Pc_delay[idelay]=mean( rewards[index])
        state2=state[index]
        rewards2=rewards[index]
        index_dw=findall(x->x==1,state2)
        index_bias=findall(x->x==2,state2)

        PcDwDelay[idelay]=mean( rewards2[index_dw])
        PcBiasDelay[idelay]=mean( rewards2[index_bias])


    end


    output_data["Pc_delay"]=Pc_delay
    output_data["PcDwDelay"]=PcDwDelay
    output_data["PcBiasDelay"]=PcBiasDelay
    PyPlot.figure()
    PyPlot.plot(delays,Pc_delay,"ko-",label="Model")
    PyPlot.plot(delays,PcDwDelay,"go-",label="DW")
    PyPlot.plot(delays,PcBiasDelay,"o-",color="Purple",label="Bias")

    PyPlot.title("P correct")

    repeat=(choices_aux.*past_choices[:,1].+1)/2.
    Prep_delay=zeros(length(delays))
    PrepDwDelay=zeros(length(delays))
    PrepBiasDelay=zeros(length(delays))

    PrepBiasDelayCorrect=zeros(length(delays))
    PrepBiasDelayError=zeros(length(delays))

    for idelay in 1:length(delays)
        println(idelay)
        index=findall(x->x==idelay,idelays)
        Prep_delay[idelay]=mean(repeat[index])

        state2=state[index]
        repeat2=repeat[index]
        reward2=rewards[index]

        index_dw=findall(x->x==1,state2)
        index_bias=findall(x->x==2,state2)
        reward2=reward2[index_bias]
        repeat3=repeat2[index_bias]

        PrepDwDelay[idelay]=mean( repeat2[index_dw])
        PrepBiasDelay[idelay]=mean( repeat2[index_bias])

        indexCorrect=findall(x->x==1,reward2)
        indexError=findall(x->x==0,reward2)

        PrepBiasDelayCorrect[idelay]=mean( repeat3[indexCorrect])
        PrepBiasDelayError[idelay]=mean( repeat3[indexError])

    end

    output_data["Prep_delay"]=Prep_delay
    output_data["PrepDwDelay"]=PrepDwDelay
    output_data["PrepBiasDelay"]=PrepBiasDelay
    PyPlot.figure()
    PyPlot.plot(delays,Prep_delay,"ko-")
    PyPlot.plot(delays,PrepDwDelay,"go-")
    PyPlot.plot(delays,PrepBiasDelay,"o-",color="Purple")
    PyPlot.title("P rep")


    Pr=zeros(length(delays))
    PrBias=zeros(length(delays))
    PrDw=zeros(length(delays))

    choices_r=(choices_aux.+1)/2

    for idelay in 1:length(delays)
        index=findall(x->x==idelay,idelays)
        Pr[idelay]=mean(choices_r[index])

        state2=state[index]
        choices_r2=choices_r[index]

        index_dw=findall(x->x==1,state2)
        index_bias=findall(x->x==2,state2)

        PrBias[idelay]=mean(choices_r2[index_bias])
        PrDw[idelay]=mean(choices_r2[index_dw])

    end
    output_data["Pr"]=Pr
    output_data["PrBias"]=PrBias
    output_data["PrDw"]=PrDw

    PyPlot.figure()
    PyPlot.plot(delays,Pr,"ko-")
    PyPlot.plot(delays,PrDw,"go-")
    PyPlot.plot(delays,PrBias,"o-",color="Purple")
    PyPlot.title("Prob right")

    return output_data,repeat

end



function vcat_all(list)
    aux=list[1]
    Nsessions=length(list)
    for isession in 2:Nsessions
        aux=vcat(aux,list[isession])
    end

    return aux

end
