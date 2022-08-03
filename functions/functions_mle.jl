using Distributed

@everywhere using ForwardDiff
@everywhere using StatsBase
@everywhere using LineSearches
@everywhere using Optim
@everywhere using LinearAlgebra
#@everywhere using DualNumbers
epsilon=1e-5
auxpath=pwd()
# if occursin("Users",auxpath)
#     path_functions="/Users/genis/wm_mice/HMM_wm_mice/functions/"
# else
#     path_functions="/home/genis/wm_mice/HMM_wm_mice/functions/"
# end

if occursin("Users",auxpath)
    path_functions="C:\\Users\\Tiffany\\Google Drive\\WORKING_MEMORY\\MODEL\\HMM_wm_mice-main\\functions\\"
else
    path_functions="/home/tiffany/HMM_wm_mice-main/functions/"
end

@everywhere include(path_functions*"functions_wm_mice.jl")
const tol=1e-3

function ForwardPass(P,T,choices,InitalP)
    """
    Function that computes the ForwardPass and the negative likelihood
    T is the transition Matrix it has dimension Nstates x Nstates

    P is the emission probabilities. Note that in our model the emision
    probabilities change in each trial, they depend on the history and the stimulus.
    Dimension: Ntrials x Nstates x NPossibleOutputs.

    choices the list of outputs it is an integer 1,2.. NpossibleOutputs. Dimension: 1 x Ntrials

    InitialP is the initial probabilities of the states. Dimension 1 x Nstates

    """
    Nout=length(P[1,1,:])
    Nstate=length(T[1,:])
    Ntrials=length(choices)
    PFwdState=zeros(typeof(P[1]),Ntrials,Nout)
    Norm_coeficcient=zeros(typeof(P[1]),Ntrials)
    #println("T: ", T)
    # println("P:", realpart(P[1,1,2])[1]," ",realpart(P[1,1,1]))
    for itrial in 1:Ntrials
        for istate in 1:Nstate
            if itrial==1
                aux_sum=InitalP[istate]
            else
                aux_sum=sum([ PFwdState[itrial-1,k]*T[k,istate] for k in 1:Nout])
            end

            PFwdState[itrial,istate]=aux_sum*P[itrial,istate,choices[itrial]]

        end
        Norm_coeficcient[itrial]=sum(PFwdState[itrial,:])
        PFwdState[itrial,:]=PFwdState[itrial,:]./(sum(PFwdState[itrial,:]))

    end
    if PFwdState!=PFwdState
        error("Nan PFwdState")
    end

    return PFwdState
end


function NegativeLoglikelihood_Nsessions(P,T,choices,InitalP)
    """
    Function that computes the ForwardPass and the negative likelihood when
    data is organized in Sessions.

    T is the transition Matrix it has dimension Nstates x Nstates

    P is the emission probabilities. Note that in our model the emision
    probabilities change in each trial, they depend on the history and the stimulus.
    Dimension: NsessionsxNtrials x Nstates x NPossibleOutputs.

    choices the list of outputs it is an integer 1,2.. NpossibleOutputs. Dimension: Nsessions x Ntrials

    InitialP is the initial probabilities of the states. Dimension Nsessions x Nstates

    """
    Nsessions=length(choices)
    LL=0
    for isession in 1:Nsessions
        LL=LL+NegativeLoglikelihood(P[isession],T,choices[isession],InitalP)
    end

    return LL


end



function NegativeLoglikelihood(P,T,choices,InitalP)
    """
    Function that computes the ForwardPass and the negative likelihood
    T is the transition Matrix it has dimension Nstates x Nstates

    P is the emission probabilities. Note that in our model the emision
    probabilities change in each trial, they depend on the history and the stimulus.
    Dimension: Ntrials x Nstates x NPossibleOutputs.

    choices the list of outputs it is an integer 1,2.. NpossibleOutputs. Dimension: 1 x Ntrials

    InitialP is the initial probabilities of the states. Dimension 1 x Nstates

    """
    Nout=length(P[1,1,:])
    Nstate=length(T[1,:])
    Ntrials=length(choices)
    PFwdState=zeros(typeof(P[1]),Ntrials,Nout)
    Norm_coeficcient=zeros(typeof(P[1]),Ntrials)
    #println("T: ", T)
    #println("P:", realpart(P[1,1,2])[1]," ",realpart(P[1,1,1]))
    for itrial in 1:Ntrials
        for istate in 1:Nstate
            if itrial==1
                aux_sum=InitalP[istate]
            else
                aux_sum=sum([ PFwdState[itrial-1,k]*T[k,istate] for k in 1:Nout])
            end
            PFwdState[itrial,istate]=aux_sum*P[itrial,istate,choices[itrial]]

        end
        Norm_coeficcient[itrial]=sum(PFwdState[itrial,:])
        PFwdState[itrial,:]=PFwdState[itrial,:]./(sum(PFwdState[itrial,:]))

    end
    if PFwdState!=PFwdState
        error("Nan PFwdState")
    end

    return -sum(log.(Norm_coeficcient))
end


#
# function NegativeLoglikelihood(P,T,choices,InitalP)
#     PFwdState=ForwardPass(P,T,choices,InitalP)
#     Norm_coeficcient=zeros(typeof(P[1]),Ntrials)
#     for itrial in 1:Ntrials
#         Norm_coeficcient[itrial]=sum(PFwdState[itrial,:])
#     end
#     return -sum(log.(Norm_coeficcient))
# end

function ComputeECLL_aux(P,T,choices,Pi)
    PFwdState,PBackState,Gamma,xi=ProbabilityState(P,T,choices,Pi)
    return ComputeECLL(P,Gamma,choices)
end

function ComputeECLL_full_aux(P,T,choices,Pi)
    PFwdState,PBackState,Gamma,xi=ProbabilityState(P,T,choices,Pi)

    return ComputeECLL_full(P,Gamma,choices,xi,T,Pi)
end


function ComputeECLL(P,Gamma,choices)

    ecll=0.0
    Ntrials,Nstates,Nchoices=size(P)
    for itrial in 1:Ntrials
        for istate in 1:Nstates
            ecll=ecll+Gamma[itrial,istate]*log(P[itrial,istate,choices[itrial]]+epsilon)
        end
    end
    return -ecll
end



function ComputeECLL_Nsessions(P,Gamma,choices)

    Nsessions=length(P)
    ecll=0
    for isession in 1:Nsessions
        ecll=ecll+ComputeECLL(P[isession],Gamma[isession],choices[isession])
    end

    return ecll

end

function ComputeECLL_full_Nsessions(P,Gamma,choices,xi,T,Pi)

    Nsessions=length(P)
    ecll=0
    for isession in 1:Nsessions
        ecll=ecll+ComputeECLL_full(P[isession],Gamma[isession],choices[isession],xi[isession],T,Pi)
    end

    return ecll

end

function ComputeECLL_full(P,Gamma,choices,xi,T,Pi)

    ecll=0.0
    Ntrials,Nstates,Nchoices=size(P)
    #println(Ntrials,Nstates,Nchoices)
    LL_pi=zeros(Nstates)
    LL_T=zeros(Ntrials-1)
    LL_p=zeros(Ntrials)
    for istate in 1:Nstates
        #ecll=ecll+Gamma[1,istate]*log(Pi[istate]+epsilon)
        LL_pi[istate]=Gamma[1,istate]*log(Pi[istate]+epsilon)
    end

    #println("size xi",size(xi))
    for itrial in 1:Ntrials-1
        for istate in 1:Nstates
            for jstate in 1:Nstates
                #ecll=ecll+xi[itrial,istate,jstate]*log(T[istate,jstate]+epsilon)
                LL_T[itrial]=LL_T[itrial]+xi[itrial,istate,jstate]*log(T[istate,jstate]+epsilon)
            end
        end
    end

    for itrial in 1:Ntrials
        for istate in 1:Nstates
            #ecll=ecll+Gamma[itrial,istate]*log(P[itrial,istate,choices[itrial]]+epsilon)
            LL_p[itrial]=LL_p[itrial]+Gamma[itrial,istate]*log(P[itrial,istate,choices[itrial]]+epsilon)
        end
    end
    #return LL_pi,LL_T,LL_p,sum(LL_pi),sum(LL_T),sum(LL_p),-(sum(LL_pi)+sum(LL_T)+sum(LL_p))

    return -(sum(LL_pi)+sum(LL_T)+sum(LL_p))
end


function ProbabilityState_Nsessions(P,T,choices,InitalP)
    """
    Function that computes the ForwardPass,
    the backwardPass, the posteriors and xi
    T is the transition Matrix it has dimension Nstates x Nstates

    P is the emission probabilities. Note that in our model the emision
    probabilities change in each trial, they depend on the history and the stimulus.
    Dimension: Nsessions x Ntrials x Nstates x NPossibleOutputs.

    choices the list of outputs it is an integer 1,2.. NpossibleOutputs. Dimension: Nsessions x Ntrials

    InitialP is the initial probabilities of the states. Dimension 1 x Nstates

    this function returns:
    Pstate[1,:],initial state probabilities
    PFwdState,Probabilities from the forwards pass for each session
    PBackState,Probabilities from the backward pass for each session
    Pstate, State probabilities (GAMMA) for each session
    xi for each session
    """
    Nsessions=length(P)
    PFwdState=[]
    PBackState=[]
    Pstate=[]
    xi=[]

    for isession in 1:Nsessions
        aux_fwd,aux_back,aux_Pstate,aux_xi=ProbabilityState(P[isession],T,choices[isession],InitalP)
        push!(PFwdState,aux_fwd)
        push!(PBackState,aux_back)
        push!(Pstate,aux_Pstate)
        push!(xi,aux_xi)
    end

    return PFwdState,PBackState,Pstate,xi

end


function ProbabilityState(P,T,choices,InitalP)
    """
    Function that computes the ForwardPass,
    the backwardPass, the posteriors and xi
    T is the transition Matrix it has dimension Nstates x Nstates

    P is the emission probabilities. Note that in our model the emision
    probabilities change in each trial, they depend on the history and the stimulus.
    Dimension: Ntrials x Nstates x NPossibleOutputs.

    choices the list of outputs it is an integer 1,2.. NpossibleOutputs. Dimension: 1 x Ntrials

    InitialP is the initial probabilities of the states. Dimension 1 x Nstates

    this function returns:
    Pstate[1,:],initial state probabilities
    PFwdState,Probabilities from the forwards pass
    PBackState,Probabilities from the backward pass
    Pstate, State probabilities (GAMMA)
    xi
    """



    FinalProb=[1,1]
    Ntrials=length(choices)
    Nstates=length(T[1,:])
    PFwdState=zeros(Ntrials,Nstates)
    PBackState=zeros(Ntrials,Nstates)
    # for itrial in 1:Ntrials
    #     #compute forward pass
    #     for istate in 1:Nstates
    #         if itrial==1
    #             aux_sum=InitalP[istate]
    #         else
    #             aux_sum=sum([ PFwdState[itrial-1,k]*T[k,istate] for k in 1:Nstates])
    #         end
    #         PFwdState[itrial,istate]=aux_sum*P[istate,choices[itrial]]
    #     end
    #     PFwdState[itrial,:]=PFwdState[itrial,:]./(sum(PFwdState[itrial,:]))
    # end

    PFwdState=ForwardPass(P,T,choices,InitalP)

    #compute backward pass
    for itrial in 1:Ntrials
        i=Ntrials+1-itrial
        for istate in 1:Nstates
            if itrial==1
                PBackState[i,istate]=FinalProb[istate]
            else
                PBackState[i,istate]=sum(  [ T[istate,k]*P[i+1,k,choices[i+1]]*PBackState[i+1,k] for k in 1:Nstates]   )
            end
        end
        PBackState[i,:]=PBackState[i,:]./sum(PBackState[i,:])
    end


    #compute merge state probabilities
    Pstate=zeros(typeof(P[1]),Ntrials,Nstates)
    for itrial in 1:Ntrials
        Pstate[itrial,:]=(PFwdState[itrial,:].*PBackState[itrial,:])./(sum(PFwdState[itrial,:].*PBackState[itrial,:]))
    end



    #compute transition prob at time t from i to j
    xi=zeros( typeof(P[1]), Ntrials-1, Nstates,Nstates )
    for itrial in 1:Ntrials-1
        #####cumpute normalitzation#####
        norm=0
        for i in 1:Nstates
            for j in 1:Nstates
                norm=norm+PFwdState[itrial,i]*T[i,j]*P[itrial+1,j,choices[itrial+1]]*PBackState[itrial+1,j]
            end
        end

        for i in 1:Nstates
            for j in 1:Nstates
                xi[itrial,i,j]=(PFwdState[itrial,i]*T[i,j]*P[itrial+1,j,choices[itrial+1]]*PBackState[itrial+1,j])/norm
            end
        end
    end

    return PFwdState,PBackState,Pstate,xi
end



function MaximizeEmissionProbabilities(stim,delays,idelays,choices,past_choices,past_rewards,Gamma,args,x,lower,upper,param,consts=0,y=0)
        z=zeros(typeof(x[1]),length(x))
        z[:]=x[:]
        # for i in 1:length(x)
        #     z[i]=x[i]
        # end
        function MaxEmission(z)
            if z!=z
                println("Nan in max emision",z)
                error("Nan in max emision")
                return 99999999

                #error("Fucking nans in MaxEmission")
            #else
                #println("zzzz",z[1]," ",z[2])
            else
                if z!=z
                    error("Fucking nans in MaxEmission")
                end
                P=ComputeEmissionProb(stim,delays,idelays,choices,past_choices,past_rewards,args,z,consts,y)
                if P !=P
                    error("Fucking nans in Ps: ",z)
                end
                ll=ComputeECLL(P,Gamma,choices)
                #println("ll: ",ll)
                if ll !=ll
                    error("Fucking nans in ComputeECLL: ",z,P)
                end
                return ll + sum(param["lambda"]*abs.(z.^2))
            end
        end

        #Optim.optimize( MaxEmission,x)

        #res=optimize(MaxEmission, x, LBFGS(); autodiff = :forward)
        #res=optimize(MaxEmission, xx, LBFGS(); autodiff = :forward)
        #lower=[0.05]
        #upper=[3]

        #res=optimize(MaxEmission,lower,upper, z, Fminbox(GradientDescent(linesearch = BackTracking())),Optim.Options(g_tol=1e-5); autodiff = :forward)
        if z!=z
            error("optimization is feed with  NaN:",z)
        end


        #res=optimize(MaxEmission, z, NelderMead() )

        #res=optimize(MaxEmission, z, LBFGS(linesearch = BackTracking(order=2),alphaguess = InitialStatic(scaled=true) ),Optim.Options(g_tol=1e-5,show_trace=false) ; autodiff = :forward)

        #res=optimize(MaxEmission, z, GradientDescent(),Optim.Options(g_tol=1e-5,show_trace=false); autodiff = :forward)

        #res=optimize(MaxEmission, z, LBFGS(),Optim.Options(g_tol=1e-5,show_trace=true); autodiff = :forward)

        res=optimize(MaxEmission,lower,upper, z, Fminbox(LBFGS(linesearch = BackTracking(order=2))),Optim.Options(show_trace=false,g_tol=1e-5); autodiff = :forward)

        #res=optimize(MaxEmission,lower,upper, z, Fminbox(LBFGS(linesearch = LineSearches.HagerZhang())),Optim.Options(show_trace=false,g_tol=1e-5); autodiff = :forward)


        #res=optimize(MaxEmission,lower,upper, z, Fminbox(LBFGS(linesearch = BackTracking())),Optim.Options(g_tol=1e-5); autodiff = :forward)

        #res=optimize(MaxEmission,lower,upper, xx, Fminbox(LBFGS()),Optim.Options(show_trace=true); autodiff = :forward)
        #res=optimize(MaxEmission, xx, LBFGS(); autodiff = :forward)
        z=res.minimizer
        if z!=z
            error("optimization returns NaN:",z)
        end
        Q=ComputeEmissionProb(stim,delays,idelays,choices,past_choices,past_rewards,args,z,consts,y)
        #res=optimize(MaxEmission, xx, LBFGS())

        return res.minimizer,Q,res.minimum
end


function MaximizeEmissionProbabilities_Nsessions_real(stim,delays,idelays,choices,past_choices,past_rewards,Gamma,args,x,lower,upper,param,consts=0,y=0)
        z=zeros(typeof(x[1]),length(x))
        z[:]=x[:]
        # for i in 1:length(x)
        #     z[i]=x[i]
        # end
        function MaxEmission(z)
            if z!=z
                println("Nan in max emision",z)
                error("Nan in max emision")
                return 99999999

                #error("Fucking nans in MaxEmission")
            #else
                #println("zzzz",z[1]," ",z[2])
            else
                if z!=z
                    error("Fucking nans in MaxEmission")
                end
                P=ComputeEmissionProb_Nsessions_real(stim,delays,idelays,choices,past_choices,past_rewards,args,z,consts,y)
                if P !=P
                    error("Fucking nans in Ps: ",z)
                end
                ll=ComputeECLL_Nsessions(P,Gamma,choices)
                #println("ll: ",ll)
                if ll !=ll
                    error("Fucking nans in ComputeECLL: ",z,P)
                end
                return ll + sum(param["lambda"]*abs.(z.^2))
            end
        end

        #Optim.optimize( MaxEmission,x)

        #res=optimize(MaxEmission, x, LBFGS(); autodiff = :forward)
        #res=optimize(MaxEmission, xx, LBFGS(); autodiff = :forward)
        #lower=[0.05]
        #upper=[3]

        #res=optimize(MaxEmission,lower,upper, z, Fminbox(GradientDescent(linesearch = BackTracking())),Optim.Options(g_tol=1e-5); autodiff = :forward)
        if z!=z
            error("optimization is feed with  NaN:",z)
        end

        res=optimize(MaxEmission,lower,upper, z, Fminbox(LBFGS(linesearch = BackTracking(order=2))),Optim.Options(show_trace=false,g_tol=1e-5); autodiff = :forward)


        z=res.minimizer
        if z!=z
            error("optimization returns NaN:",z)
        end
        Q=ComputeEmissionProb_Nsessions_real(stim,delays,idelays,choices,past_choices,past_rewards,args,z,consts,y)

        #res=optimize(MaxEmission, xx, LBFGS())

        return res.minimizer,Q,res.minimum
end

function MaximizeEmissionProbabilities_Nsessions(stim,delays,idelays,choices,past_choices,past_rewards,Gamma,args,x,lower,upper,param,consts=0,y=0)
        z=zeros(typeof(x[1]),length(x))
        z[:]=x[:]
        # for i in 1:length(x)
        #     z[i]=x[i]
        # end
        function MaxEmission(z)
            if z!=z
                println("Nan in max emision",z)
                error("Nan in max emision")
                return 99999999

                #error("Fucking nans in MaxEmission")
            #else
                #println("zzzz",z[1]," ",z[2])
            else
                if z!=z
                    error("Fucking nans in MaxEmission")
                end
                P=ComputeEmissionProb_Nsessions(stim,delays,idelays,choices,past_choices,past_rewards,args,z,consts,y)
                if P !=P
                    error("Fucking nans in Ps: ",z)
                end
                ll=ComputeECLL_Nsessions(P,Gamma,choices)
                #println("ll: ",ll)
                if ll !=ll
                    error("Fucking nans in ComputeECLL: ",z,P)
                end
                return ll + sum(param["lambda"]*abs.(z.^2))
            end
        end

        #Optim.optimize( MaxEmission,x)

        #res=optimize(MaxEmission, x, LBFGS(); autodiff = :forward)
        #res=optimize(MaxEmission, xx, LBFGS(); autodiff = :forward)
        #lower=[0.05]
        #upper=[3]

        #res=optimize(MaxEmission,lower,upper, z, Fminbox(GradientDescent(linesearch = BackTracking())),Optim.Options(g_tol=1e-5); autodiff = :forward)
        if z!=z
            error("optimization is feed with  NaN:",z)
        end

        res=optimize(MaxEmission,lower,upper, z, Fminbox(LBFGS(linesearch = BackTracking(order=2))),Optim.Options(show_trace=false,g_tol=1e-5); autodiff = :forward)


        z=res.minimizer
        if z!=z
            error("optimization returns NaN:",z)
        end
        Q=ComputeEmissionProb_Nsessions(stim,delays,idelays,choices,past_choices,past_rewards,args,z,consts,y)

        #res=optimize(MaxEmission, xx, LBFGS())

        return res.minimizer,Q,res.minimum
end










function fitBaumWelchAlgorithm(stim,delays,idelays,choices,past_choices,past_rewards,
                            args,ParamInitial,x,lower,upper,PossibleOutputs,Nstates,
                            consts=0,y=0)
    """
    Function that computes the ForwardPass, the ECLL,
    the backwardPass, the posteriors and xi
    T is the transition Matrix it has dimension Nstates x Nstates

    P are the emission probabilities. Note that in our model the emision
    probabilities change in each trial, they depend on the history and the stimulus.
    Dimension: Ntrials x Nstates x NPossibleOutputs.

    choices the list of outputs it is an integer 1,2.. NpossibleOutputs. Dimension: 1 x Ntrials

    InitialP are the initial probabilities of the states. Dimension 1 x Nstates

    """
    # println("fit",Nstates)
    param=make_dict(args,ParamInitial,consts,y)

    z=zeros(typeof(x[1]),length(x))
    z[:]=x[:]

    Nout=length(PossibleOutputs)

    iter=1
    # println("hola fit",x)
    TNew=zeros(Nstates,Nstates)
    PNew=zeros(Nstates,Nout) #useless
    PiNew=zero(Nstates)
    POld=ComputeEmissionProb(stim,delays,idelays,choices,past_choices,past_rewards,args[4:end],x,consts,y)
    TOld=[param["t11"] 1-param["t11"] ;
         1-param["t22"] param["t22"]]
    PiOld=[param["pi"] 1-param["pi"]]

    ll=0.0
    DeltaAux=0.01
    LL=[]
    #for iter in 1:Niter-1
    #while all(tol.<delta) aixo esta malament
    while DeltaAux>tol
        #expectation step
        AlphaNew,BetaNew,GammaNew,XiNew=ProbabilityState(POld,TOld,choices,PiOld)
        llNew=ComputeECLL_full(POld,GammaNew,choices,XiNew,TOld,PiOld)
        #Maximization of transition probabilities
        #############compute new transition matrix##########
        for i in 1:Nstates
            den=sum(GammaNew[:,i])
            for j in 1:Nstates
                TNew[i,j]=sum(XiNew[:,i,j])/den
            end
        end

        #Maximization of initial probabilities
        PiNew=GammaNew[1,:]

        if TOld!=TOld
            error("Nan in T")
        end

        ######## Maximization of emission probabilities ##########
        minimizer,PNew,_=MaximizeEmissionProbabilities(stim,delays,idelays,choices,past_choices,past_rewards,GammaNew,args[4:end],z,lower[4:end],upper[4:end],param,consts,y)
        z=minimizer
        #the [4:end] removes the args and lower and upper bounds of the pi t11 and t22
        #println(z,TNew[1,1]," ",TNew[2,2]," "," ",PiNew,llNew)




        ##### compute difference between previous and current parameters ####
        #Now with negativelikelihood
        # for i in 1:Nstates*Nstates
        #     delta[i]=TAux[i]-TIter[i]
        # end
        #
        # for i in 1:Nstates*Nout
        #     delta[Nstates*Nstates+i]=PAux[i]-PIter[i]
        # end

        #delta=abs.(delta)

        #iter=iter+1

        DeltaAux=abs(llNew-ll)

        #update Arrays


        TOld=TNew
        PiOld=PiNew
        POld=PNew
        ll=llNew
        push!(LL,llNew)

    end

    param=vcat(PiNew[1],TNew[1,1],TNew[2,2],z)
    ll2=NegativeLoglikelihood(PNew,TNew,choices,PiNew)
    return PNew,TNew,PiNew,ll2,param,z


end


function fitBaumWelchAlgorithm_Nsessions(stim,delays,idelays,choices,past_choices,past_rewards,
                            args,ParamInitial,x,lower,upper,PossibleOutputs,Nstates,
                            consts=0,y=0)
    """
    Function that computes the ForwardPass, the ECLL,
    the backwardPass, the posteriors and xi
    T is the transition Matrix it has dimension Nstates x Nstates

    P are the emission probabilities. Note that in our model the emision
    probabilities change in each trial, they depend on the history and the stimulus.
    Dimension: Nsessions x Ntrials x Nstates x NPossibleOutputs.

    choices the list of outputs it is an integer 1,2.. NpossibleOutputs. Dimension: 1 x Ntrials

    InitialP are the initial probabilities of the states. Dimension 1 x Nstates

    it returns:
        PNew Probabilities of Right and Left for each trial and each module
        TNew Fitted transition probabilities
        PiNew fitted initial transition
        ll2 Negative log likelihood
        param fitted parameters
        z fitted parameters of the modules (Wm module and history module)

    """
    Nsessions=length(stim)
    # println("fit: ",Nstates)
    param=make_dict(args,ParamInitial,consts,y)

    z=zeros(typeof(x[1]),length(x))
    z[:]=x[:]

    Nout=length(PossibleOutputs)

    iter=1
    # println("Inside minimizing, fiting Bauch Welch ",x)
    TNew=zeros(Nstates,Nstates)
    PNew=zeros(Nstates,Nout)
    PiNew=zero(Nstates)

    if "t11" in args
        POld=ComputeEmissionProb_Nsessions(stim,delays,idelays,choices,past_choices,past_rewards,args[4:end],x,consts,y)
    else
        println("without WM")
        POld=ComputeEmissionProb_Nsessions(stim,delays,idelays,choices,past_choices,past_rewards,args,x,consts,y)
    end

    #args[4:end]: we can not pass all args to the ComputeEmissionProb, I remove t11, t22 and pi

    TOld=[param["t11"] 1-param["t11"] ;
         1-param["t22"] param["t22"]]
    PiOld=[param["pi"] 1-param["pi"]]

    ll=0.0
    DeltaAux=0.01
    LL=[]
    #for iter in 1:Niter-1
    #while all(tol.<delta) aixo esta malament
    while DeltaAux>tol && iter <1000
        iter+=1
        println("Minimising, delta bigger than tolerance: ", DeltaAux)
        flush(stdout)
        #expectation step
        AlphaNew,BetaNew,GammaNew,XiNew=ProbabilityState_Nsessions(POld,TOld,choices,PiOld)
        llNew=ComputeECLL_full_Nsessions(POld,GammaNew,choices,XiNew,TOld,PiOld)


        #Maximization of transition probabilities
        #############compute new transition matrix##########

        for i in 1:Nstates
            den=0
            for isession in 1:Nsessions
                den=den+sum(GammaNew[isession][:,i]) #sum over session and state
            end
            for j in 1:Nstates
                aux_T=0
                for isession in 1:Nsessions
                    aux_T=aux_T+sum(XiNew[isession][:,i,j])
                end
                TNew[i,j]=aux_T/den
            end
        end

        #Maximization of initial probabilities
        aux_pi=0
        for isession in 1:Nsessions
            aux_pi=aux_pi+GammaNew[isession][1,1]
        end
        PiNew=[aux_pi/Nsessions,1-aux_pi/Nsessions] #this is only valid if there is only 2 states


        if TOld!=TOld
            error("Nan in T")
        end

        ######## Maximization of emission probabilities ##########
        if "t11" in args
            minimizer,PNew,_=MaximizeEmissionProbabilities_Nsessions(stim,delays,idelays,choices,past_choices,past_rewards,GammaNew,args[4:end],z,lower[4:end],upper[4:end],param,consts,y)
        else
            println("without WM 2")
            minimizer,PNew,_=MaximizeEmissionProbabilities_Nsessions(stim,delays,idelays,choices,past_choices,past_rewards,GammaNew,args,z,lower,upper,param,consts,y)
        end
        
        z=minimizer
        #the [4:end] removes the args and lower and upper bounds of the pi t11 and t22
        #println(z,TNew[1,1]," ",TNew[2,2]," "," ",PiNew,llNew)




        ##### compute difference between previous and current parameters ####
        #Now with negativelikelihood
        # for i in 1:Nstates*Nstates
        #     delta[i]=TAux[i]-TIter[i]
        # end
        #
        # for i in 1:Nstates*Nout
        #     delta[Nstates*Nstates+i]=PAux[i]-PIter[i]
        # end

        #delta=abs.(delta)

        #iter=iter+1

        DeltaAux=abs(llNew-ll)

        #update Arrays


        TOld=TNew
        PiOld=PiNew
        POld=PNew
        ll=llNew
        push!(LL,llNew)

    end
    println(iter)
    param=vcat(PiNew[1],TNew[1,1],TNew[2,2],z)
    ll2=NegativeLoglikelihood_Nsessions(PNew,TNew,choices,PiNew)
    return PNew,TNew,PiNew,ll2,param,z,iter


end

function ComputeConfidenceIntervals(stim,delays,idelays,choices,past_choices,past_rewards,args,x,lower,upper,TFit,PiFit,PossibleOutputs,consts=0,y=0,z_aux=1.96)
    Nstates=length(TFit[1,:])
    Nout=length(PossibleOutputs)
    PARAM=zeros(typeof(TFit[1]),Nstates+length(x))
    for istate in 1:Nstates
        PARAM[istate]=TFit[istate,istate]
    end


    for iparam in 1:length(x)
        PARAM[iparam+Nstates]=x[iparam]
    end

    function ComputeNegativeLogLikelihood2(PARAM)
        T=zeros(typeof(PARAM[1]),Nstates,Nstates)

        for istate in 1:Nstates  #only valid for Nstates=2
            T[istate,istate]=PARAM[istate]
        end
        T[1,2]=1-T[1,1]
        T[2,1]=1-T[2,2]

        PFit=ComputeEmissionProb(stim,delays,idelays,choices,past_choices,past_rewards,args,PARAM[Nstates+1:end],consts,y)


        return ComputeNegativeLogLikelihood(PFit,T,choices,PiFit)

    end

    H=ForwardDiff.hessian(ComputeNegativeLogLikelihood2, PARAM)
    #This Hessian is directly the Information matrix because we compute the
    #negative likelihood
    println("ci PARAM",PARAM)
    #println(H)
    ci=zeros(length(PARAM))
    try
        HI=inv(H)
        if all(diag(HI).>0)
            ci[:]=z_aux.*sqrt.(diag(HI))
            println("A minimum ", diag(HI))
        else
            ci[:]=zeros(length(PARAM)).-1.0
            println("Not a minimum ", diag(HI))
        end

    catch e
        ci[:]=zeros(length(PARAM)).-2.0
        println("H is not invertible")
    end
    return ci
end
