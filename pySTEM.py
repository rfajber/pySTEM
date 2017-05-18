import numpy as np

class STEM:

    ################################# Constants and intialization

    #geometrical factor for joint mass distributions
    const=np.sqrt(2.*np.pi)*6.371*10**6/9.8
    #heat capacity 
    Cp=1.004*10**3
    #radius of earth
    a=6.371*10**6
    
    #initialize the object with the control data
    def __init__(self,vm,om,vo,o2,L,P,T): 
        self.L=L
        self.P=P
        self.T=T
        self.Th=.5*(T[1:]+T[:-1])#mid point grid 
        self.vm=vm
        self.om=om
        self.vo=vo
        self.o2=o2

    ################################# Basic STEM Calculatinns

    def calc_m(self,Lat=None,Plev=None,Tlev=None):
        if Lat is None:
            Lat=self.L
            iLA=0
            iLB=len(Lat)
        else:
            iLA=np.where(self.L==Lat[0])[0]
            iLB=iLA+np.shape(Lat)[0]
        if Plev is None:
            Plev=self.P
            iPA=0
            iPB=len(Plev)
        else:
            iPA=np.where(self.P==Plev[0])[0]
            iPB=iPA+np.shape(Plev)[0]
        if Tlev is None:
            Tlev=self.T
        #resize the variables to the same size as the input data, whatever that is
        #get the right subset of data to use 
        vm=self.vm[iPA:iPB,iLA:iLB]
        om=self.om[iPA:iPB,iLA:iLB]
        vo=self.vo[iPA:iPB,iLA:iLB]
        o2=self.o2[iPA:iPB,iLA:iLB]
        #grid data onto (T,P,L) space for calculation
        vm=np.tile(vm,(len(Tlev),1,1))
        om=np.tile(om,(len(Tlev),1,1))
        vo=np.tile(vo,(len(Tlev),1,1))
        o2=np.tile(o2,(len(Tlev),1,1))
        Tgrid=np.tile(Tlev,(len(Plev),len(Lat),1))
        Tgrid=np.transpose(Tgrid,[2,0,1])
        #geometrical factor 
        geo_fac=self.const*np.tile(np.cos(Lat/180.*np.pi),[len(Tlev),len(Plev),1])
        #exponential factor
        E=np.exp(-.5*np.power(Tgrid-om,2)/o2)
        #joint mass fluxes 
        mM=geo_fac*vm/np.power(o2,.5)*E
        mE=geo_fac*vo*(Tgrid-om)/np.power(o2,1.5)*E
        
        return mM,mE

    def calc_M(self,m,Plev):
        if Plev is None:
            Plev=self.P
        return np.trapz(x=Plev,y=m,axis=1)

    def set_M(self,Lat=None,Plev=None,Tlev=None,out=False):
        if Lat is None:
            Lat=self.L
        if Plev is None:
            Plev=self.P
        if Tlev is None:
            Tlev=self.T
        mM,mE=self.calc_m(Lat,Plev,Tlev)
        MM=self.calc_M(mM,Plev)
        ME=self.calc_M(mE,Plev)
        if out==True:
            return MM,ME
        else:
            self.MM=MM
            self.ME=ME

    def calc_S(self,M,Tlev=None):
        if Tlev is None:
            Tlev=self.T
        Tgrid=np.tile(Tlev,(np.shape(M)[1],1))
        Tgrid=np.transpose(Tgrid,[1,0])
        return np.cumsum(.5*(Tgrid[1:]-Tgrid[:-1])*(M[1:]+M[:-1]),axis=0)

    def set_S(self,Lat=None,Plev=None,Tlev=None,out=False):
        if Lat is None:
            Lat=self.L
        if Plev is None:
            Plev=self.P
        if Tlev is None:
            Tlev=self.T
        mM,mE=self.calc_m(Lat,Plev,Tlev)
        MM=self.calc_M(mM,Plev)
        ME=self.calc_M(mE,Plev)
        SM=self.calc_S(MM,Tlev)
        SE=self.calc_S(ME,Tlev)
        if out==True:
            return SM,SE
        else:
            self.SM=SM
            self.SE=SE
        
    def calc_H(self,M,Tlev=None):
        if Tlev is None:
            Tlev=self.T
        Tgrid=np.tile(Tlev,(np.shape(M)[1],1))
        Tgrid=np.transpose(Tgrid,[1,0])
        return np.trapz(x=Tlev,y=self.Cp*Tgrid*M,axis=0)
    
    def set_H(self,Lat=None,Plev=None,Tlev=None,out=False):
        if Lat is None:
            Lat=self.L
        if Plev is None:
            Plev=self.P
        if Tlev is None:
            Tlev=self.T
        mM,mE=self.calc_m(Lat,Plev,Tlev)
        MM=self.calc_M(mM,Plev)
        ME=self.calc_M(mE,Plev)
        HM=self.calc_H(MM,Tlev)
        HE=self.calc_H(ME,Tlev)
        if out==True:
            return HM,HE
        else:
            self.HM=HM
            self.HE=HE
        
    ################################# Sensitivities

    def calc_dMxdy(self,Lat=None,Plev=None,Tlev=None):
        if Lat is None:
            Lat=self.L
            iLA=0
            iLB=len(Lat)
        else:
            iLA=np.max(np.where(self.L==Lat[0]))
            iLB=iLA+np.shape(Lat)[0]
        if Plev is None:
            Plev=self.P
            iPA=0
            iPB=len(Plev)
        else:
            iPA=np.max(np.where(self.P==Plev[0]))
            iPB=iPA+np.shape(Plev)[0]
        if Tlev is None:
            Tlev=self.T
        #resize the variables to the same size as the input data, whatever that is
        #get the right subset of data to use 
        vm=self.vm[iPA:iPB,iLA:iLB]
        om=self.om[iPA:iPB,iLA:iLB]
        vo=self.vo[iPA:iPB,iLA:iLB]
        o2=self.o2[iPA:iPB,iLA:iLB]
        #grid data onto (T,P,L) space for calculation
        vm=np.tile(vm,(len(Tlev),1,1))
        om=np.tile(om,(len(Tlev),1,1))
        vo=np.tile(vo,(len(Tlev),1,1))
        o2=np.tile(o2,(len(Tlev),1,1))
        Tgrid=np.tile(Tlev,(len(Plev),len(Lat),1))
        Tgrid=np.transpose(Tgrid,[2,0,1])
        #geometrical factor 
        geo_fac=self.const*np.tile(np.cos(Lat/180.*np.pi),[len(Tlev),len(Plev),1])
        #exponential factor
        E=np.exp(-.5*np.power(Tgrid-om,2)/o2)

        #sensitivity calvulations 
        dMMdvm=geo_fac*E/o2
        dMMdom=geo_fac*E/np.power(o2,1.5)*(Tgrid-om)*vm
        dMMdo2=geo_fac*E/np.power(o2,1.5)*(np.power(Tgrid-om,2)/o2-1.)*vm/2.
        dMEdvo=geo_fac*E/np.power(o2,1.5)*(Tgrid-om)
        dMEdom=geo_fac*E/np.power(o2,1.5)*vo*(np.power(Tgrid-om,2)/o2-1.)
        dMEdo2=geo_fac*E/np.power(o2,2.5)*.5*vo*(Tgrid-om)*(np.power(Tgrid-om,2)/o2-3.)

        return dMMdvm,dMMdom,dMMdo2,dMEdvo,dMEdom,dMEdo2

    def set_dMxdy(self,Lat=None,Plev=None,Tlev=None,out=False):
        if Lat is None:
            Lat=self.L
        if Plev is None:
            Plev=self.P
        if Tlev is None:
            Tlev=self.T
        dMMdvm,dMMdom,dMMdo2,dMEdvo,dMEdom,dMEdo2=self.calc_dMxdy(Lat,Plev,Tlev)
        if out==True:
            return dMMdvm,dMMdom,dMMdo2,dMEdvo,dMEdom,dMEdo2
        else:
            self.dMMdvm=dMMdvm
            self.dMMdom=dMMdom
            self.dMMdo2=dMMdo2
            self.dMEdvo=dMEdvo
            self.dMEdom=dMEdom
            self.dMEdo2=dMEdo2

    def calc_dSxdy(self,M,Tlev=None):
        if Tlev is None:
            Tlev=self.T
        Tgrid=np.tile(Tlev,(np.shape(M)[1],np.shape(M)[2],1))
        Tgrid=np.transpose(Tgrid,[2,0,1])
        return np.cumsum(.5*(Tgrid[1:]-Tgrid[:-1])*(M[1:]+M[:-1]),axis=0)

    def set_dSxdy(self,Lat=None,Plev=None,Tlev=None,out=False):
        if Lat is None:
            Lat=self.L
        if Plev is None:
            Plev=self.P
        if Tlev is None:
            Tlev=self.T
        dMMdvm,dMMdom,dMMdo2,dMEdvo,dMEdom,dMEdo2=self.calc_dMxdy(Lat,Plev,Tlev)
        dSMdvm=self.calc_dSxdy(dMMdvm,Tlev)
        dSMdom=self.calc_dSxdy(dMMdom,Tlev)
        dSMdo2=self.calc_dSxdy(dMMdo2,Tlev)
        dSEdvo=self.calc_dSxdy(dMEdvo,Tlev)
        dSEdom=self.calc_dSxdy(dMEdom,Tlev)
        dSEdo2=self.calc_dSxdy(dMEdo2,Tlev)
        if out==True:
            return dSMdvm,dSMdom,dSMdo2,dSEdvo,dSEdom,dSEdo2
        else:
            self.dSMdvm=dSMdvm
            self.dSMdom=dSMdom
            self.dSMdo2=dSMdo2
            self.dSEdvo=dSEdvo
            self.dSEdom=dSEdom
            self.dSEdo2=dSEdo2

    def calc_dHxdy(self,M,Tlev=None):
        if Tlev is None:
            Tlev=self.T
        Tgrid=np.tile(Tlev,(np.shape(M)[1],np.shape(M)[2],1))
        Tgrid=np.transpose(Tgrid,[2,0,1])
        return np.trapz(x=Tlev,y=self.Cp*Tgrid*M,axis=0)

    def set_dHxdy(self,Lat=None,Plev=None,Tlev=None,out=False):
        if Lat is None:
            Lat=self.L
        if Plev is None:
            Plev=self.P
        if Tlev is None:
            Tlev=self.T
        dMMdvm,dMMdom,dMMdo2,dMEdvo,dMEdom,dMEdo2=self.calc_dMxdy(Lat,Plev,Tlev)
        dHMdvm=self.calc_dHxdy(dMMdvm,Tlev)
        dHMdom=self.calc_dHxdy(dMMdom,Tlev)
        dHMdo2=self.calc_dHxdy(dMMdo2,Tlev)
        dHEdvo=self.calc_dHxdy(dMEdvo,Tlev)
        dHEdom=self.calc_dHxdy(dMEdom,Tlev)
        dHEdo2=self.calc_dHxdy(dMEdo2,Tlev)
        if out==True:
            return dHMdvm,dHMdom,dHMdo2,dHEdvo,dHEdom,dHEdo2
        else:
            self.dHMdvm=dHMdvm
            self.dHMdom=dHMdom
            self.dHMdo2=dHMdo2
            self.dHEdvo=dHEdvo
            self.dHEdom=dHEdom
            self.dHEdo2=dHEdo2

    def set_dHxdy_plus(self,Lat=None,Plev=None,Tlev=None,out=False):
        if Lat is None:
            Lat=self.L
        if Plev is None:
            Plev=self.P
        if Tlev is None:
            Tlev=self.T
        MM,ME=self.set_M(Lat,Plev,Tlev,out=True)
        dMMdvm,dMMdom,dMMdo2,dMEdvo,dMEdom,dMEdo2=self.calc_dMxdy(Lat,Plev,Tlev)

        MM_mask=np.copy(MM)
        MM[MM<0]=0.
        MM[MM>0]=1.
        ME_mask=np.copy(ME)
        ME[ME<0]=0.
        ME[ME>0]=1.

        for ii in range(0,len(Plev)):
            dMMdvm[:,ii,:]*=MM_mask
            dMMdom[:,ii,:]*=MM_mask
            dMMdo2[:,ii,:]*=MM_mask
            dMEdvo[:,ii,:]*=ME_mask
            dMEdom[:,ii,:]*=ME_mask
            dMEdo2[:,ii,:]*=ME_mask

        dHMdvm=self.calc_dHxdy(dMMdvm,Tlev)
        dHMdom=self.calc_dHxdy(dMMdom,Tlev)
        dHMdo2=self.calc_dHxdy(dMMdo2,Tlev)
        dHEdvo=self.calc_dHxdy(dMEdvo,Tlev)
        dHEdom=self.calc_dHxdy(dMEdom,Tlev)
        dHEdo2=self.calc_dHxdy(dMEdo2,Tlev)
        if out==True:
            return dHMdvm,dHMdom,dHMdo2,dHEdvo,dHEdom,dHEdo2
        else:
            self.dHMdvm=dHMdvm
            self.dHMdom=dHMdom
            self.dHMdo2=dHMdo2
            self.dHEdvo=dHEdvo
            self.dHEdom=dHEdom
            self.dHEdo2=dHEdo2

    def set_dHxdy_minus(self,Lat=None,Plev=None,Tlev=None,out=False):
        if Lat is None:
            Lat=self.L
        if Plev is None:
            Plev=self.P
        if Tlev is None:
            Tlev=self.T
        MM,ME=self.set_M(Lat,Plev,Tlev,out=True)
        dMMdvm,dMMdom,dMMdo2,dMEdvo,dMEdom,dMEdo2=self.calc_dMxdy(Lat,Plev,Tlev)

        MM_mask=np.copy(MM)
        MM[MM<0]=1.
        MM[MM>0]=0.
        ME_mask=np.copy(ME)
        ME[ME<0]=1.
        ME[ME>0]=0.

        for ii in range(0,len(Plev)):
            dMMdvm[:,ii,:]*=MM_mask
            dMMdom[:,ii,:]*=MM_mask
            dMMdo2[:,ii,:]*=MM_mask
            dMEdvo[:,ii,:]*=ME_mask
            dMEdom[:,ii,:]*=ME_mask
            dMEdo2[:,ii,:]*=ME_mask

        dHMdvm=self.calc_dHxdy(dMMdvm,Tlev)
        dHMdom=self.calc_dHxdy(dMMdom,Tlev)
        dHMdo2=self.calc_dHxdy(dMMdo2,Tlev)
        dHEdvo=self.calc_dHxdy(dMEdvo,Tlev)
        dHEdom=self.calc_dHxdy(dMEdom,Tlev)
        dHEdo2=self.calc_dHxdy(dMEdo2,Tlev)
        if out==True:
            return dHMdvm,dHMdom,dHMdo2,dHEdvo,dHEdom,dHEdo2
        else:
            self.dHMdvm=dHMdvm
            self.dHMdom=dHMdom
            self.dHMdo2=dHMdo2
            self.dHEdvo=dHEdvo
            self.dHEdom=dHEdom
            self.dHEdo2=dHEdo2


    #######################################################################################
