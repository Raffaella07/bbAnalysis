import sys
import os
import ROOT
import numpy as np
from ROOT import RDataFrame as rd
from ROOT import RooFit
import cmsstyle as CMS
import time

ROOT.ROOT.EnableImplicitMT()
#some useful c++ routines
ROOT.gInterpreter.Declare("""
                          
#include <ROOT/RVec.hxx>
#include <Math/Vector4D.h>
#include <vector>
#include<fstream>
using namespace ROOT::VecOps;                        
int JetMatch(float B_mass,float B_pt, float B_eta,float B_phi,const RVec<float>& Jet_mass, const RVec<float>& Jet_pt,const RVec<float>& Jet_eta,const RVec<float>& Jet_phi ) {
    std::fstream f("example.txt", std::fstream::app);
    TLorentzVector B;
    B.SetPtEtaPhiM(B_pt,B_eta,B_phi,B_mass);

    TLorentzVector jet[Jet_pt.size()];

    float deltaR[Jet_pt.size()]; 
    float deltaPt[Jet_pt.size()]; 
    float deltaM[Jet_pt.size()]; 
                          
    float deltaRMin = 0.4;
    int min_idx = -1;
                          
    for(int i=0; i<Jet_pt.size(); i++){
                          
        jet[i].SetPtEtaPhiM(Jet_pt[i],Jet_eta[i],Jet_phi[i],Jet_mass[i]);
        
        deltaR[i] = B.DeltaR(jet[i]);
//        deltaPt[i] = (B.Pt() - jet[i].Pt())/B.Pt();
//        deltaM[i] = (B.M() - jet[i].M())/B.M();

        if (deltaR[i] < deltaRMin){                              
            deltaRMin = deltaR[i];
            min_idx  = i;
        }                                                
    }                   
    
    return min_idx;                                                                     
                        
}
""")

Bdecays = ["BToKMuMu"]

#variables to store for reco B mesons
B_features = ["mass","eta","phi","pt","isMatched","kIdx","l1Idx","l2Idx"]

#variables to store for bjets 
bJet_features = ["mass","eta","phi","pt","rawFactor","bReg2018","qgl","neHEF","neEmEF","muEF","chHEF","chEmEF","area","nElectrons","nConstituents","puId","nMuons"]

#variables to store for trig-mu
muon_features = ["pt","eta","phi","dz","dxy","dzErr","dxyErr","ip3d","sip3d","pfRelIso03_all","pfRelIso04_all","ptErr","isGlobal","isTracker"]


#plotting + labeling array for common plots 
common =  [["mass",[0,50],"Mass (GeV)", ROOT.RDF.TH1DModel("", "", 25, 0, 50),ROOT.RDF.TH1DModel("", "", 100, 0, 5)],    \
            ["eta",[-4,4],"#eta",ROOT.RDF.TH1DModel("", "", 50, -4,4), ROOT.RDF.TH1DModel("", "", 100, 0, 5)],   \
            ["phi",[-4,4],"#phi",ROOT.RDF.TH1DModel("", "",50, -4,4), ROOT.RDF.TH1DModel("", "", 100, 0, 5)],    \
            ["pt",[0,100],"p_T (GeV)",ROOT.RDF.TH1DModel("", "", 50, 0,100), ROOT.RDF.TH1DModel("", "", 100, 0, 5)]]

#location of the tested tuples 
nano = rd("Events","/pnfs/psi.ch/cms/trivcat/store/user/ratramon/Btob_UL/BuToJpsiK_BMuonFilter_TuneCP5_13TeV-pythia8-evtgen/crab_BuToKJPsi_MuFilter/240305_141338/0000/BParkingNANO_Run3_mc_2024Mar05_*1.root")

#running on BuToJpsiK - check if the sig candidate is reconstructed 
nano = nano.Filter("nBToKMuMu>0","B cand existence")

#checks for jets 
nano.Filter("nJet>0","jet existence")
print("before matching")


#take only the B decays matched to MC
for decay in Bdecays:
    for f in B_features:

        nano = nano.Define(decay+"_matched_"+f,decay+"_"+f+"[ROOT::VecOps::ArgMax("+decay+"_isMatched)]" )
#idx = nano.AsNumpy(["BToKMuMu_matched_mass"])
#print(idx)




#implement match to full B meson candidate
nano = nano.Filter("BToKMuMu_matched_isMatched==1","MC match ==1 ")
#define vars for BToKMuMu mother
nano = nano.Define("BToKMuMu_matched_motherIdx","GenPart_genPartIdxMother[GenPart_genPartIdxMother[ProbeTracks_genPartIdx[BToKMuMu_matched_kIdx]]]")
nano = nano.Define("BToKMuMu_matched_mother_pdgId","GenPart_pdgId[GenPart_genPartIdxMother[GenPart_genPartIdxMother[ProbeTracks_genPartIdx[BToKMuMu_matched_kIdx]]]]")
nano = nano.Filter("abs(BToKMuMu_matched_mother_pdgId) == 5 || (abs(BToKMuMu_matched_mother_pdgId) > 500 && abs(BToKMuMu_matched_mother_pdgId) < 600)")
nano = nano.Define("BToKMuMu_matched_mother_pt","GenPart_pt[GenPart_genPartIdxMother[GenPart_genPartIdxMother[ProbeTracks_genPartIdx[BToKMuMu_matched_kIdx]]]]")
nano = nano.Define("BToKMuMu_matched_mother_eta","GenPart_eta[GenPart_genPartIdxMother[GenPart_genPartIdxMother[ProbeTracks_genPartIdx[BToKMuMu_matched_kIdx]]]]")
nano = nano.Define("BToKMuMu_matched_mother_phi","GenPart_phi[GenPart_genPartIdxMother[GenPart_genPartIdxMother[ProbeTracks_genPartIdx[BToKMuMu_matched_kIdx]]]]")
nano = nano.Define("BToKMuMu_matched_mother_mass","GenPart_mass[GenPart_genPartIdxMother[GenPart_genPartIdxMother[ProbeTracks_genPartIdx[BToKMuMu_matched_kIdx]]]]")
idx = nano.AsNumpy(["BToKMuMu_matched_mother_pdgId"])

#define trigger muon sel + trigger muon features to save:
nano = nano.Define("B_trgMuon_idx","int idx; if (Muon_isTriggering[BToKMuMu_matched_l1Idx]==1){idx = BToKMuMu_matched_l1Idx;} else if  (Muon_isTriggering[BToKMuMu_matched_l2Idx]==1){idx = BToKMuMu_matched_l2Idx;} return idx;")
nano = nano.Filter("B_trgMuon_idx!=-1 ","trig muon in event")

#muon features for muon inside matched BToKMuMu candidate 
for f in muon_features:
     nano = nano.Define("Muon_matched_"+f,"Muon_"+f+"[B_trgMuon_idx]" ) 

#just a check point
idx = nano.AsNumpy(["Muon_matched_pt"])
print(idx)

#match BToKMuMu gen matched candidate to jet  
nano = nano.Define("Jet_BToKMuMuMatched_idx","JetMatch(BToKMuMu_matched_mass,BToKMuMu_matched_pt,BToKMuMu_matched_eta, BToKMuMu_matched_phi, Jet_mass,Jet_pt, Jet_eta, Jet_phi )")
print("after matching")

idx = nano.AsNumpy(["Jet_BToKMuMuMatched_idx"])
print(len(idx["Jet_BToKMuMuMatched_idx"]))
nano = nano.Filter("Jet_BToKMuMuMatched_idx!=-1"," matched jet B pair ")

nano = nano.Define("Jet_size","Jet_mass.size()")
nano = nano.Filter("Jet_size>0","jet presence")
mass = nano.AsNumpy(["Jet_size"])

#features of the jet matched to the BToKMuMu (gen-matched) candidate 
for f in bJet_features:

    nano = nano.Define("Jet_matched_"+f,"Jet_"+f+"[Jet_BToKMuMuMatched_idx]" )

nano = nano.Define("Jet_matched_partonFlavour","Jet_partonFlavour[Jet_BToKMuMuMatched_idx]" )
mass = nano.AsNumpy(["Jet_matched_partonFlavour"])
print(mass)

#filters filters filters 
nano = nano.Filter("abs(Jet_matched_partonFlavour)==5 ","b flav jet match")
nano = nano.Filter("Jet_genJetIdx[Jet_BToKMuMuMatched_idx]!=-1")
nano = nano.Filter("fabs(GenJet_partonFlavour[Jet_genJetIdx[Jet_BToKMuMuMatched_idx]])==5")

nano = nano.Define("GenJet_matched_pt","GenJet_pt[Jet_genJetIdx[Jet_BToKMuMuMatched_idx]]" )
nano = nano.Define("GenJet_matched_eta","GenJet_eta[Jet_genJetIdx[Jet_BToKMuMuMatched_idx]]" )
nano = nano.Define("GenJet_matched_phi","GenJet_phi[Jet_genJetIdx[Jet_BToKMuMuMatched_idx]]" )
nano = nano.Define("GenJet_matched_mass","GenJet_mass[Jet_genJetIdx[Jet_BToKMuMuMatched_idx]]" )

print(nano.Report().Print())

#have a pT raw variable
nano =  nano.Define("Jet_matched_pTraw","Jet_matched_pt*(1-Jet_matched_rawFactor)")

#build the GenJet_pt/Jen_pt ratio with pt raw 
nano =  nano.Define("Jet_GenJet_pTratio","GenJet_matched_pt/Jet_matched_pTraw")

#do plots?
doPlot = False

if (doPlot):
    for f in common: 
    
        CMS.SetExtraText("Simulation Preliminary")
        CMS.SetLumi("")
        print(f[1])
    
        hist_bjet = nano.Histo1D(f[3],"Jet_matched_"+f[0])
        hist_b = nano.Histo1D(f[3],"BToKMuMu_matched_"+f[0])
        hist_quark = nano.Histo1D(f[3],"BToKMuMu_matched_mother_"+f[0])
        hist_genJet = nano.Histo1D(f[3],"GenJet_matched_"+f[0])
       
    
        hist_bjet.SetName("b jet")
        hist_b.SetName("B meson")
        hist_quark.SetName(" b quark")
        hist_genJet.SetName(" b GenJet")
    
   
        hist_bjet.Scale(1./hist_bjet.Integral())
        hist_b.Scale(1./hist_b.Integral())
        hist_quark.Scale(1./hist_quark.Integral())
        hist_genJet.Scale(1./hist_genJet.Integral())
        max_q = float(hist_quark.GetMaximum())
        max_b = float(hist_b.GetMaximum())
        max_jet = float(hist_bjet.GetMaximum())
        max_genjet = float(hist_genJet.GetMaximum())
        canva = CMS.cmsCanvas('', f[1][0], f[1][1], 0, max(max_q,max_b,max_jet,max_genjet)*1.2, f[2], 'normalized units', square = CMS.kSquare, extraSpace=0.01, iPos=0)
        leg = CMS.cmsLeg(0.55, 0.6, 0.9, 0.80, textSize=0.05)
        
        CMS.cmsDraw(hist_bjet, "hist", lcolor= ROOT.kRed, fcolor = 0,lwidth =3 )
        CMS.cmsDraw(hist_b, "hist", lcolor= ROOT.kBlue-6,fcolor = 0,lwidth =3 )
        CMS.cmsDraw(hist_quark, "hist", lcolor= ROOT.kGreen-3, fcolor = 0,lwidth =3 )
        CMS.cmsDraw(hist_genJet, "hist",lcolor= ROOT.kOrange+7,fcolor = 0,lwidth =3 )
    
        leg.AddEntry(hist_bjet.GetName(), "bjet", "l")
        leg.AddEntry(hist_b.GetName(), "B meson", "l")
        leg.AddEntry(hist_quark.GetName(), "quark", "l")
        leg.AddEntry(hist_genJet.GetName(), "b GenJet", "l")
        
    
        CMS.SaveCanvas(canva, "../plots/overlay_"+f[0]+".pdf")
    
    for f in common: 
    
        CMS.SetExtraText("Preliminary")
        CMS.SetLumi("")
        hist_b_jet = nano.Define("Delta_jet_B_"+f[0],"(Jet_matched_"+f[0]+"/BToKMuMu_matched_"+f[0]+")").Histo1D(f[4],"Delta_jet_B_"+f[0])
        hist_GenJet_jet = nano.Define("Delta_jet_GenJet_"+f[0],"(Jet_matched_"+f[0]+"/GenJet_matched_"+f[0]+")").Histo1D(f[4],"Delta_jet_GenJet_"+f[0])
    
        hist_b_jet.SetName("#Delta B meson- bJet")
        hist_GenJet_jet.SetName("#Delta GenJet - Jet")
    

        hist_b_jet.Scale(1./hist_b_jet.Integral())
        hist_GenJet_jet.Scale(1./hist_GenJet_jet.Integral())
        max_b = float(hist_b_jet.GetMaximum())
        max_j = float(hist_GenJet_jet.GetMaximum())
        canva = CMS.cmsCanvas('', 0, 5, 0, max(max_b,max_j)*1.2, f[2], 'normalized units', square = CMS.kSquare, extraSpace=0.01, iPos=0)
        leg = CMS.cmsLeg(0.55, 0.6, 0.9, 0.80, textSize=0.05)
        
        CMS.cmsDraw(hist_b_jet, "hist", lcolor= ROOT.kRed, fcolor = 0,lwidth =3 )
        CMS.cmsDraw(hist_GenJet_jet, "hist", lcolor= ROOT.kBlue-6,fcolor = 0,lwidth =3 )
    
        leg.AddEntry(hist_b_jet.GetName(), hist_b_jet.GetName(), "l")

save_cols = ["GenJet_matched_pt", \  
             #"GenJet_matched_eta","GenJet_matched_phi","GenJet_matched_mass","BToKMuMu_matched_mother_pt", \
            #"BToKMuMu_matched_mother_eta","BToKMuMu_matched_mother_phi","BToKMuMu_matched_mother_pdgId", \
            #"BToKMuMu_matched_pt","BToKMuMu_matched_eta","BToKMuMu_matched_phi","BToKMuMu_matched_mass", \
            "Jet_matched_mass","Jet_matched_eta","Jet_matched_phi","Jet_matched_pt","Jet_matched_pTraw","Jet_matched_bReg2018","Jet_matched_qgl","Jet_matched_neHEF","Jet_matched_neEmEF","Jet_matched_muEF","Jet_matched_chHEF","Jet_matched_chEmEF","Jet_matched_area","Jet_matched_nElectrons","Jet_matched_nConstituents","Jet_matched_puId","Jet_matched_nMuons"
            ,"Jet_GenJet_pTratio",\
            "Muon_matched_pt","Muon_matched_eta","Muon_matched_phi","Muon_matched_dz","Muon_matched_dxy","Muon_matched_dzErr","Muon_matched_dxyErr","Muon_matched_ip3d","Muon_matched_sip3d","Muon_matched_pfRelIso03_all","Muon_matched_pfRelIso04_all","Muon_matched_ptErr"]#,"Muon_matched_isGlobal","Muon_matched_isTracker"]

#do you wanna save a tree?

#nano.Snapshot("Events","reg_flat.root",save_cols)

#save numpy dict to be directly fed in the trainer

reg_input = nano.AsNumpy(save_cols)

np.save('../data/reg_tuples.py',reg_input)



