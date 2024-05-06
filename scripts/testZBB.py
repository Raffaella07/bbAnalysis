

import sys
import os
import ROOT
from ROOT import RDataFrame as rd
import numpy as np
#import samples 
import argparse


import cmsstyle as CMS
 

ROOT.gROOT.SetBatch()
ROOT.EnableImplicitMT(); 
#rd = ROOT.RDF.Experimental.Distributed.Dask.RDataFrame

	#print("Hi, I'm calling myInit, lets check if multiple workers are calling this")
ROOT.gInterpreter.Declare("""
		#include <ROOT/RVec.hxx>
		#include <Math/Vector4D.h>
		#include <vector>
		#include<fstream>
		#include <algorithm>				                     
		using namespace ROOT::VecOps;    

		const RVec<int> trig_requirement(const RVec<int> nmuons,const RVec<int> muonIdx1,const RVec<int> muonIdx2, const RVec<int> muon_isTriggering ){
		
			ROOT::VecOps::RVec<int> newVec;		
			for (int i=0; i<nmuons.size();i++){			  
			
				if (nmuons[i]==0 )newVec.push_back(-1); // no muons in jet 
				else{
									//nmuons>0, at least one muon
					//std::cout << "test" << nmuons[i] <<" "<<  muon_isTriggering[muonIdx1[i]] << " " << muonIdx2[i] <<  std::endl;		  
					if (muon_isTriggering[muonIdx1[i]]==1) newVec.push_back(muonIdx1[i]); // if first muon triggers, return index of first muon
					else if (nmuons[i]>1) { // if first muon does not trigger and there is another muon
								if ( muon_isTriggering[muonIdx2[i]]==1) newVec.push_back(muonIdx2[i]); // if first muon triggers, return index of second muon
								else newVec.push_back(-1); // else nothing
						//std::cout << "in nmuons >1 if " << muonIdx1[i] << " " << muonIdx1[i] <<  std::endl;	
					}else newVec.push_back(-1); 
				} 
			}
			//if (newVec.size()!=nmuons.size()  )std::cout  << newVec.size() << " " << nmuons.size() << " " <<std::endl;	
			return newVec;	  
	
		}						  
	
		int find_pair(const RVec<float>& Jet_trgMu,const RVec<float>& Jet_bTag,int nProbes,int idx ){
			//std::fstream f("example.txt", std::fstream::app);
			int i = 0 ;
			if (Jet_trgMu.size()==0) return -1;
			while (Jet_trgMu[i]<0 ){
			
				i++;						  
	
			}
			if (idx ==0) return i;
			else{					  
				ROOT::VecOps::RVec<float> bTag_probe;
				ROOT::VecOps::RVec<int> bTag_idx;
				for (int j = 0; j < Jet_bTag.size(); j++) {
					if (j != i) {
					bTag_probe.push_back(Jet_bTag[j]);
					bTag_idx.push_back(j);
					//f << "btag  " << Jet_bTag[j]<< endl; 
				}
			}
	
				if (bTag_probe.size()==0) return -1;
				else {
					int maxIndex = ROOT::VecOps::ArgMax(bTag_probe);
				
					//std:: cout << i << " " << maxIndex <<  " " << bTag_idx[maxIndex] <<std::endl;
					return bTag_idx[maxIndex];
					}			  
	
			}
		}
	""")
	
ROOT.gInterpreter.Declare("""
		#include <Math/Vector4D.h>
		#include <TLorentzVector.h>
		float mass(float j1_pt, float j1_eta,float j1_phi,float j1_mass, float j2_pt, float j2_eta, float j2_phi, float j2_mass)
		{
			ROOT::Math::PtEtaPhiMVector j1(j1_pt, j1_eta,j1_phi, j1_mass);
			ROOT::Math::PtEtaPhiMVector j2(j2_pt, j2_eta,j2_phi, j2_mass);
			return (j1 + j2).M();
		}
		float pt(float j1_pt, float j1_eta,float j1_phi,float j1_mass, float j2_pt, float j2_eta, float j2_phi, float j2_mass)
		{
			ROOT::Math::PtEtaPhiMVector j1(j1_pt, j1_eta,j1_phi, j1_mass);
			ROOT::Math::PtEtaPhiMVector j2(j2_pt, j2_eta,j2_phi, j2_mass);
			return (j1 + j2).Pt();
		}
		float mt(float j1_pt, float j1_eta,float j1_phi,float j1_mass, float j2_pt, float j2_eta, float j2_phi, float j2_mass)
		{
			ROOT::Math::PtEtaPhiMVector j1(j1_pt, j1_eta,j1_phi, j1_mass);
			ROOT::Math::PtEtaPhiMVector j2(j2_pt, j2_eta,j2_phi, j2_mass);
			return (j1 + j2).Pt();
		}
		float eta(float j1_pt, float j1_eta,float j1_phi,float j1_mass, float j2_pt, float j2_eta, float j2_phi, float j2_mass)
		{
			ROOT::Math::PtEtaPhiMVector j1(j1_pt, j1_eta,j1_phi, j1_mass);
			ROOT::Math::PtEtaPhiMVector j2(j2_pt, j2_eta,j2_phi, j2_mass);
			return (j1 + j2).Eta();
		}
		float phi(float j1_pt, float j1_eta,float j1_phi,float j1_mass, float j2_pt, float j2_eta, float j2_phi, float j2_mass)
		{
			ROOT::Math::PtEtaPhiMVector j1(j1_pt, j1_eta,j1_phi, j1_mass);
			ROOT::Math::PtEtaPhiMVector j2(j2_pt, j2_eta,j2_phi, j2_mass);
			return (j1 + j2).Phi();
		}	
		float dPhi(float j1_pt, float j1_eta,float j1_phi,float j1_mass, float j2_pt, float j2_eta, float j2_phi, float j2_mass)
		{
			TLorentzVector j1;
			j1.SetPtEtaPhiM(j1_pt, j1_eta,j1_phi, j1_mass);
			TLorentzVector j2;
			j2.SetPtEtaPhiM(j2_pt, j2_eta,j2_phi, j2_mass);
			Double_t dphi = TVector2::Phi_mpi_pi(j1_phi-j2_phi);
			return dphi; 
		}
		float dR(float j1_pt, float j1_eta,float j1_phi,float j1_mass, float j2_pt, float j2_eta, float j2_phi, float j2_mass)
		{
		TLorentzVector j1;
		j1.SetPtEtaPhiM(j1_pt, j1_eta,j1_phi, j1_mass);
		TLorentzVector j2;
		j2.SetPtEtaPhiM(j2_pt, j2_eta,j2_phi, j2_mass);
		return j1.DeltaR(j2); 
		}
		float dEta(float j1_pt, float j1_eta,float j1_phi,float j1_mass, float j2_pt, float j2_eta, float j2_phi, float j2_mass)
		{
		TLorentzVector j1;
		j1.SetPtEtaPhiM(j1_pt, j1_eta,j1_phi, j1_mass);
		TLorentzVector j2;
		j2.SetPtEtaPhiM(j2_pt, j2_eta,j2_phi, j2_mass);
		return (j1_eta-j2_eta); 
		}

""")

	# Setup connection to a Dask cluster

if __name__ == '__main__':

	#myInit()
	
	#multithreading
	parser = argparse.ArgumentParser(description="A simple command-line tool")
	parser.add_argument("input_list", nargs="+",type=str, help="List of input file paths")
	parser.add_argument("output_file", help="Output file path")
	parser.add_argument("--doMC", action="store_true",help="Output file path")
	parser.add_argument("-v", "--verbose", action="store_true", help="Verbose mode")

	args = parser.parse_args()
	#myInit()
	isMC = args.doMC

	print(args.input_list)
	#frame = rd('Events','root://cms-xrd-global.cern.ch///store/mc/RunIIAutumn18NanoAODv4/ZJetsToQQ_HT400to600_qc19_4j_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/Nano14Dec2018_102X_upgrade2018_realistic_v16-v1/60000/F72274D4-EB6F-AA49-8E6E-EC263D45F2F0.root',10)
	#frame = rd('Events','/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/GluGluHToBB2024Mar05/GluGluHToBB_M-125_TuneCP5_13TeV-powheg-pythia8/crab_GluGluHToBB/240305_081723/0000/*1.root')
	frame = rd('Events',args.input_list)
	print ("dataset initialized")
	columns = frame.GetColumnNames()

	#print(columns)

	#define branches to propagate
	jet_branches = [str(item).replace("Jet_","") for item in columns if str(item).startswith('Jet_') ]

	if isMC:
		Genjet_branches = [str(item).replace("GenJet_","") for item in columns if str(item).startswith('GenJet_') ]

	dijet_branches = ["mass","pt","mt","eta","phi","dPhi","dR","dEta"]

	#muon_branches = ['Muon_inJet_charge', 'Muon_inJet_dxy', 'Muon_inJet_dxyErr', 'Muon_inJet_dz', 'Muon_inJet_dzErr', 'Muon_inJet_eta', 'Muon_inJet_fired_HLT_Mu10p5_IP3p5', 'Muon_inJet_fired_HLT_Mu12_IP6', 'Muon_inJet_fired_HLT_Mu7_IP4', 'Muon_inJet_fired_HLT_Mu8_IP3', 'Muon_inJet_fired_HLT_Mu8_IP5', 'Muon_inJet_fired_HLT_Mu8_IP6', 'Muon_inJet_fired_HLT_Mu8p5_IP3p5', 'Muon_inJet_fired_HLT_Mu9_IP4', 'Muon_inJet_fired_HLT_Mu9_IP5', 'Muon_inJet_fired_HLT_Mu9_IP6', 'Muon_inJet_ip3d', 'Muon_inJet_isGlobal', 'Muon_inJet_isPFcand', 'Muon_inJet_isTracker', 'Muon_inJet_isTriggering', 'Muon_inJet_looseId', 'Muon_inJet_mass', 'Muon_inJet_matched_dpt', 'Muon_inJet_matched_dr', 'Muon_inJet_mediumId', 'Muon_inJet_pdgId', 'Muon_inJet_pfIsoId', 'Muon_inJet_pfRelIso03_all', 'Muon_inJet_pfRelIso04_all', 'Muon_inJet_phi', 'Muon_inJet_pt', 'Muon_inJet_ptErr', 'Muon_inJet_sip3d', 'Muon_inJet_skipMuon', 'Muon_inJet_softId', 'Muon_inJet_tightId', 'Muon_inJet_tkIsoId', 'Muon_inJet_trgIdx', 'Muon_inJet_triggerIdLoose', 'Muon_inJet_vx', 'Muon_inJet_vy', 'Muon_inJet_vz']
	muon_branches =  [str(item).replace("Muon_","") for item in columns if str(item).startswith('Muon_') ]#['Muon_inJet_charge', 'Muon_inJet_dxy', 'Muon_inJet_dxyErr', 'Muon_inJet_dz', 'Muon_inJet_dzErr', 'Muon_inJet_eta', 'Muon_inJet_fired_HLT_Mu10p5_IP3p5', 'Muon_inJet_fired_HLT_Mu12_IP6', 'Muon_inJet_fired_HLT_Mu7_IP4', 'Muon_inJet_fired_HLT_Mu8_IP3', 'Muon_inJet_fired_HLT_Mu8_IP5', 'Muon_inJet_fired_HLT_Mu8_IP6', 'Muon_inJet_fired_HLT_Mu8p5_IP3p5', 'Muon_inJet_fired_HLT_Mu9_IP4', 'Muon_inJet_fired_HLT_Mu9_IP5', 'Muon_inJet_fired_HLT_Mu9_IP6', 'Muon_inJet_ip3d',  'Muon_inJet_isTriggering', 'Muon_inJet_mass', 'Muon_inJet_pfRelIso03_all', 'Muon_inJet_pfRelIso04_all', 'Muon_inJet_phi', 'Muon_inJet_pt', 'Muon_inJet_ptErr', 'Muon_inJet_sip3d', 'Muon_inJet_vx', 'Muon_inJet_vy', 'Muon_inJet_vz','Muon_inJet_looseId','Muon_inJet_softId']


	muon_branches = [str(item).replace("Muon_inJet_","") for item in muon_branches]

	f_pair = frame.Filter("nJet>0","nJet >0 ")
	f_pair = f_pair.Filter("nMuon>0","nMuons>0")
	f_pair = frame.Define("Jet_isTrigBased", "trig_requirement(Jet_nMuons,Jet_muonIdx1,Jet_muonIdx2,Muon_isTriggering)")
	f_pair = f_pair.Define("Jet_isTrigBased_size","Jet_isTrigBased.size()")
	f_pair = f_pair.Define("Jet_pt_size","Jet_pt.size()")
	f_pair = f_pair.Define("Jet_pair_idx0", "find_pair(Jet_isTrigBased,Jet_btagDeepFlavB,4,0)")
	f_pair= f_pair.Define("Jet_pair_idx1", "find_pair(Jet_isTrigBased,Jet_btagDeepFlavB,4,1)")
	f_pair = f_pair.Filter("Jet_pair_idx0!=-1","First indx of pair ")
	f_pair = f_pair.Filter("Jet_pair_idx1!=-1","Second idx of pair ")
	f_pair = f_pair.Define("Muon_inJet_trgIdx", "Jet_isTrigBased[Jet_pair_idx0]")

	f_pair = f_pair.Filter("Muon_inJet_trgIdx!=-1 && fabs(Muon_inJet_trgIdx)< 100000 && Muon_inJet_trgIdx< nMuon","trg muon makes sense")
	#idxes = f_pair.AsNumpy(["Muon_inJet_trgIdx"])
#	print(idxes["Muon_inJet_trgIdx"][idxes["Muon_inJet_trgIdx"]!=0])
	idxes = f_pair.AsNumpy(["Jet_pair_idx0"])
	#idxes_1 = f_pair.AsNumpy(["Jet_pair_idx1"])
	print(idxes)



	for b in muon_branches:
		f_pair= f_pair.Define("Muon_inJet_"+b,"float(Muon_"+b+"[Muon_inJet_trgIdx])")

	for b in jet_branches:

		#jets branches 

		f_pair= f_pair.Define("JetTrg_"+b,"Jet_"+b+"[Jet_pair_idx0]")
		f_pair= f_pair.Define("JetSel_"+b,"Jet_"+b+"[Jet_pair_idx1]")

	f_pair= f_pair.Define("JetSel_ptReg","JetSel_pt*JetSel_bReg2018")
	f_pair= f_pair.Define("JetSel_massReg","JetSel_mass*JetSel_bReg2018")
	f_pair= f_pair.Define("JetTrg_ptReg","JetTrg_pt*JetTrg_bReg2018")
	f_pair= f_pair.Define("JetTrg_massReg","JetTrg_mass*JetTrg_bReg2018")
		#dijet features 

	if isMC:
		f_pair = f_pair.Filter("JetTrg_genJetIdx!=-1 ","match JetTrg to GenJet")
		f_pair = f_pair.Filter("JetSel_genJetIdx!=-1 ","match SelTrg to GenJet")

		for b in Genjet_branches:

			f_pair= f_pair.Define("GenJetSel_"+b,"GenJet_"+b+"[JetSel_genJetIdx]")
			f_pair= f_pair.Define("GenJetTrg_"+b,"GenJet_"+b+"[JetTrg_genJetIdx]")

		f_pair = f_pair.Filter("abs(GenJetTrg_partonFlavour)==5 ","match TrgGenJet to b quark ")
		f_pair = f_pair.Filter("abs(GenJetSel_partonFlavour)==5 ","match SelGenJet to b quark")
	#raw quantities
	for b in dijet_branches:
		f_pair= f_pair.Define("DiJet_"+b,b+"(JetTrg_pt,JetTrg_eta,JetTrg_phi,JetTrg_mass,JetSel_pt,JetSel_eta,JetSel_phi,JetSel_mass)")

	#regressed quantities 
	for b in dijet_branches:
		f_pair= f_pair.Define("DiJet_"+b+"_reg",b+"(JetTrg_ptReg,JetTrg_eta,JetTrg_phi,JetTrg_massReg,JetSel_ptReg,JetSel_eta,JetSel_phi,JetSel_massReg)")


	print(f_pair.Report().Print())
	save_cols = []
	if isMC:
		save_cols =["run","event","luminosityBlock","genWeight","PV_npvs","PV_npvsGood"]
	else:
		save_cols =["run","event","luminosityBlock","PV_npvs","PV_npvsGood"]
	defined_cols = [str(item) for item in f_pair.GetDefinedColumnNames()]
	print(defined_cols)
	print(defined_cols[115:-1])
	#save_cols.extend(sorted(defined_cols[0:10]+defined_cols[135:139])) 
	save_cols.extend(sorted(defined_cols))
	#c = ROOT.TCanvas("distrdf002", "distrdf002", 800, 400)
	#h = f_pair.Histo1D(("DiJet_mass_reg","dijet_mass_reg",100,0,300),"DiJet_mass_reg")
	#h.Draw()
	#c.SaveAs("dijetMass_dask_connection.png")
	f_pair.Snapshot("Events",args.output_file,save_cols)




