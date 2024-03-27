Repo for object management and analysis of bb final states in BParking/Bscouting datasets. Currently hosts b jet regression development for low Pt jets containing a slightly displaced muons.


- BtobPairs.py -  ntuplizer based on RDataFrame: can produce both flat ROOT tree and numpy dictionary ready to be fed in training script. Current logic: matching between b jet and BToKMuMu candidate + extract info on the triggering muon in BToKMuMu decay. TO DO: genealize to muon gen matched + having a B meson/b quark as mother or grandmother.

- train_reg.py - XGBoost regresser - very naive for the moment, regresses GenJet_pt/Jen_pt.


 TO DO:
	- move to quantile based NN: https://github.com/michelif/HHbbgg_ETH/blob/master/bregression/notebooks/trainBreg_quatile.py 
	- check presence of JER/ JEC dependent variables - they may bias the regression with thruth info
	- check training against other samples (ttbar?)
