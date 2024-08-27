#!/bin/bash
#SBATCH --job-name=sim_GluGluSpin0_M_50_array        # Job name
#SBATCH --output=out/cms_sim_%A_%a.out      # Output file (%A is the array job ID, %a is the task ID)
#SBATCH --error=err/cms_sim_%A_%a.err       # Error file (%A is the array job ID, %a is the task ID)
#SBATCH --ntasks=1                      # Number of tasks
#SBATCH --cpus-per-task=8               # Number of CPU cores per task
#SBATCH --mem=8G                        # Memory per node
#SBATCH --time=12:00:00                 # Time limit hrs:min:sec
#SBATCH --partition=standard            # Partition name
#SBATCH --array=1-1000                  # Array range, one job per file


# Environment setup


# List of GEN files
GEN_FILES=(/pnfs/psi.ch/cms/trivcat/store/user/ratramon/GluGluSpin0ToBBbar_W_1p0_M_70/RunIISummer20UL18_GEN/240815_194311/0000/*.root)
OUT_PATH=/pnfs/psi.ch/cms/trivcat/store/user/ratramon/GluGluSpin0ToBBbar_W_1p0_M_70/RunIISummer20UL18_MINI/240815_194311/0000/
OUT_PATH_NANO=/pnfs/psi.ch/cms/trivcat/store/user/ratramon/GluGluSpin0ToBBbar_W_1p0_M_70/RunIISummer20UL18_NANO/240815_194311/0000/
#mkdir -p $OUT_PATH
# Get the file for this array task
GEN_FILE=${GEN_FILES[$SLURM_ARRAY_TASK_ID-1]}

# Processing steps
TMPDIR=/scratch/$USER/${SLURM_ARRAY_TASK_ID}
echo "preparing tempDir under" $TMPDIR
mkdir -p /scratch/$USER/${SLURM_ARRAY_TASK_ID}
cd /scratch/$USER/${SLURM_ARRAY_TASK_ID} 
ls -ll

echo "SCRAM CMSSW_10_6_37 for SIM + DIGI"
scram p CMSSW_10_6_37
cd CMSSW_10_6_37/src
eval `scram runtime -sh`
cd ../..
echo "dir content: "
ls -ll 

cmsDriver.py step1 --mc --python_filename sim_cfg.py --filein file:${GEN_FILE} --fileout SIM_$SLURM_ARRAY_TASK_ID.root \
            --eventcontent RAWSIM --datatier GEN-SIM --runUnscheduled --step SIM --geometry DB:Extended -n -1 --nThreads 8 \
            --customise_commands "process.MessageLogger.cerr.FwkReport.reportEvery = 1000" \
            --conditions 106X_upgrade2018_realistic_v11_L1v1 --beamspot Realistic25ns13TeVEarly2018Collision --era Run2_2018



echo "dir content: "
ls -ll 

cmsDriver.py step2 --mc --python_filename digi_cfg.py --filein file:SIM_$SLURM_ARRAY_TASK_ID.root --fileout DIGIPremix_$SLURM_ARRAY_TASK_ID.root \
            --eventcontent PREMIXRAW --datatier GEN-SIM-DIGI --runUnscheduled --step DIGI,DATAMIX,L1,DIGI2RAW -n -1 --nThreads 8 \
            --procModifiers premix_stage2 --datamix PreMix --geometry DB:Extended  \
            --pileup_input dbs:/Neutrino_E-10_gun/RunIISummer20ULPrePremix-UL18_106X_upgrade2018_realistic_v11_L1v1-v2/PREMIX\
            --customise_commands "process.MessageLogger.cerr.FwkReport.reportEvery = 1000" \
            --conditions 106X_upgrade2018_realistic_v16_L1v1 --era Run2_2018



echo "SCRAM CMSSW_10_2_16_UL for HLT"
scram p CMSSW CMSSW_10_2_16_UL
cd CMSSW_10_2_16_UL/src
eval `scram runtime -sh`
cd ../..
echo "dir content: "
ls -ll 

cmsDriver.py step3 --mc --python_filename hlt_cfg.py  --filein file:DIGIPremix_$SLURM_ARRAY_TASK_ID.root --fileout HLT_$SLURM_ARRAY_TASK_ID.root \
            --eventcontent RAWSIM --datatier GEN-SIM-RAW --geometry DB:Extended -n -1 --nThreads 8 \
           --customise_commands 'process.source.bypassVersionCheck = cms.untracked.bool(True);process.MessageLogger.cerr.FwkReport.reportEvery = 1000' \
            --conditions 102X_upgrade2018_realistic_v15 --step HLT:2018v32 --era Run2_2018



echo "back to CMSSW_10_6_37 for RECO + MINI"
cd CMSSW_10_6_37/src
eval `scram runtime -sh`
cd ../../
echo "dir content: "
ls -ll 

cmsDriver.py step4 --mc --python_filename reco_cfg.py  --filein file:HLT_$SLURM_ARRAY_TASK_ID.root --fileout RECO_$SLURM_ARRAY_TASK_ID.root \
            --eventcontent AODSIM --runUnscheduled --datatier AODSIM --geometry DB:Extended -n -1 --nThreads 8  \
            --customise_commands "process.MessageLogger.cerr.FwkReport.reportEvery = 1000" \
            --conditions 106X_upgrade2018_realistic_v11_L1v1 --step RAW2DIGI,RECO --era Run2_2018 \


cmsDriver.py step5 --mc --python_filename mini_crab.py --filein file:RECO_$SLURM_ARRAY_TASK_ID.root --fileout MINI_$SLURM_ARRAY_TASK_ID.root \
            --eventcontent MINIAODSIM --runUnscheduled --datatier MINIAODSIM --geometry DB:Extended --step PAT -n -1 --nThreads 8 \
            --customise_commands "process.MessageLogger.cerr.FwkReport.reportEvery = 1000" \
            --conditions 106X_upgrade2018_realistic_v16_L1v1 --era Run2_2018


xrdcp -f -N $TMPDIR/MINI_$SLURM_ARRAY_TASK_ID.root root://t3dcachedb.psi.ch:1094/$OUT_PATH/.

cp -r /t3home/ratramon/ggHBB/Btob_studies/CMSSW_12_4_8 $TMPDIR/.
cd $TMPDIR/CMSSW_12_4_8/src 
eval `scram runtime -sh`
cd PhysicsTools/BParkingNano/test
cmsRun run_nano_cfg_bToB.py inputFiles=file:$TMPDIR/MINI_$SLURM_ARRAY_TASK_ID.root tag=$SLURM_ARRAY_TASK_ID maxEvents=-1 
ls -ll 
xrdcp -f -N $TMPDIR/CMSSW_12_4_8/src/PhysicsTools/BParkingNano/test/BParkingNANO_Run2_mc_$SLURM_ARRAY_TASK_ID.root root://t3dcachedb.psi.ch:1094/$OUT_PATH_NANO/.
rm  -rf /scratch/$USER/${SLURM_JOB_ID}