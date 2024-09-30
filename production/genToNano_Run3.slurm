#!/bin/bash
#SBATCH --job-name=sim_GluGluHH4b_array        # Job name
#SBATCH --output=out/cms_sim_%A_%a.out      # Output file (%A is the array job ID, %a is the task ID)
#SBATCH --error=err/cms_sim_%A_%a.err       # Error file (%A is the array job ID, %a is the task ID)
#SBATCH --ntasks=1                      # Number of tasks
#SBATCH --cpus-per-task=8            # Number of CPU cores per task
#SBATCH --mem=6G                        # Memory per node
#SBATCH --time=4-00:00:00                 # Time limit hrs:min:sec
#SBATCH --partition=long           # Partition name
#SBATCH --array=1-8                 # Array range, one job per file


# Environment setup


# List of GEN files
GEN_FILES=(/pnfs/psi.ch/cms/trivcat/store/user/ratramon/tsg_ggHH4b_gen_sim/cms/store/mc/Run3Winter24wmLHEGS/GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV_powheg-pythia8/GEN-SIM/133X_mcRun3_2024_realistic_v7-v3/2820000/*.root)
OUT_PATH=/pnfs/psi.ch/cms/trivcat/store/user/ratramon/GluGluToHHTo4B_central_grid/Run3_MINI/0000/
OUT_PATH_NANO=/pnfs/psi.ch/cms/trivcat/store/user/ratramon/GluGluToHHTo4B_central_grid/Run3_NANO/0000/
#mkdir -p $OUT_PATH
# Get the file for this array task
GEN_FILE=${GEN_FILES[$SLURM_ARRAY_TASK_ID-1]}

# Processing steps
#SLURM_ARRAY_JOB_ID=1
#SLURM_ARRAY_TASK_ID=1
TMPDIR=/scratch/$USER/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
echo "preparing tempDir under" $TMPDIR
mkdir -p $TMPDIR
cp -r /t3home/ratramon/ggHBB/Btob_studies/bb_analysis/production/CMSSW_12_4_11_patch3 $TMPDIR/.
cd $TMPDIR
ls -ll



#Start from gen-sim file

echo "CMSSW_14 to include all 2024 trigger paths"
scram p CMSSW CMSSW_14_1_0_pre7
cd CMSSW_14_1_0_pre7/src
eval `scram runtime -sh`
cd ../../
echo "dir content: "
ls -ll 

#Generate and run DIGI-HLT step
cmsDriver.py step2 --python_filename Digi_step.py --eventcontent RAWSIM --pileup 2023_LHC_Simulation_12p5h_9h_hybrid2p23 --customise Configuration/DataProcessing/Utils.addMonitoring \
            --datatier GEN-SIM-RAW --fileout file:DIGI_$SLURM_ARRAY_TASK_ID.root --pileup_input dbs:/MinBias_TuneCP5_13p6TeV-pythia8/Run3Winter24GS-133X_mcRun3_2024_realistic_v7-v1/GEN-SIM \
            --conditions 140X_mcRun3_2024_realistic_v21 --step DIGI,L1,DIGI2RAW,HLT:2024v14 --nThreads 2 --geometry DB:Extended --filein file:$GEN_FILE --era Run3_2024 --mc -n -1
#
echo "dir content: "
ls -ll 
#Generate and run Reco step
cmsDriver.py step3 --python_filename Reco_step.py --eventcontent RECOSIM,AODSIM --customise Configuration/DataProcessing/Utils.addMonitoring --datatier GEN-SIM-RECO,AODSIM \
            --fileout file:RECO_$SLURM_ARRAY_TASK_ID.root --conditions 140X_mcRun3_2024_realistic_v21 --step RAW2DIGI,L1Reco,RECO,RECOSIM --nThreads 8 --geometry DB:Extended \
            --filein file:DIGI_$SLURM_ARRAY_TASK_ID.root --era Run3_2024 -n -1 --mc


#Generate and run mini step
cmsDriver.py step4 --python_filename Mini_step.py --eventcontent MINIAODSIM --customise Configuration/DataProcessing/Utils.addMonitoring --datatier MINIAODSIM \
            --fileout file:MINI_$SLURM_ARRAY_TASK_ID.root --conditions 140X_mcRun3_2024_realistic_v21 --step PAT --nThreads 8 --geometry DB:Extended \
            --filein  file:RECO_$SLURM_ARRAY_TASK_ID.root --mc -n -1 --era Run3_2024
            
cmsDriver.py step5 --python_filename Nano_step.py --eventcontent NANOAODSIM --customise Configuration/DataProcessing/Utils.addMonitoring --datatier NANOAODSIM \
            --fileout file:NANO_$SLURM_ARRAY_TASK_ID.root --conditions 140X_mcRun3_2024_realistic_v21 --step NANO --nThreads 4 --filein file:MINI_$SLURM_ARRAY_TASK_ID.root -n -1 --era Run3_2024 --mc
##
##
xrdcp -f -N $TMPDIR/MINI_$SLURM_ARRAY_TASK_ID.root root://t3dcachedb.psi.ch:1094/$OUT_PATH/.
###Generate and run nano step
##cmsDriver.py step6 --step NANO --conditions  133X_mcRun3_2024_realistic_v9  --mc \
##            --eventcontent NANOAOD --datatier NANOAOD --era Run3 --filein file:MINI_$SLURM_ARRAY_TASK_ID.root \
##            --fileout file:NANO_$SLURM_ARRAY_TASK_ID.root -n 3 --nThreads 1 --python_filename nano_crab.py
##
###cp -r /t3home/ratramon/ggHBB/Btob_studies/CMSSW_12_4_8 $TMPDIR/.
###cd $TMPDIR/CMSSW_12_4_8/src 
###eval `scram runtime -sh`
###cd PhysicsTools/BParkingNano/test
###cmsRun run_nano_cfg_bToB.py inputFiles=file:$TMPDIR/MINI_$SLURM_ARRAY_TASK_ID.root tag=$SLURM_ARRAY_TASK_ID maxEvents=-1  
###ls -ll 
xrdcp -f -N $TMPDIR/NANO_$SLURM_ARRAY_TASK_ID.root root://t3dcachedb.psi.ch:1094/$OUT_PATH_NANO/.
##rm  -rf $TMPDIR