# Slurm Sample Generation- README

## Introduction
This README provides documentation for the usage of a set of script used to generate and reconstruct CMS events within the slurm batch service. The baseline doc contains description and instruction for the `genToMini.slurm` and `genToNano.slurm` scripts, along with instructions on how to set up the `cron_completed.sh` script as a cronjob.

## genToMini.slurm
The `genToMini.slurm` script is used for generating mini datasets. It sets up an array of 1000 jobs to run GEN--> MINI steps of CMS generation and reconstruction. To use this script, follow these steps:
1. Ensure that you have the necessary input files and dependencies.
2. Modify the script parameters as needed. IF the statistics needed is low,resize the nThreads parameters and reduce the memory usage (in the slurm config on top) to have access to multiple machines 
3. Submit the script to the SLURM job scheduler using the following command:
    ```bash
    sbatch genToMini.slurm
    ```

## genToNano.slurm
The `genToNano.slurm` script is used for generating nano datasets, for a custom nano production. To use this script, follow these steps:
1. Ensure that you have the necessary input files and dependencies.
2. Modify the script parameters as needed.IF the statistics needed is low,resize the nThreads parameters and reduce the memory usage (in the slurm config on top) to have access to multiple machines 
3. Submit the script to the SLURM job scheduler using the following command:
    ```bash
    sbatch genToNano.slurm
    ```

## cron_completed.sh
The `cron_completed.sh` script is used to check if a job has completed and perform additional actions - if so, it deletes the logs of the completed jobs. To set it up as a cronjob, follow these steps:
1. Open the crontab file using the following command:
    ```bash
    crontab -e
    ```
2. Add the following line to the crontab file to run the script every hour:
    ```bash
    0 * * * * /path/to/cron_completed.sh
    ```
    Replace `/path/to/cron_completed.sh` with the actual path to the script.
3. Save the crontab file and exit.

