# Using the HPC on ITU

## 1. ITU network needed

You can either physically be on ITU premises or remote access the ITU network via Forticlient.

If using Forticlient, use normal itu credentials, and when prompted enter the 4-digit pin number sent via sms. 

There is more information on Forticlient on itustudent.itu.dk

## 2. SSH connection

Then ssh onto the HPC with

`ssh frph@hpc.itu.dk`

It is a ***very*** good idea to do this through VS code (use bottom left icon and then "Connect to host...") as it makes moving files on the HPC much easier

where frph is replaced with your own credentials.

## 3. Copy file or entire directory

You can transfer single files or entire directories to the HPC. If I wanted to transfer this repository, I would use the following command in my local terminal:

`scp -r /Users/fh/documents/GitHub/2ndyearproject frph@hpc.itu.dk:`

You will then be prompted for your ITU password, and the transfer will start. 

To transfer a single file:

`scp <filepath> frph@hpc.itu.dk:`

The colon at the end is very important.

If you are connecting to the HPC using VS code you can browse through your directories with the explorer. 

The usual unix commands `cd, pwd, rm, mkdir` and can be used as well. 

## 4. Submitting a job with the HPC

Here you need to use a shell script. This script will point to the relative path of whather python script you want to run, which is why it is helpful to transfer your entire repo to the HPC. 

The command to run the `runjob.sh` script is:

`sbatch runjob.sh`

In the remote terminal. You can view the queue with `squeue`

When the job finishes the log will be written to /hpc/logs in this repo ***on the remote***, so you must manually to copy it to your local machine if needed. Similarly if your script produces a file. The file will appear in the /hpc folder in this repo.