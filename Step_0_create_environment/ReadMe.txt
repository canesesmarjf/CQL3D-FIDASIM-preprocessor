Follow these steps to setup the environment required to run CQL3D-FIDASIM-PREPROCESSIR:

1 - Run the following command on the terminal to create the CONDA environment:

    source ./Step_0_create_CONDA_env.sh

    To activate the new environment FIDASIM_env in the current shell, type the following:

        source ./Step_0_create_CONDA_env.sh --activate

        OR

        conda activate FIDASIM_env

    Your terminal prompt should now have (FIDASIM_env) on the left-hand side. In my computer it looks as follows:

        (FIDASIM_env) jfcm@Tanooki:~$

2 - Create environmental variable for the CQL3D-FIDASIM-PREPROCESSOR repo by adding the following to your .bashrc:

    export PREPROCESSOR_DIR=/path/to/preprocessor_repo

    Replace "/path/to/preprocessor_repo" with the actual install location for the repository.
    In my computer, I replaced it with "/home/jfcm/Repos/CQL3D-FIDASIM-preprocessor"

    Then, type the following in the command line type to make the env var available in the current shell:

    source ~/.bashrc
