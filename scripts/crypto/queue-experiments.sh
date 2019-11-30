CWD="$(pwd)"
cd ~
SCRATCH_DIR="$(pwd)/scratch"
cd $CWD

OPTION_GLUCOSE="glucose"
OPTION_DRAT="drat"
OPTION_DEPENDENCY="dependency"
OPTION_GR="gr"
OPTION_FLOW_CUTTER="flowcutter"

if [[ $# -ne 8 ]]; then
    echo "Usage: $0 <BEGIN_ROUNDS> <END_ROUNDS> <BEGIN_RESTRICTIONS> <END_RESTRICTIONS> <CDCL_REPOSITORY> <OUT_DIRECTORY> <OPTION> <TIMEOUT>"
    echo "Options: \"${OPTION_GLUCOSE}\", \"${OPTION_DRAT}\", \"${OPTION_DEPENDENCY}\", \"${OPTION_GR}\", \"${OPTION_FLOW_CUTTER}\""
    exit 1
fi

BEGIN_ROUNDS=$1
END_ROUNDS=$2
BEGIN_RESTRICTIONS=$3
END_RESTRICTIONS=$4
CDCL_REPOSITORY=$5
OUT_DIRECTORY="${SCRATCH_DIR}/$6"
OPTION=$7
TIMEOUT=$8

GENERATE_PROOF="true"
SHARCNET_ACCOUNT_NAME="vganesh"
SHARCNET_TIMEOUT=${TIMEOUT}
SHARCNET_MEMORY="2G"

if [[ $END_ROUNDS < $BEGIN_ROUNDS ]]; then
    echo "Cannot begin rounds after the end"
    exit 1
fi

if [[ $END_RESTRICTIONS < $BEGIN_RESTRICTIONS ]]; then
    echo "Cannot begin restrictions after the end"
    exit 1
fi

if [[ $OPTION != $OPTION_GLUCOSE ]] && [[ $OPTION != $OPTION_DRAT ]] && [[ $OPTION != $OPTION_DEPENDENCY ]] && [[ $OPTION != $OPTION_GR ]]; then
    echo "Received invalid option ${OPTION}"
    echo "Options: \"${OPTION_GLUCOSE}\", \"${OPTION_DRAT}\", \"${OPTION_DEPENDENCY}\", \"${OPTION_GR}\", \"${OPTION_FLOW_CUTTER}\""
    exit 1
fi

CRYPTO_EXEC="${CDCL_REPOSITORY}/SAT-encoding/crypto/main"
RESTRICT_EXEC="${CDCL_REPOSITORY}/scripts/crypto/restrictSHA1"
GLUCOSE_EXEC="${CDCL_REPOSITORY}/executables/glucose"
DRAT_EXEC="${CDCL_REPOSITORY}/executables/drat-trim"
GR_EXEC="${CDCL_REPOSITORY}/scripts/dependencyToGR.py"
FLOWCUTTER_EXEC="${CDCL_REPOSITORY}/executables/flow_cutter_pace17"

echo
echo "----------------------------------"
echo "CRYPTO_EXEC:     $CRYPTO_EXEC"
echo "RESTRICT_EXEC:   $RESTRICT_EXEC"
echo "GLUCOSE_EXEC:    $GLUCOSE_EXEC"
echo "DRAT_EXEC:       $DRAT_EXEC"
echo "GR_EXEC:         $GR_EXEC"
echo "FLOWCUTTER_EXEC: $FLOWCUTTER_EXEC"
echo "----------------------------------"
echo

mkdir -p $OUT_DIRECTORY

for ((i = $BEGIN_ROUNDS; i <= $END_ROUNDS; i++)); do
    OUT_SUBDIRECTORY="${OUT_DIRECTORY}/${i}_rounds"
    mkdir -p $OUT_SUBDIRECTORY

    for ((j = $BEGIN_RESTRICTIONS; j <= $END_RESTRICTIONS; j++)); do
        OUT_SUBSUBDIRECTORY="${OUT_SUBDIRECTORY}/${j}_restrictions"
        mkdir -p $OUT_SUBSUBDIRECTORY

        BASE_NAME="${i}_${j}_"
        GENERATED_CNF="${OUT_SUBSUBDIRECTORY}/${BASE_NAME}_generated.cnf"
        RESTRICTED_CNF="${OUT_SUBSUBDIRECTORY}/${BASE_NAME}_restricted.cnf"
        DRAT_PROOF="${OUT_SUBSUBDIRECTORY}/${BASE_NAME}_proof.drup"
        CORE_PROOF="${OUT_SUBSUBDIRECTORY}/${BASE_NAME}_core.drat"
        CORE_DEPENDENCY="${OUT_SUBSUBDIRECTORY}/${BASE_NAME}_core.dependency"
        DEPENDENCY_GRAPH="${OUT_SUBSUBDIRECTORY}/${BASE_NAME}_core.dependency.gr"

        if [[ $OPTION == $OPTION_GLUCOSE ]]; then
            # Generate CNF from a randomly generated SHA-1 instance
            echo "Generating CNF from SHA-1 instance with ${i} rounds and ${j} restrictions..."
            $CRYPTO_EXEC -A counter_chain -r $i --random_target --print_target |
            $CRYPTO_EXEC -A counter_chain -r $i > $GENERATED_CNF

            # Flip n random bits in the generated CNF
            $RESTRICT_EXEC "$j" "$GENERATED_CNF" "$RESTRICTED_CNF"

            # Ensure the CNF file exists
            if [[ -f $RESTRICTED_CNF ]]; then
                if [[ $GENERATE_PROOF == "true" ]]; then
                    JOB_COMMAND="${GLUCOSE_EXEC} -certified -certified-output=\"${DRAT_PROOF}\" ${RESTRICTED_CNF}"
                else
                    JOB_COMMAND="${GLUCOSE_EXEC} ${RESTRICTED_CNF}"
                fi
            else
                JOB_COMMAND=""
            fi
        elif [[ $OPTION == $OPTION_DRAT ]]; then
            # Ensure the CNF and proof files exist
            if [[ -f $RESTRICTED_CNF ]] && [[ -f $DRAT_PROOF ]]; then
                JOB_COMMAND="${DRAT_EXEC} ${RESTRICTED_CNF} ${DRAT_PROOF} -l ${CORE_PROOF}"
            else
                JOB_COMMAND=""
            fi
        elif [[ $OPTION == $OPTION_DEPENDENCY ]]; then
            # Ensure the CNF and core proof files exist
            if [[ -f $RESTRICTED_CNF ]] && [[ -f $CORE_PROOF ]]; then
                JOB_COMMAND="${DRAT_EXEC} ${RESTRICTED_CNF} ${CORE_PROOF} -r ${CORE_DEPENDENCY}"
            else
                JOB_COMMAND=""
            fi
        elif [[ $OPTION == $OPTION_GR ]]; then
            # Ensure the dependency file exists
            if [[ -f $CORE_DEPENDENCY ]]; then 
                JOB_COMMAND="python ${GR_EXEC} ${OUT_SUBSUBDIRECTORY}/ ${BASE_NAME}_core.dependency"
            else
                JOB_COMMAND=""
            fi
        elif [[ $OPTION == $OPTION_FLOW_CUTTER ]]; then
            # Ensure the gr file exists
            if [[ -f $DEPENDENCY_GRAPH ]]; then 
                JOB_COMMAND="${FLOWCUTTER_EXEC} < ${DEPENDENCY_GRAPH}"
            else
                JOB_COMMAND=""
            fi
        fi

        # Generate job file
        if [[ $JOB_COMMAND == "" ]]; then
            echo "Generating job script"
            JOB_SCRIPT="${OUT_SUBSUBDIRECTORY}/${BASE_NAME}_${OPTION}.sh"
            echo "#!/bin/bash" > $JOB_SCRIPT
            echo "#SBATCH --account=def-${SHARCNET_ACCOUNT_NAME}" >> $JOB_SCRIPT
            echo "#SBATCH --time=${SHARCNET_TIMEOUT}" >> $JOB_SCRIPT
            echo "#SBATCH --mem=${SHARCNET_MEMORY}" >> $JOB_SCRIPT
            echo "#SBATCH --job-name=${BASE_NAME}_${OPTION}" >> $JOB_SCRIPT
            echo "#SBATCH --output=${OUT_SUBSUBDIRECTORY}/${BASE_NAME}_${OPTION}.log" >> $JOB_SCRIPT

            echo "echo \"CPU information:\"" >> $JOB_SCRIPT
            echo "echo \$(lscpu)" >> $JOB_SCRIPT
            echo "echo" >> $JOB_SCRIPT
            echo "echo \"RAM information:\"" >> $JOB_SCRIPT
            echo "echo \$(free -m)" >> $JOB_SCRIPT

            echo "time ${JOB_COMMAND}" >> $JOB_SCRIPT

            # Queue job
            sbatch $JOB_SCRIPT

            # Wait between queuing jobs
            sleep 2
        fi
    done
done