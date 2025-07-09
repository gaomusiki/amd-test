#!/bin/bash

export CUDA_VISIBLE_DEVICES="cpu"

SEP="========================================================================"
SUBMITS_ROOT="submits"
SCORE_FEEBACK_BRANCH_NAME="score-feedback"

CLASSROOM="nju-llm-course-classroom"
ASSIGNMENT="assignment-3-modeling-attn"

STUDENT_REPO_ROOT="$SUBMITS_ROOT"
mkdir -p "${STUDENT_REPO_ROOT}"

CLONE_REPOS=false
DO_TEST=true
PUSH_FEEBACK=false


allowed_students=""
if [ -f "allowed_students.txt" ]; then
    allowed_students=$(cat allowed_students.txt)
fi
ignored_students=""
if [ -f "ignored_students.txt" ]; then
    ignored_students=$(cat ignored_students.txt)
fi


# clone students' repos
if [[ $CLONE_REPOS == true ]]; then
    echo "$SEP"
    echo "Downloading student repos..."
    echo "$SEP"

    start_time=$(date +%s)
    (rm -rf $STUDENT_REPO_ROOT && \
    gh classroom clone student-repos --directory "${SUBMITS_ROOT}") || exit 1
    end_time=$(date +%s)
    execution_time=$((end_time - start_time))
    minutes=$((execution_time / 60))
    seconds=$((execution_time % 60))

    echo ""; echo "Done in $minutes min $seconds sec."; echo ""
fi


CNT=0
start_time=$(date +%s)
for student_repo_path in $(find ${STUDENT_REPO_ROOT} -mindepth 1 -maxdepth 1 -type d); do
    student_repo_name=$(basename "$student_repo_path")
    
    if ( [ -z "$allowed_students" ] || echo "$allowed_students" | grep -q "^$student_repo_name$" ) && ! echo "$ignored_students" | grep -q "^$student_repo_name$"; then
        # test this valid student's implementations
        if [[ $DO_TEST == true ]]; then
            echo "$SEP"
            echo "Testing for $student_repo_name"
            echo "$SEP"

            # get the latest student's commit
            cd "${student_repo_path}"

            git checkout main && \
            git pull origin main

            cd -

            # test student's implementations on all test cases
            # and write the scores to their repo
            export STUDENT_REPO_PATH="$(pwd)/${student_repo_path}"
            python test_score.py
        fi

        # construct a new branch under their remote repo to offer score feedback
        if [[ $PUSH_FEEBACK == true ]]; then
            echo ""
            echo "$SEP"
            echo "Pushing score feedback for $student_repo_name"
            echo "$SEP"

            cd "${student_repo_path}"
            git checkout main

            git pull origin main && \
            ( git checkout -b $SCORE_FEEBACK_BRANCH_NAME || \
            ( git branch -D $SCORE_FEEBACK_BRANCH_NAME && \
                (git push --force origin --delete $SCORE_FEEBACK_BRANCH_NAME || true) \
                && git checkout -b $SCORE_FEEBACK_BRANCH_NAME ) ) && \
            git add score.md && \
            git commit -m "Done test and added score feedback" && \
            (  git push -u origin $SCORE_FEEBACK_BRANCH_NAME || \
            (   git push --force origin --delete $SCORE_FEEBACK_BRANCH_NAME && \
                git push -u origin $SCORE_FEEBACK_BRANCH_NAME) )

            git checkout main
            cd -
        fi

        # increment counter
        echo ""
        CNT=$((CNT + 1))
    fi
done
end_time=$(date +%s)
execution_time=$((end_time - start_time))
minutes=$((execution_time / 60))
seconds=$((execution_time % 60))

echo "$SEP"
echo "DONE! Processed $CNT student(s) in total, costed $minutes min $seconds sec."
echo "$SEP"

