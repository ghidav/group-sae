
for i in {0..3}; do
    session_name="gpu${i}"
    script_name="gpu${i}.sh"

    # Launch a detached tmux session that directly runs
    # "CUDA_VISIBLE_DEVICES=$i ./$script_name" then waits in bash
    tmux new-session -d -s "$session_name" \
        "export CUDA_VISIBLE_DEVICES=$i; ./$script_name; bash"

done

echo
echo "All 4 sessions created! Use 'tmux ls' to see them."
echo "Attach to a session with: tmux attach -t gpuN (where N is 0..3)."
