for i in {4..7}; do
    tmux kill-session -t "gpu${i}"
done