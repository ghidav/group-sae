for i in {0..7}; do
    tmux kill-session -t "gpu${i}"
done
