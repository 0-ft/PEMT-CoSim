docker rm helics2
docker run -it --privileged -p 2223:22 -p 5952:5901 \
    --env="DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --volume="$HOME/.Xauthority:/root/.Xauthority:rw" \
    --mount type=bind,source=/home/atlas/sync/uni/year3/toshiba/PEMT-CoSim,destination=/PEMT-CoSim --name helics2 new-helics