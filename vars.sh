echo 'export TESPDIR=/root/tesp' >> ~/.bashrc
echo 'export GLPATH=$TESPDIR/lib/gridlabd:/opt/tesp/share/gridlabd' >> ~/.bashrc
echo 'export TESP_INSTALL=$TESPDIR' >> ~/.bashrc

# add TESP bin to path
echo 'export PATH=$TESPDIR/bin:$PATH' >> ~/.bashrc

# add to load library path
echo 'export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$TESPDIR/lib/gridlabd:$LD_LIBRARY_PATH' >> ~/.bashrc

# add path to energyplus
echo 'export PATH=$TESPDIR:$PATH' >> ~/.bashrc

# add path to TESP PostProcess and PreProcess
echo 'export PATH=$TESPDIR/PreProcess:$TESPDIR/PostProcess:$PATH' >> ~/.bashrc

# tesp chokes on timezone for some reason..., so unset it
echo "unset TZ"  >> ~/.bashrc