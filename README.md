# master-of-bones

Master Thesis about finding local shape variations in sheep bones.

### Install

This has been tested on a new Ubuntu 14.04 box. The requirements outside of PyPi are python2.7, Qt4, PyQt4 and VTK

Install Dependencies:  
```
sudo apt-get install git gfortran libqt4-dev libblas-dev liblapack-dev libfreetype6-dev libpng-dev python-dev python-vtk6 python-qt4 python-virtualenv
```

Clone Repo:  
```
git clone https://github.com/selaux/master-of-bones.git
```

Create and Activate Virtualenv:  
```
cd master-of-bones && virtualenv ./venv && source ./venv/bin/activate
```

Link global libraries into venv:  
```
cp -r /usr/lib/python2.7/dist-packages/PyQt4 ./venv/lib/python2.7/site-packages/
cp -r /usr/lib/python2.7/dist-packages/sip.so ./venv/lib/python2.7/site-packages/
cp -r /usr/lib/python2.7/dist-packages/vtk ./venv/lib/python2.7/site-packages/
```

Fix VTK installation as su (https://bugs.launchpad.net/ubuntu/+source/vtk6/+bug/1354127):  
```
for link in `ls /usr/lib/python2.7/dist-packages/vtk/vtk*.so`; do target=$(readlink $link).6.0; echo $link; sudo rm $link; sudo ln -s $target $link ; done 
```

Install Dependencies (this might take a while):  
```
pip install numpy && pip install scipy && pip install -r requirements.txt
```

### Important Tools

Note: The Virtualenv has to be activated before starting anything:  
```
source ./venv/bin/activate
```

Comparison Helper: Actually does the Comparison  
```
python comparison_helper.py
```

Registration Helper: Register Bones  
```
python registration_helper.py
```

Triangulation Helper: Extract bone outlines from images
```
python registration_helper.py
```

Synthetic Generation Helper: Generate Synthetic Bone Data  
```
python synthetic_generation_helper.py
```

### Contents

```thesis```: The masters thesis in source form and pdf from (dbsma.tex, dbsma.pdf)

```algorithms```: The algorithms presented in this thesis

```helpers```: Helper Functions and UI
