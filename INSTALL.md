From a newly setup Ubuntu 22.04 installation, you may run the following sequence:

`sudo apt install git git-lfs cmake build-essential ccache ninja-build pkg-config liglib2.0-dev libpixman-1-dev cargo python3 python3-is-python curl protobuf-compiler libftdi-dev libftdi1 doxygen libsdl2-dev scons gtkwave libsndfile1-dev rsync autoconf automake texinfo libtool libsdl2-ttf-dev`

`curl "https://bootstrap.pypa.io/get-pip.py" -o "get-pip.py"`
`python get-pip.py`
`rm get-pip.py`
`export PATH=~/.local/bin:$PATH`

`pip install -e .`
`pip install argcomplete pyelftools meson`

Please also make sure to use a Rust version that is compatible with LLVM 15, like 1.63.0:

`rustup install 1.63.0`
`rustup default 1.63.0`

#In case you are working from an IIS machine, please also export the following symbols:

`export CC=gcc-9.2.0`
`export CXX=g++-9.2.0`
`export CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=/usr/pack/gcc-9.2.0-af/linux-x64/bin/gcc`

Finally, you should be able to run

`make all`

to build all Deeploy dependencies. Make sure to run

`make echo-bash`

to get instructions for setting up your environment.
