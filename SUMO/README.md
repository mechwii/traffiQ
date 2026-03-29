# Traffiq Sumo version 

## How to install and configure your environment

1. Firstly install SUMO on your laptop (https://sumo.dlr.de)

2. Then set environment variables

```sh
export SUMO_HOME=/path/to/sumo          # Linux/Mac
set SUMO_HOME=C:\path\to\sumo           # Windows
```

3. Add SUMO tools to Python path

```sh
export PYTHONPATH=$SUMO_HOME/tools:$PYTHONPATH
```

4. Create your venv and install dependencies

```sh
python3 -m venv venv
source venv/bin/acivate # Linux/Mac
./venv/Script/Active # Windows
pip install -r requirements.txt
```

