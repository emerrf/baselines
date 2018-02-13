FROM continuumio/anaconda3

# Install Baselines
RUN conda install tensorflow numpy scipy tqdm joblib pyzmq dill cloudpickle progressbar2 mpi4py opencv
RUN pip install gym==0.9.2 zmq azure==1.0.3

# Install Anaconda's fortran compiler
RUN conda install gfortran_linux-64 gcc_linux-64

# Install gym-wind-turbine
RUN git clone https://github.com/emerrf/gym-wind-turbine.git -b tuning/reward_function /app/gym-wind-turbine
WORKDIR /app/gym-wind-turbine
RUN /bin/bash -c "source activate root" \
#&& ln -s #GFORTRAN $(dirname $GFORTRAN)/gfortran \
&& ln -s /opt/conda/bin/x86_64-conda_cos6-linux-gnu-gfortran /opt/conda/bin/gfortran \
&& ln -s /opt/conda/bin/x86_64-conda_cos6-linux-gnu-gcc /opt/conda/bin/gcc \
&& pip install -r requirements.txt

RUN conda clean --yes --all

# Transfer source
ADD . /app/baselines
WORKDIR /app/baselines

#CMD ["-np", "$(nproc)", "python", "-m", "baselines.ppo1.run_env_mpi", "--env", "WindTurbine-v0", "--num-timesteps", "9000"]
#ENTRYPOINT ["mpirun"]

ENTRYPOINT mpirun -np $(nproc) python -m baselines.ppo1.run_env_mpi --env WindTurbine-v0 --num-timesteps 9000